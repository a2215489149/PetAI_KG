import json
import asyncio
import logging
import base64
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage, VideoMessage
from fastapi import Request, BackgroundTasks

from config.settings import settings
from services.rag_service import rag_pipeline
from services.multimodal_service import multimodal_inference
from workers.anchoring_task import process_anchoring
from core.redis_client import redis_db

logger = logging.getLogger(__name__)

# Redis key templates
IMAGE_LOCK_KEY = "image_lock:{session_id}"      # Lock while Gemini is analyzing
IMAGE_OBS_KEY = "image_obs:{session_id}"         # Observation result cache (10 min)

IMAGE_LOCK_TTL = 60       # Lock auto-expires after 60s (safety net)
IMAGE_OBS_TTL = 600       # Observation result persists for 10 minutes
IMAGE_WAIT_MAX = 30       # Max seconds to wait for image analysis
IMAGE_WAIT_INTERVAL = 1   # Poll every 1 second

CONTEXT_MAX_MESSAGES = 20  # Only keep the latest 20 messages in conversation history


def _trim_context(context: str, max_messages: int = CONTEXT_MAX_MESSAGES) -> str:
    """Trim conversation context to keep only the most recent N messages.
    Each 'User:' or 'AI:' line counts as one message.
    Image observations (stored on a separate Redis key) are not affected.
    """
    lines = context.strip().split("\n")
    # Identify message lines (start with 'User:' or 'AI:')
    message_indices = [i for i, line in enumerate(lines) if line.startswith("User:") or line.startswith("AI:")]
    
    if len(message_indices) <= max_messages:
        return context
    
    # Keep only the last N message lines
    cutoff_index = message_indices[-max_messages]
    trimmed = "\n".join(lines[cutoff_index:])
    return trimmed


class LineBotIntegration:
    def __init__(self):
        self.line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
        self.handler = WebhookHandler(settings.LINE_CHANNEL_SECRET)
        self.background_tasks = None
        self._pending_events = []

        @self.handler.add(MessageEvent, message=(TextMessage, ImageMessage, VideoMessage))
        def handle_message(event: MessageEvent):
            # Just collect events; actual processing happens in async context
            self._pending_events.append(event)

    async def handle_request(self, request: Request, background_tasks: BackgroundTasks):
        """Validates LINE signature, collects events, then processes them in async context."""
        self.background_tasks = background_tasks
        self._pending_events = []
        signature = request.headers.get("X-Line-Signature", "")
        body = await request.body()
        body_decoded = body.decode("utf-8")

        try:
            self.handler.handle(body_decoded, signature)
        except InvalidSignatureError:
            logger.warning("Invalid LINE signature.")
            return {"status": "error", "message": "Invalid signature"}
        except Exception as e:
            logger.error(f"Error handling LINE request: {e}")
            return {"status": "error", "message": "Server error"}

        # Process collected events in the existing FastAPI async context
        for event in self._pending_events:
            await self._process_event_async(event)

        return {"status": "ok"}

    async def _process_event_async(self, event: MessageEvent):
        """
        Processes a LINE message event fully within the FastAPI async context.
        Uses session_id (Group > Room > User) to ensure shared context in group chats.
        """
        session_id = self._get_session_id(event)
        user_id = event.source.user_id
        reply_token = event.reply_token
        lock_key = IMAGE_LOCK_KEY.format(session_id=session_id)
        obs_key = IMAGE_OBS_KEY.format(session_id=session_id)

        # ===== Clean structured logging =====
        logger.info("=" * 60)
        logger.info(f"[{event.source.type.upper()}] 開始處理事件")
        logger.info(f"Session ID: {session_id[:12]}...")
        logger.info(f"User ID:    {user_id[:12]}...")
        logger.info("-" * 40)

        # ========== IMAGE MESSAGE ==========
        if isinstance(event.message, ImageMessage):
            await self._handle_media_message(event, session_id, reply_token, lock_key, obs_key, media_type="image")
            return

        # ========== VIDEO MESSAGE ==========
        if isinstance(event.message, VideoMessage):
            await self._handle_media_message(event, session_id, reply_token, lock_key, obs_key, media_type="video")
            return

        # ========== TEXT MESSAGE ==========
        if isinstance(event.message, TextMessage):
            await self._handle_text_message(event, session_id, reply_token, lock_key, obs_key)
            return

    def _get_session_id(self, event: MessageEvent) -> str:
        """Resolve the session ID: Priority is Group ID > Room ID > User ID."""
        source = event.source
        if source.type == "group":
            return source.group_id
        if source.type == "room":
            return source.room_id
        return source.user_id

    async def _handle_media_message(self, event, session_id, reply_token, lock_key, obs_key, media_type="image"):
        """Handle image/video: set session-level lock → Gemini analysis → cache result → unlock."""
        # Set processing lock
        redis_db.set(lock_key, "processing", ex=IMAGE_LOCK_TTL)
        media_label = "image" if media_type == "image" else "video"
        logger.info(f"[{media_label.capitalize()}] Processing for session {session_id[:8]}... Lock set.")

        try:
            # Download content from LINE
            message_content = self.line_bot_api.get_message_content(event.message.id)
            media_bytes = b"".join([chunk for chunk in message_content.iter_content()])
            media_base64 = base64.b64encode(media_bytes).decode('utf-8')

            # Analyze with Gemini (Flash)
            prompt_context = "對話成員傳送了一個影片，請觀察動作細節" if media_type == "video" else "對話成員傳送了一張寵物圖片，請仔細觀察"
            observation = await multimodal_inference(media_base64, prompt_context, media_type=media_type)

            if observation:
                # Cache observation in Redis for 10 minutes
                obs_json = json.dumps(observation, ensure_ascii=False)
                redis_db.set(obs_key, obs_json, ex=IMAGE_OBS_TTL)
                logger.info(f"[{media_label.capitalize()}] Observation cached for session {session_id[:8]}... (TTL: {IMAGE_OBS_TTL}s)")
            else:
                logger.warning(f"[{media_label.capitalize()}] No observation returned for session {session_id[:8]}")

        except Exception as e:
            logger.error(f"[{media_label.capitalize()}] Failed to process for session {session_id[:8]}: {e}")
        finally:
            # Always release lock
            redis_db.delete(lock_key)
            logger.info(f"[{media_label.capitalize()}] Lock released for session {session_id[:8]}...")

        # No immediate reply for media to avoid double-replying. 
        # The result is cached in Redis and injected when the next text message arrives.

        # Save to conversation history
        context_key = f"context:{session_id}"
        previous_context = redis_db.get(context_key) or ""
        type_str = "圖片" if media_type == "image" else "影片"
        updated = f"{previous_context}\n[{type_str}訊息已處理並存入對話背景]"
        redis_db.set(context_key, _trim_context(updated), ex=3600)

    async def _handle_text_message(self, event, session_id, reply_token, lock_key, obs_key):
        """Handle text: check relevance → wait for pending image → inject observation → RAG pipeline."""
        user_text = event.message.text
        source_type = event.source.type
        is_group = source_type in ["group", "room"]

        # --- Group chat: check if message is pet-related QUESTION ---
        if is_group:
            is_relevant = await self._is_pet_related(user_text)
            if not is_relevant:
                logger.info(f"[{source_type.capitalize()}] Non-pet question from {session_id[:8]}: '{user_text[:20]}...' → Skip reply, run anchoring only.")
                # Still save to context (shared across group)
                context_key = f"context:{session_id}"
                previous_context = redis_db.get(context_key) or ""
                redis_db.set(context_key, _trim_context(f"{previous_context}\nUser: {user_text}"), ex=3600)
                # Still run background anchoring for triple extraction
                if self.background_tasks:
                    self.background_tasks.add_task(
                        process_anchoring,
                        user_id=event.source.user_id, # Use individual user_id
                        text=user_text,
                        ai_response=""
                    )
                return

        # --- Wait for pending image analysis if lock exists (Shared in Group) ---
        if redis_db.exists(lock_key):
            logger.info(f"[Text] Image still processing in session {session_id[:8]}... Waiting up to {IMAGE_WAIT_MAX}s.")
            # Notify members via push_message (to current session: group/room/user)
            # Note: For groups/rooms, push_message works if the bot has permission
            try:
                dest_id = session_id  # Push to group/room if applicable
                self.line_bot_api.push_message(
                    dest_id,
                    TextSendMessage(text="🔍 發現有人剛傳送過圖片/影片，我正在努力解析中，請稍候片刻我再一併回答...")
                )
            except Exception as e:
                logger.warning(f"[Text] Failed to send waiting notification to {session_id[:8]}: {e}")

            waited = 0
            while redis_db.exists(lock_key) and waited < IMAGE_WAIT_MAX:
                await asyncio.sleep(IMAGE_WAIT_INTERVAL)
                waited += IMAGE_WAIT_INTERVAL
            if waited >= IMAGE_WAIT_MAX:
                logger.warning(f"[Text] Image wait timed out for session {session_id[:8]}... Proceeding without media.")

        # --- Retrieve cached image observation (Shared in Group) ---
        image_observation_str = redis_db.get(obs_key)
        image_context_injection = ""
        if image_observation_str:
            try:
                obs_data = json.loads(image_observation_str)
                image_context_injection = (
                    f"[最近的畫面內容分析]\n"
                    f"種類: {obs_data.get('species', '未知')}\n"
                    f"品種: {obs_data.get('breed_guess', '未知')}\n"
                    f"外觀: {obs_data.get('appearance', '')}\n"
                    f"健康狀態: {obs_data.get('health_status', '')}\n"
                    f"受傷/異常: {obs_data.get('injury_or_abnormality', '')}\n"
                    f"穿著配件: {obs_data.get('clothing_accessories', '')}\n"
                    f"姿態行為: {obs_data.get('posture_behavior', '')}\n"
                    f"環境: {obs_data.get('environment', '')}\n"
                    f"總覽: {obs_data.get('overall_summary', '')}\n"
                )
                logger.info(f"[Text] Injecting shared image observation for session {session_id[:8]}...")
            except json.JSONDecodeError:
                image_context_injection = f"[最近的畫面觀察: {image_observation_str}]"

        # --- Build context (Shared in Group) ---
        context_key = f"context:{session_id}"
        previous_context = redis_db.get(context_key) or ""

        context_for_rag = previous_context
        if image_context_injection:
            context_for_rag = image_context_injection + "\n" + context_for_rag

        updated_context = f"{previous_context}\nUser: {user_text}"
        redis_db.set(context_key, _trim_context(updated_context), ex=3600)

        # --- RAG Pipeline ---
        try:
            # Important: We use session_id for current context/obs, 
            # but original user_id for knowledge retrieval/storage
            response_data = await rag_pipeline.ainvoke({
                "query": user_text,
                "image_base64": None,
                "context": context_for_rag,
                "user_id": event.source.user_id  # Use individual user_id
            })
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            response_data = {"final_answer": "抱歉，系統目前無法回應。"}

        final_answer = response_data.get("final_answer", "抱歉，系統目前無法回應。")

        # Reply to the specific user/thread that asked
        self.line_bot_api.reply_message(reply_token, TextSendMessage(text=final_answer))

        # Save session history
        redis_db.set(context_key, _trim_context(f"{updated_context}\nAI: {final_answer}"), ex=3600)

        # Background anchoring (using user_id to keep knowledge personal)
        if self.background_tasks:
            self.background_tasks.add_task(
                process_anchoring,
                user_id=event.source.user_id, # Use individual user_id
                text=user_text,
                ai_response=final_answer
            )

    async def _is_pet_related(self, text: str) -> bool:
        """Quick LLM classification: is this message a pet-related QUESTION that needs a response?"""
        from core.llm_client import llm_client
        try:
            llm = llm_client.get_llm()
            result = llm.invoke(
                f"判斷以下訊息是否為「需要回覆的寵物相關提問或求助」。"
                f"例如詢問寵物健康、飼養建議、醫療問題等算「是」，必須要是一個「問題」。"
                f"單純閒聊、打招呼、分享日常（即使提到寵物）一樣都算「否」。"
                f"只回答「是」或「否」，不要解釋。\n\n訊息：{text}"
            )
            answer = result.content.strip()
            logger.info(f"[Relevance] '{text[:30]}...' → {answer}")
            return "是" in answer
        except Exception as e:
            logger.warning(f"[Relevance] Classification failed: {e}. Defaulting to relevant.")
            return True  # 分類失敗時預設為相關，避免漏回覆


line_bot_app = LineBotIntegration()
