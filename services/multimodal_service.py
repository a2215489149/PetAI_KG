import json
import logging
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings

logger = logging.getLogger(__name__)

class PetObservation(BaseModel):
    """寵物畫面（圖片或影片）分析結果的結構化輸出"""
    species: str = Field(description="動物種類（如：狗、貓、兔子、鳥類等）")
    breed_guess: str = Field(description="推測的品種（如：柴犬、波斯貓、米克斯等），無法判斷則填「無法判斷」")
    appearance: str = Field(description="外觀描述：毛色、體型大小、年齡推測（幼年/成年/老年）")
    health_status: str = Field(description="肉眼可見的健康狀態：精神是否良好、毛髮光澤、是否有明顯傷口、皮膚異常、眼睛鼻子是否有分泌物等")
    injury_or_abnormality: str = Field(description="是否有受傷或異常：跛行、腫脹、流血、脫毛區域等。沒有則填「未觀察到明顯異常」")
    clothing_accessories: str = Field(description="是否穿著衣物或配件：衣服顏色/樣式、項圈、牽繩、帽子等。沒有則填「無」")
    posture_behavior: str = Field(description="姿態與行為：站立/趴著/蜷縮、表情是否放鬆、是否有焦慮或疼痛跡象")
    environment: str = Field(description="環境觀察：室內/室外、是否在籠子裡、周圍是否有危險物品")
    overall_summary: str = Field(description="綜合一段話描述這個畫面中的寵物狀況，使用繁體中文，像是觀察筆記")


async def multimodal_inference(
    media_base64: str, 
    context: str, 
    media_type: str = "image"
) -> Optional[Dict[str, Any]]:
    """
    Sends the media (image or video) to Gemini 2.5 Flash for detailed pet observation.
    Returns structured JSON with comprehensive visual analysis.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2,
            max_tokens=1024
        )
        
        media_label = "圖片" if media_type == "image" else "影片"
        prompt = f"""你是一位專業的寵物觀察師與獸醫助理。請仔細觀察這個{media_label}，並提供完整詳細的寵物觀察報告。

請注意以下重點：
1. 仔細觀察寵物的外觀特徵（毛色、體型、品種推測）
2. 評估肉眼可見的健康狀態（精神狀態、毛髮光澤、是否有傷口或皮膚異常）
3. 注意是否有受傷或身體異常
4. 描述寵物是否穿著衣物或配件
5. 觀察寵物的姿態/行為表現/互動對象
6. 描述周圍環境
{"7. 如果是影片，請特別注意寵物的動態行為：走路姿態、呼吸頻率、是否有異常動作（如反覆舔舐、跛行、搖頭等）" if media_type == "video" else ""}

飼主提供的額外訊息/背景：{context}

所有輸出必須使用繁體中文。請嚴格按照指定的 JSON 格式回傳。"""

        mime_type = "image/jpeg" if media_type == "image" else "video/mp4"
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:{mime_type};base64,{media_base64}"}
                }
            ]
        )
        
        structured_llm = llm.with_structured_output(PetObservation)
        result = structured_llm.invoke([message])
        obs_dict = result.model_dump()

        # ===== Clean structured logging =====
        logger.info("=" * 60)
        logger.info(f"[Multimodal] Gemini {media_label} 分析結果")
        logger.info("=" * 60)
        logger.info(f"種類: {obs_dict.get('species')}")
        logger.info(f"品種: {obs_dict.get('breed_guess')}")
        logger.info(f"健康: {obs_dict.get('health_status')}")
        logger.info(f"摘要: {obs_dict.get('overall_summary')}")
        logger.info("=" * 60)
        
        return obs_dict
        
    except Exception as e:
        logger.error(f"Multimodal inference failed ({media_type}): {e}")
        return None
