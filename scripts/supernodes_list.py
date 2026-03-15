# -*- coding: utf-8 -*-
import sys
import os
import logging
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j_client import neo4j_db
from core.qdrant_client import qdrant_db
from core.llm_client import llm_client
from qdrant_client.http.models import PointStruct

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SuperNodes_Seeder")

# 50+ 寵物醫療、行為、照護的精確大主題 (Ontology Dictionary)
PREDEFINED_SUPERNODES = [
    # 感染與寄生蟲 (Infections & Parasites)
    "體外寄生蟲感染 (如跳蚤、壁蝨、耳疥蟲)",
    "體內寄生蟲感染 (如蛔蟲、絛蟲、心絲蟲)",
    "犬傳染性疾病 (如犬小病毒、犬瘟熱、狂犬病)",
    "貓傳染性疾病 (如貓白血病FeLV、貓愛滋FIV、貓傳腹FIP)",
    "黴菌與皮癬菌感染",
    
    # 皮膚與耳科 (Dermatology & Otology)
    "異位性皮膚炎與過敏",
    "細菌性膿皮症",
    "脫毛與毛髮異常",
    "耳炎與耳道清潔保健",
    "趾間炎與足部護理",
    
    # 消化系統 (Gastroenterology)
    "急性腸胃炎",
    "慢性腹瀉與腸道吸收不良",
    "胰臟炎",
    "胃扭轉與消化道阻塞",
    "食物過敏與腸胃道處方飲食",
    
    # 泌尿與腎臟 (Urology & Nephrology)
    "貓下泌尿道症候群 (FLUTD)",
    "犬貓慢性腎衰竭 (CKD)",
    "急性腎損傷 (AKI)",
    "膀胱炎與泌尿道結石",
    "尿失禁與排尿異常行為",
    
    # 心血管與呼吸 (Cardiology & Pulmonology)
    "犬慢性瓣膜性心臟病",
    "貓肥厚性心肌病 (HCM)",
    "氣喘與支氣管炎",
    "短吻犬呼吸道症候群",
    "心血管支持與處方飲食",
    
    # 內分泌與代謝 (Endocrinology & Metabolism)
    "犬貓糖尿病",
    "甲狀腺機能亢進 (貓常見)",
    "甲狀腺機能低下 (狗常見)",
    "腎上腺皮質機能亢進 (庫興氏症)",
    "肥胖與體重管理計畫",
    
    # 骨骼關節與神經 (Orthopedics & Neurology)
    "退化性關節炎 (OA)",
    "髖關節發育不良 (CHD)",
    "膝關節異位 (Patellar Luxation)",
    "椎間盤疾病 (IVDD)",
    "癲癇與神經系統異常",
    
    # 眼科與口腔 (Ophthalmology & Dentistry)
    "白內障與核硬化",
    "青光眼與角膜潰瘍",
    "乾眼症 (KCS)",
    "牙周病與牙結石",
    "貓慢性口炎 (FCGS)",
    
    # 腫瘤與免疫 (Oncology & Immunology)
    "惡性腫瘤 (癌症)",
    "良性皮膚與脂肪腫瘤",
    "免疫介導性溶血性貧血 (IMHA)",
    "淋巴瘤",
    "化療與安寧照護",
    
    # 行為、情緒與外出安全 (Behavior, Emotion & Environmental Safety)
    "分離焦慮症",
    "攻擊行為與社會化訓練",
    "過度吠叫與強迫行為",
    "居家環境毒素 (如誤食植物、百合花、木醣醇)",
    "中暑與熱衰竭預防",
    
    # 日常保養與高齡照護 (Preventive Care & Senior Care)
    "核心疫苗與預防針接種計畫",
    "絕育手術與術後照護",
    "高齡犬貓認知障礙 (失智症)",
    "高齡犬貓關節與視力退化輔助",
    "日常營養保健品 (Omega-3, 益生菌, 關節軟骨素等)"
]

def seed_supernodes():
    """Ingests the predefined 50+ SuperNodes into Neo4j and Qdrant."""
    logger.info(f"Starting seeding of {len(PREDEFINED_SUPERNODES)} predefined SuperNodes...")
    graph = neo4j_db.get_graph()
    embeddings_model = llm_client.get_embeddings()
    collection_name = "pet_light_rag"
    
    # Ensure collection exists before we try to upsert
    from qdrant_client.http.models import Distance, VectorParams
    try:
        qdrant_db.client.get_collection(collection_name)
    except Exception:
        logger.info(f"Creating Qdrant collection {collection_name}...")
        qdrant_db.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
    # Embed all in one batch
    logger.info("Generating embeddings for SuperNodes...")
    vectors = embeddings_model.embed_documents(PREDEFINED_SUPERNODES)
    
    points = []
    
    # Iterate and write to Neo4j & Qdrant points
    for idx, (sn_name, vec) in enumerate(zip(PREDEFINED_SUPERNODES, vectors)):
        # Write Neo4j
        graph.query("""
            MERGE (s:SuperNode {name: $name})
            ON CREATE SET s.source = 'predefined', s.created_at = timestamp()
        """, {"name": sn_name})
        
        # Prepare Qdrant Point
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=vec,
            payload={"text": sn_name, "type": "supernode", "category": "PREDEFINED"}
        ))
        
    # Write Qdrant
    if points:
        try:
            qdrant_db.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} SuperNodes into Qdrant '{collection_name}' collection.")
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant: {e}")
            
    logger.info("✅ SuperNodes predefined seeding complete!")

if __name__ == "__main__":
    seed_supernodes()
