import sqlite3
import os
import sys

def clear_pool():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "candidate_pool.db")
    if not os.path.exists(db_path):
        print(f"找不到資料庫檔案: {db_path}，沒有需要清除的侯選資料。")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查詢目前有多少筆
        cursor.execute("SELECT COUNT(*) FROM candidate_pool")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("目前 candidate_pool 候選池內沒有任何記錄。")
        else:
            # 清空表格
            cursor.execute("DELETE FROM candidate_pool")
            conn.commit()
            print(f"成功清空候選池！共刪除 {count} 筆孤立節點記錄。")
        
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"資料庫操作失敗，可能表格尚未建立: {e}")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")

if __name__ == "__main__":
    clear_pool()
