import requests
import json
import sys

# 本地測試 URL
BASE_URL = "http://localhost:8000"

# 如果API已部署，可以替換為線上URL
# BASE_URL = "https://your-deployed-api.vercel.app"

def test_predict_single():
    """測試單個肽預測"""
    url = f"{BASE_URL}/api/predict/"
    headers = {"Content-Type": "application/json"}
    data = {"sequence": "AAKIILNPKFR"}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        
        print("\n=== 單個肽預測結果 ===")
        print(f"序列: {result['sequence']}")
        print(f"預測 HC5: {result['predicted_HC5']}")
        print(f"預測 HC10: {result['predicted_HC10']}")
        print(f"預測 HC50: {result['predicted_HC50']}")
        
        return True
    except Exception as e:
        print(f"預測單個肽時出錯: {str(e)}")
        return False

def test_predict_batch():
    """測試批量肽預測"""
    url = f"{BASE_URL}/api/predict/batch/"
    headers = {"Content-Type": "application/json"}
    data = {"sequences": ["AAKIILNPKFR", "FLPILASLAAKFGPKLFCLVTKKC", "ALWKTLLKKVLKAAAKAALNAVLVGANA"]}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        results = response.json()
        
        print("\n=== 批量肽預測結果 ===")
        for result in results:
            print(f"序列: {result['sequence']}")
            print(f"預測 HC5: {result['predicted_HC5']}")
            print(f"預測 HC10: {result['predicted_HC10']}")
            print(f"預測 HC50: {result['predicted_HC50']}")
            print("---")
        
        return True
    except Exception as e:
        print(f"批量預測肽時出錯: {str(e)}")
        return False

def test_predict_fasta():
    """測試FASTA文件上傳預測"""
    url = f"{BASE_URL}/api/predict/fasta/"
    
    try:
        with open("example_input.fasta", "rb") as f:
            files = {"file": ("example_input.fasta", f, "application/octet-stream")}
            response = requests.post(url, files=files)
            response.raise_for_status()
            results = response.json()
            
            print(f"\n=== FASTA文件預測結果 (共 {len(results)} 個序列) ===")
            for i, result in enumerate(results[:3]):  # 只顯示前3個結果
                print(f"序列 {i+1}: {result['sequence'][:20]}..." if len(result['sequence']) > 20 else result['sequence'])
                print(f"預測 HC5: {result['predicted_HC5']}")
                print(f"預測 HC10: {result['predicted_HC10']}")
                print(f"預測 HC50: {result['predicted_HC50']}")
                print("---")
            
            if len(results) > 3:
                print(f"... 還有 {len(results) - 3} 個結果未顯示")
            
            return True
    except Exception as e:
        print(f"上傳FASTA文件進行預測時出錯: {str(e)}")
        return False

if __name__ == "__main__":
    print("開始測試 BERT-HemoPep60 API...")
    
    # 根據命令行參數選擇測試
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "single":
            test_predict_single()
        elif test_type == "batch":
            test_predict_batch()
        elif test_type == "fasta":
            test_predict_fasta()
        else:
            print("未知的測試類型。請使用 'single', 'batch', 或 'fasta'")
    else:
        # 默認運行所有測試
        print("運行所有測試...")
        test_predict_single()
        test_predict_batch()
        test_predict_fasta()
    
    print("\n測試完成!") 