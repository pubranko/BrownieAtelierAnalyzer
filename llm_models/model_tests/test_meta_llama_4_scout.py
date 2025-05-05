# test_llama4_scout.py
import logging
from typing import Any
from dotenv import load_dotenv
from openai import APIError
from decouple import config

# テスト対象クラスのインポート
# from your_module import Llama4ScoutOpenRouter
from BrownieAtelierAnalyzer.llm_models.open_router.meta_llama_4_scout import MetaLlama4Scout

def main() -> None:
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        # クラスの初期化
        llama = MetaLlama4Scout(
            logger=logger,
            api_key=str(config("BROWNIE_ATELIER_ANALYZER__OPEN_ROUTER_API_KEY", default="")),  # 実際のAPIキーに置き換え
            # model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct"
            model_name=str(config("BROWNIE_ATELIER_ANALYZER__OPEN_ROUTER_MODEL_NAME", default="")), # "meta-llama/llama-4-scout:free",
        )

        # 基本チャットテスト
        response = llama.chat(
            "量子コンピューティングを小学生でもわかるように説明してください",
            temperature=0.5,
            max_tokens=300
        )
        
        # レスポンス変換テスト
        response_text = llama.chat_response_to_text(response)
        print("\n=== 基本レスポンス ===")
        print(response_text)

        # 使用量情報取得テスト
        usage = llama.usage_info(response)
        print("\n=== トークン使用量 ===")
        print(f"入力トークン: {usage['prompt_tokens']}")
        print(f"出力トークン: {usage['completion_tokens']}")
        print(f"合計トークン: {usage['total_tokens']}")

        # メッセージ履歴テスト
        messages = [
            {"role": "system", "content": "あなたは優秀な科学解説者です"},
            {"role": "user", "content": "量子もつれとは何ですか？"}
        ]
        history_response = llama.chat(
            prompt="",  # 空文字でもよい
            messages=messages
        )
        print("\n=== メッセージ履歴を使ったレスポンス ===")
        print(llama.chat_response_to_text(history_response))

        # 異常系テスト（無効なモデル名）
        try:
            invalid_llama = MetaLlama4Scout(
                logger=logger,
                model_name="invalid-model-name"
            )
            invalid_llama.chat("テストメッセージ")
        except APIError as e:
            print("\n=== 異常系テスト結果 ===")
            print(f"想定通りのエラーを検出: {e.message}")

    except Exception as e:
        logger.exception(f"テスト実行中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    load_dotenv()  # .envファイルから環境変数を読み込み
    main()
