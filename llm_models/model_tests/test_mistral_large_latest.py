# test_mistral_large_latest.py

import logging
from dotenv import load_dotenv
from decouple import config

from BrownieAtelierAnalyzer.llm_models.mistral.mistral_large_latest import MistralLargeLatest

def main() -> None:
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        # クラスの初期化
        mistral = MistralLargeLatest(
            logger=logger,
            api_key=str(config("BROWNIE_ATELIER_ANALYZER__MISTRAL_API_KEY")),
            model_name=str(config("BROWNIE_ATELIER_ANALYZER__MISTRAL_MODEL_NAME"))
        )

        # 基本チャットテスト
        mistral.chat(
            "量子コンピュータを小学生でも分かるように説明してください"
        )
        response_text = mistral.chat_response_to_text()
        print("\n=== 基本レスポンス ===")
        print(response_text)

        # 使用量情報取得テスト
        print("\n=== トークン使用量 ===")
        mistral.usage_info()  # ログに出力される

        # モデル情報取得テスト
        model_info = mistral.model_infomation()
        print("\n=== モデル情報 ===")
        for k, v in model_info.items():
            print(f"{k}: {v}")

        # メッセージ履歴テスト
        messages = [
            {"role": "system", "content": "あなたは優秀な科学解説者です"},
            {"role": "user", "content": "量子もつれとは何ですか？"}
        ]
        mistral.chat(
            prompt="",  # promptは空でもよい
            messages=messages
        )
        print("\n=== メッセージ履歴を使ったレスポンス ===")
        print(mistral.chat_response_to_text())

        # 異常系テスト（無効なモデル名）
        try:
            invalid_mistral = MistralLargeLatest(
                logger=logger,
                api_key=str(config("BROWNIE_ATELIER_ANALYZER__MISTRAL_API_KEY", default="")),
                model_name="invalid-model-name"
            )
            invalid_mistral.chat("テストメッセージ")
            print("\n=== 異常系テスト結果 ===")
            print("エラーが発生しませんでした（想定外）")
        except Exception as e:
            print("\n=== 異常系テスト結果 ===")
            print(f"想定通りのエラーを検出: {str(e)}")

    except Exception as e:
        logger.exception(f"テスト実行中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    load_dotenv()  # .envファイルから環境変数を読み込み
    main()
