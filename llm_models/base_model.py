from typing import Union, Optional, Any
from logging import Logger, LoggerAdapter

class BaseModel:
    """
    各LLMモデル用型定義用の基底クラス。
    """
    model_name: str = "base_model"
    logger: Union[Logger, LoggerAdapter]
    __api_key: str
    _client: Any
    _chat_response: Any

    def __init__(
        self, 
        logger: Union[Logger, LoggerAdapter], 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None
    ) -> None:
        """
        各モデルのAPIクライアントを初期化
        
        Args:
            logger: ロガーインスタンス
            api_key: OpenRouter APIキー（未指定時は環境変数から取得）
            model_name: モデル名（未指定時は環境変数から取得）
        """
        pass

    def chat(
        self, 
        prompt: str, 
        messages: Optional[list] = None,
        **kwargs
    ) -> None:
        """
        LLMとのチャットを実行
        
        Args:
            prompt: プロンプトテキスト
            messages: メッセージ履歴（指定時はpromptを無視）
            **kwargs: 追加パラメータ（temperature, max_tokens等）
        Returns:
            ChatCompletion: OpenAI互換のレスポンスオブジェクト
        """
        pass

    def chat_response_to_text(self) -> str:
        """
        チャットレスポンスをテキストに変換
        
        Args:
            response: chat()メソッドの戻り値
        Returns:
            str: 応答テキスト
        """
        return ""

    def model_infometion(self) -> dict:
        """
        指定モデルの詳細情報をdict形式で取得
        """
        return {}

    def usage_info(self) -> dict:
        """
        トークン使用量情報を取得
        
        Args:
            response: chat()メソッドの戻り値
        Returns:
            dict: トークン使用量情報
        """
        return {}
