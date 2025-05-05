from typing import Union, Optional, cast
from decouple import config
from logging import Logger, LoggerAdapter
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

class MetaLlama4Scout:
    
    model_name: str
    logger: Union[Logger, LoggerAdapter]
    
    __api_key: str
    _client: OpenAI

    def __init__(
        self, 
        logger: Union[Logger, LoggerAdapter], 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None
    ) -> None:
        """
        Llama 4 ScoutのAPIクライアントを初期化します。
        
        Args:
            logger: ロガーインスタンス
            api_key: OpenRouter APIキー（未指定時は環境変数から取得）
            model_name: モデル名（未指定時は環境変数から取得）
        """
        self.logger = logger
        
        # APIキーの設定
        self.__api_key = api_key or str(config(
            "BROWNIE_ATELIER_ANALYZER__OPEN_ROUTER_API_KEY", 
            default=""
        ))

        # モデル名の設定
        self.model_name = model_name or str(config(
            "BROWNIE_ATELIER_ANALYZER__LLAMA_MODEL_NAME", 
            default="meta-llama/Llama-4-Scout-17B-16E-Instruct"
        ))

        # OpenAIクライアントの初期化（OpenRouter向け設定）
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.__api_key,
        )

        logger.info(
            f"Llama4ScoutOpenRouter 初期化 {self.model_name = } "
            f"api_key = {self.__api_key[:6]}...{self.__api_key[-3:]}"
        )

    def chat(
        self, 
        prompt: str, 
        messages: Optional[list] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        LLMとのチャットを実行
        
        Args:
            prompt: プロンプトテキスト
            messages: メッセージ履歴（指定時はpromptを無視）
            **kwargs: 追加パラメータ（temperature, max_tokens等）
        
        Returns:
            ChatCompletion: OpenAI互換のレスポンスオブジェクト
        """
        self.logger.info("chat 実行開始")
        
        messages = messages or [{"role": "user", "content": prompt}]
        
        return self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"provider": {"require_parameters": True}},  # OpenRouter固有設定
            **kwargs
        )
    
    def chat_response_to_text(self, response: ChatCompletion) -> str:
        """
        チャットレスポンスをテキストに変換
        
        Args:
            response: chat()メソッドの戻り値
            
        Returns:
            str: 応答テキスト
        """
        if response.choices:
            message = cast(ChatCompletionMessage, response.choices[0].message)
            return message.content or ""
        return ""

    def usage_info(self, response: ChatCompletion) -> dict:
        """
        トークン使用量情報を取得
        
        Args:
            response: chat()メソッドの戻り値
            
        Returns:
            dict: トークン使用量情報
        """
        if response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        return {}
