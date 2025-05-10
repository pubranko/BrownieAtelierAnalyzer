from typing import Union, cast
from typing import Optional
from decouple import config
from logging import Logger, LoggerAdapter
from mistralai import Mistral
from mistralai.models import ChatCompletionResponse, ChatCompletionChoice
from mistralai.models.basemodelcard import BaseModelCard
from mistralai.models.usageinfo import UsageInfo
from BrownieAtelierAnalyzer.llm_models.base_model import BaseModel

"""
apiドキュメント
    https://docs.mistral.ai/
apiサイト (api-key等の発行、)
    https://console.mistral.ai/home
api
    https://docs.mistral.ai/api/

"""

class MistralLargeLatest(BaseModel):
    
    model_name: str = "mistral-large-latest"
    _client: Mistral
    _chat_response: ChatCompletionResponse

    def __init__(self, logger: Union[Logger, LoggerAdapter], api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """
            Mistral Large のAPIクライアントを初期化します。
        Args:
            logger (Union[Logger, LoggerAdapter]): 
            api_key (Optional[str], optional): 
            model_name (Optional[str], optional): 
        """
        self.logger = logger
        
        if api_key:
            self.__api_key:str = api_key
        else:
            self.__api_key:str = str(config("BROWNIE_ATELIER_ANALYZER__MISTRAL_API_KEY", default=""))

        if model_name:
            self.model_name:str = model_name
        else:
            self.model_name:str = str(config("BROWNIE_ATELIER_ANALYZER__MISTRAL_MODEL_NAME", default=""))

        self._client = Mistral(api_key=self.__api_key)

        logger.info(f"MistralLargeLatest 初期化 {self.model_name = } api_key = {self.__api_key[:6]}...{self.__api_key[-3:]}")

    def chat(self, prompt:str, messages: Optional[list] = None) -> None:
        """
        llmとのチャットを実行する

        Args:
            prompt (str): LLMへ送信するチャットのテキストを指定する。messagesが指定されている場合は無視される。
            messages (list[dict[str, str]]): LLMへ送信するチャットの履歴を含むメッセージを指定する。
        """
        self.logger.info(f"chat_response_to_text 開始")
        if messages:
            wk_messages: list = messages
        else:
            wk_messages: list = [
                {"role": "user", "content": prompt}
            ]
        
        self._chat_response = self._client.chat.complete(
            model=self.model_name,
            messages=wk_messages,
        )
        
    def chat_response_to_text(self) -> str:
        """
        chat_responseの内容をテキストに変換する
        """
        chat_response_choices: Optional[list[ChatCompletionChoice]] = self._chat_response.choices
        if chat_response_choices:
            return str(chat_response_choices[0].message.content)
        else:
            return ""
    
    def model_infomation(self) -> dict:
        """
        指定モデルの詳細情報をdict形式で取得
        Returns:
            dict: モデルに関する情報
            （参考例）
            {
                'id': 'mistral-large-2411', 'capabilities': ModelCapabilities(completion_chat=True, 
                completion_fim=False, function_calling=True, fine_tuning=True, vision=False), 
                'object': 'model', 'created': 1745484085, 'owned_by': 'mistralai', 'name': 'mistral-large-2411', 
                'description': 'Official mistral-large-2411 Mistral AI model', 'max_context_length': 131072, 
                'aliases': ['mistral-large-latest'], 'deprecation': None, 'default_model_temperature': 0.7, 'TYPE': 'base'
            }
        """
        model_card = self._client.models.retrieve(model_id=self.model_name)
        # Pydanticモデルの場合は .model_dump()、それ以外は __dict__ で対応
        if hasattr(model_card, "model_dump"):
            return model_card.model_dump()
        elif hasattr(model_card, "__dict__"):
            return dict(model_card.__dict__)
        else:
            # それ以外の場合は型変換を試みる
            return dict(model_card)


    def usage_info(self) -> None:
        """ トークンの使用量情報をログへ出力する"""
        usage = cast(UsageInfo, getattr(self._chat_response, "usage", None))
        self.logger.info(f"プロンプト(入力 : 出力 : 合計)  {usage.prompt_tokens} : {usage.completion_tokens} : {usage.total_tokens}")   