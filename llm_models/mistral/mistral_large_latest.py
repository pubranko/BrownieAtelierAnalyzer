from typing import Union, cast
from typing import Optional
from decouple import config
from logging import Logger, LoggerAdapter
from mistralai import Mistral
from mistralai.models import ChatCompletionResponse, ChatCompletionChoice
from mistralai.models.basemodelcard import BaseModelCard


"""
apiドキュメント
    https://docs.mistral.ai/
apiサイト (api-key等の発行、)
    https://console.mistral.ai/home
api
    https://docs.mistral.ai/api/

"""

class MistralLargeLatest:
    
    model_name: str
    logger: Union[Logger, LoggerAdapter]
    
    __api_key:str
    
    _client:Mistral

    def __init__(self, logger: Union[Logger, LoggerAdapter], api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        
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

    def chat(self, prompt:str, messages: list[dict[str, str]]) -> None:
        """
        llmとのチャットを実行する

        Args:
            prompt (str): LLMへ送信するチャットのテキストを指定する。messagesが指定されている場合は無視される。
            messages (list[dict[str, str]]): LLMへ送信するチャットの履歴を含むメッセージを指定する。
        """
        
        if messages:
            _ = messages
        else:
            _ = [
                {"role": "user", "content": prompt}
            ]
        
        self.chat_response:ChatCompletionResponse = self._client.chat.complete(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # messages=[
            #     {"role": "user", "content": prompt}
            # ],
            # top_p
            # max_tokens
            # timeout_ms
        )
        
    def chat_response_to_text(self) -> str:
        """
        chat_responseの内容をテキストに変換する
        """
        chat_response_choices: Optional[list[ChatCompletionChoice]] = self.chat_response.choices
        if chat_response_choices:
            return str(chat_response_choices[0].message.content)
        else:
            return ""
    
    def model_infomation(self) -> BaseModelCard:
        """ 指定モデルの詳細情報を取得
        Returns:
            BaseModelCard: モデルに関する情報をプロパティで参照できます。
            （参考例）
            {
                'id': 'mistral-large-2411', 'capabilities': ModelCapabilities(completion_chat=True, 
                completion_fim=False, function_calling=True, fine_tuning=True, vision=False), 
                'object': 'model', 'created': 1745484085, 'owned_by': 'mistralai', 'name': 'mistral-large-2411', 
                'description': 'Official mistral-large-2411 Mistral AI model', 'max_context_length': 131072, 
                'aliases': ['mistral-large-latest'], 'deprecation': None, 'default_model_temperature': 0.7, 'TYPE': 'base'
            }
        """
        _ = self._client.models.retrieve(model_id=self.model_name)  # 指定モデルの詳細情報を取得
        return cast(BaseModelCard, self._client.models.retrieve(model_id=self.model_name))   # 型定義をBaseModelCardに変換
