import logging
from logging import Logger, LoggerAdapter
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from typing import Any, Union


class JapaneseGPT1B:
    
    model_name = "rinna/japanese-gpt-1b"
    cache_dir="/mnt/c/LLM/rinna/japanese-gpt-1b"
    # tokenizer: AutoTokenizer
    # model: Any  # AutoModelForCausalLM or torch.device
    initialized = False
    logger: Union[Logger, LoggerAdapter]

    def __init__(self, logger: Union[Logger, LoggerAdapter]):
        
        self.logger = logger
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, use_fast=True, legacy=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        self.initialized = True
        logger.info(f"=== ロード完了 {self.model_name}")
        
        
    
        if torch.cuda.is_available():
            # model = model.to("cuda")    # RAMからGPU側へ転送
            device = torch.device("cuda")  # GPUデバイスを取得
            self.model = self.model.to(device)       # GPUにモデルを転送
            logger.info(f"=== モデルのGPU利用: {torch.cuda.get_device_name(0)}")
            
            
    def generate(self,
            prompt: str, 
            max_length: int=0, 
            temperature:float=0.7, 
            top_p:float=0.9, 
            top_k:float=0, 
            repetition_penalty: float=1.0
        ) -> str:
        """
        プロンプトを受け取り、モデルからの応答を生成する
        
        Args:
            prompt (str): 入力プロンプト
            max_length (int): 生成するテキストの最大長
            temperature (float): 生成の温度パラメータ
            top_p (float): 核サンプリングのパラメータ
            top_k (int): top-kサンプリングのパラメータ
            repetition_penalty (float): 繰り返しペナルティ
            
        Returns:
            str: 生成されたテキスト
        """

        # 入力テキストをトークン化
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        any: Any = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids: torch.Tensor = any

        # トークンもGPUに転送
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")    # RAMからGPU側へ転送
            self.logger.info(f"=== トークンのGPU利用: {torch.cuda.get_device_name(0)}")
        
        # テキスト生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids.to(self.model.device),
                max_length=100,
                min_length=100,
                do_sample=True,
                top_k=500,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[self.tokenizer.unk_token_id]]
            )
        
        
        # 生成されたテキストをデコード
        # generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output:str = self.tokenizer.decode(output_ids.tolist()[0])
        
        # プロンプトの正規化のみ実行
        normalized_text = self.tokenizer.backend_tokenizer.normalizer.normalize_str(prompt)
        # 入力プロンプトを除いた生成部分のみを返す
        # プロンプトの長さによっては完全に一致しない場合があるため、近似的に処理
        if output.startswith(normalized_text):
            return output[len(normalized_text):]
        else:
            # プロンプトが完全に一致しない場合は全体を返す
            return output

    # def __call__(self, prompt, **kwargs):
    #     """
    #     クラスを関数のように呼び出せるようにする
        
    #     Args:
    #         prompt (str): 入力プロンプト
    #         **kwargs: generate関数に渡す追加パラメータ
            
    #     Returns:
    #         str: 生成されたテキスト
    #     """
    #     return self.generate(prompt, **kwargs)
    
