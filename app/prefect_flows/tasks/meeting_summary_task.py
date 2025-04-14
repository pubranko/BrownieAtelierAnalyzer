import logging
from logging import FileHandler, StreamHandler
from typing import Any

from prefect import get_run_logger, task
from llm_models.rinna.japanese_gpt_1b import JapaneseGPT1B


@task
def meeting_summary_task(
):
    """
    """
    logger = get_run_logger()  # PrefectLogAdapter
    logger.info("meeting_summary_task 開始")


    japanese_gpt1_b = JapaneseGPT1B(logger)
    result = japanese_gpt1_b.generate(prompt="西田幾多郎は、どのような人ですか？")
    
    print(type(result))
    print(result)
