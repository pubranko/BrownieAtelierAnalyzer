from typing import Any

from BrownieAtelierMongo.collection_models.mongo_model import MongoModel
from prefect import flow, get_run_logger
from prefect.futures import PrefectFuture
from prefect_flows.flows import START_TIME
from prefect_flows.flows.init_flow import init_flow
from prefect_flows.tasks.meeting_summary_task import meeting_summary_task
from prefect_flows.tasks.end_task import end_task
from prefect_flows.tasks.init_task import init_task


@flow(name="Manual meeting summary flow")
def manual_meeting_summary_flow():
    init_flow()

    # ロガー取得
    logger = get_run_logger()  # PrefectLogAdapter
    # 初期処理
    init_task_instance: PrefectFuture = init_task.submit()
    # 実行結果が返ってくるまで待機し、戻り値を保存。 
    #   ※タスクのステータスをresultを受け取る前に判定してもPendingとなる。インスタンスのステータスはリアルタイムで更新されているので注意。
    init_task_result = init_task_instance.result()

    if init_task_instance.state.is_completed():
        mongo: MongoModel = init_task_result

        try:

            meeting_summary_task()


        except Exception as e:
            # 例外をキャッチしてログ出力等の処理を行う
            logger.error(f"=== {e}")
        finally:
            # 後続の処理を実行する
            end_task(mongo)

    else:
        logger.error(f"=== init_taskが正常に完了しなかったため、後続タスクの実行を中止しました。")
