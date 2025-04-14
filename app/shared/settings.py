import os
from datetime import timedelta, timezone
from typing import Final

from decouple import AutoConfig, config

# .envファイルが存在するパスを指定。実行時のカレントディレクトリに.envを配置している場合、以下の設定不要。
# config = AutoConfig(search_path="./shared")


TIMEZONE = timezone(timedelta(hours=9), "JST")
"""タイムゾーン"""

DATA = os.path.abspath(str(config("PREFECT__DATA", default="../data")))
"""データ類の保存ベースディレクトリ"""

DATA__LOGS = os.path.join(DATA, "logs")
"""ログの一時保存先"""
DATA__BACKUP_BASE_DIR: str = os.path.join(DATA, "backup_files")
"""バックアップファイルを保存するベースディレクトリパス"""


LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] : %(message)s"
"""ログフォーマット"""
# LOG_FORMAT = '[%(asctime)s] %(levelname)s - %(name)s | %(message)s'
# 日付/時刻をフォーマットするための文字列、 LOG_FORMAT の %(asctime)s プレース・ホルダーの展開。
# 利用可能なディレクティブのリストについては、 Python datetime documentation を参照してください。
LOG_DATEFORMAT = "%Y-%m-%d %H:%M:%S"
"""ログ日時フォーマット"""
