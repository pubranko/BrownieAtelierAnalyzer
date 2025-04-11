# BrownieAtelierAnalyzer
LLMを使った解析用リポジトリ


python3.10 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pipenv install torch torchvision torchaudio transformers sentencepiece protobuf
## jupyterを使う場合以下も必要。
pipenv install ipykernel
