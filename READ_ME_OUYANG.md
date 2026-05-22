# Setup

 uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
  uv pip install --python .venv/bin/python -e .
  uv pip install --python .venv/bin/python -e ../nahual

Load the weights:
  uv run --python .venv/bin/python python ensure_model.py --model-channels rybg --model-type mae_contrast_supcon_model

Run the server

  uv run --python .venv/bin/python python server.py tcp://0.0.0.0:5110
