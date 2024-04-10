# CoupleGPT

## Setup
```bash
# create virtual environment
python3 -m venv venv

# activate 
source venv/bin/activate

# installing the LanguageGame package and dependencies
pip install -e .
pip install -r requirements.txt
```

## Run the Game
```bash
echo "turning on coordinator server"
venv/bin/python3 languagegame/coordinator/main.py

echo "turning on inference server"
# venv/bin/python3 languagegame/inference_server/main.py \
# 	--config configs/gpt2_inference_server.json \
# 	--port 4444
venv/bin/python3 languagegame/inference_server/main.py \
	--config configs/capybara_inference_server.json \
	--port 4444

echo "turning on gui server" 
venv/bin/python3 gui/main.py --port 7998

echo "starting processes..." 
python3 program1.py & 
python3 program2.py & # etc.
```

## Mess with the Game
```bash

```

