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
# inference server, no coordinator, gpt-2
python3 languagegame/inference_server/main.py \
	--config configs/min_inference_server.json \
	--port 4444

# convo simulator with Eos and Athena
python3 couple_gpt.py
```
