# CoupleGPT

## Setup
```
# create virtual environment
python3 -m venv venv

# activate 
source venv/bin/activate

# install the package and dependencies
pip install -e .
pip install -r requirements.txt
```

## Run the Game
```
# turn on inference server
python3 languagegame/inference_server/main.py \
	--config configs/gpt2_inference_server.json \
	--port 4444
```



## Messages from the pair
