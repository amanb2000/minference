# minference 

Makes it easy to spin up a quick inference server that I can easily add new
functions to (interact with the model directly, test research code) and not 
worry about building a batched system in some scripts. 



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

## Run an Inference Server
```bash
# inference server, no coordinator, gpt-2
# localhost:4444/docs for docs 
python3 languagegame/inference_server/main.py \
	--config configs/min_inference_server.json \
	--port 4444

# chat interface -- saves chat logs to chat/*.log
# pass --help for docs 
python3 chat.py 

# minimal demo of making inference server calls
# pass --help for docs
python3 manual_inference_client.py
```
