""" Main executable for the agent server. 

Accepts the following command line arguments:
```
    python3 languagegame/agent_server/main.py --port 8000 [--config CONFIG_FILE]
```

Config JSON file should specify all the constructor args for the Agent class. 
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import argparse
import uvicorn
import json
from threading import Thread


from agent import Agent
from languagegame.models import MessageModel, InferenceServerModel, AgentServerModel, CorpusChunkModel, NeighborModel, SystemPromptModel

app = FastAPI()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, 
                    help="Path to the external config file. Default is languagegame/agent_server/config.json", 
                    default="languagegame/agent_server/config.json")
parser.add_argument("--port", type=int, default=8000, help="Port to run the agent server on. Default 8000")
args = parser.parse_args()

with open(args.config, 'r') as f: 
    CONFIG = json.load(f)

# Create agent instance
agent_instance = Agent(args.port, **CONFIG)  # Assuming the CONFIG provides all required parameters for Agent
agent_instance.load_state_from_disk() # Load state from disk

consumer_thread = Thread(target=agent_instance.message_daemon)
consumer_thread.start()

# Shutdown logic
@app.on_event("shutdown")
async def shutdown_event():
    # Any shutdown related logic for agent, if required
    # agent_instance.save_state_to_disk()
    agent_instance.unregister_from_coordinator()

#################
### ENDPOINTS ###
#################
@app.post("/assign_llm")
async def assign_llm(llm: InferenceServerModel):
    try:
        agent_instance.assign_llm(llm)
        return {"status": "LLM assigned successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status(): 
    try: 
        return agent_instance.get_self_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/full_status")
async def status(): 
    try: 
        return agent_instance.full_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/receive_message")
async def receive_message(incoming_message: MessageModel):
    try:
        return agent_instance.receive_message(incoming_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_neighbor")
async def add_neighbor(neighbor: NeighborModel):
    try:
        agent_instance.add_neighbor(neighbor)
        return {"status": "Neighbor added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/receive_neighbor_request")
async def receive_neighbor_request(neighbor: NeighborModel):
    try:
        response = agent_instance.receive_neighbor_request(neighbor)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_neighbor")
async def remove_neighbor(neighbor: NeighborModel):
    try:
        agent_instance.remove_neighbor(neighbor.uid)
        return {"status": "Neighbor removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/process_corpus")
async def process_corpus(corpus_chunk: CorpusChunkModel):
    try:
        agent_instance.process_corpus(corpus_chunk)
        return {"status": "Processed corpus successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/init_message_exchange")
async def init_message_exchange():
    try:
        agent_instance.message_exchange()
        return {"message": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System prompt/prompt evolution endpoints
@app.get("/get_system_prompt")
async def get_system_prompt():
    try:
        return {"system_prompt": agent_instance.get_system_prompt()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_system_prompt")
async def set_system_prompt(system_prompt: SystemPromptModel):
    try:
        agent_instance.set_system_prompt(system_prompt)
        return {"status": "System prompt set successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG['host'], port=args.port)