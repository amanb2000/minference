""" Server script for the coordinator node. Each agent and LLM inference server 
will register itself with the coordinator when it first starts up. The 
coordinator serves this information at an API endpoint `/status`. 
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import json
import argparse

# List of allowed origins. You can also use ["*"] to allow any origin
origins = [
    "http://dev.languagegame.io:7998",  # Adjust this to your needs
    "*"
]


from languagegame import Coordinator
from languagegame.models import InferenceServerModel, AgentServerModel, GameInit, CorpusChunkModel, GameReq

import pdb

global app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="Path to the external config file. Default is languagegame/coordinator/config.json",
                    default="languagegame/coordinator/config.json")
parser.add_argument("--port", type=int, default=7999, help="Port to run the coordinator server on. Default 8000")
args = parser.parse_args()

with open(args.config, 'r') as f:
    CONFIG = json.load(f)

# Create coordinator instance
# Assuming the CONFIG provides all required parameters for Coordinator
# Define a startup event handler
@app.on_event("startup")
async def startup_event():
    global coordinator_instance
    coordinator_instance = Coordinator(args.port, **CONFIG)  


# API endpoint for registering an LLM inference server
@app.post("/register_inference_server")
async def register_inference_server(server: InferenceServerModel):
    try: 
        response = coordinator_instance.register_inference_server(server)
        # return the InferenceServer object with an assigned UID, etc. 
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # global inference_servers
    # inference_servers.append(server)
    # return {"message": "success"}

# API endpoint for registering an agent server
@app.post("/register_agent_server")
async def register_agent_server(server: AgentServerModel):
    print("Received agent server connection request ", server)
    try: 
        response = coordinator_instance.register_agent_server(server)
        # return the AgentServer object with an assigned UID, etc. 
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # global agent_servers
    # agent_servers.append(server)
    # return {"message": "success"}



# API endpoint for un-registering an agent server
@app.post("/unregister_agent_server")
async def unregister_agent_server(req: AgentServerModel):
    try:
        coordinator_instance.unregister_agent_server(req)
        return {"message": "success"}
    except:
        raise HTTPException(status_code=500, detail="Server with uid {} not found".format(req.uid))

# API endpoint for un-registering an LLM inference server
@app.post("/unregister_inference_server")
async def unregister_inference_server(req: InferenceServerModel):
    try:
        coordinator_instance.unregister_inference_server(req)
        return {"message": "success"}
    except:
        raise HTTPException(status_code=500, detail="Server with uid {} not found".format(req.uid))

# API endpoint for initializing the game
@app.post("/initialize_game")
async def initialize_game(req:GameInit):
    try:
        coordinator_instance.initialize_game(req.k, req.p)
        return {"message": "success"}
    except:
        raise HTTPException(status_code=500, detail="initialize_game failed with input args: {}".format(arg_dict))

# API endpoint for getting the status of the coordinator
@app.get("/status")
async def get_status():
    try: 
        return coordinator_instance.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Endpoint for broadcasting a chunk of the corpus to all nodes assembled.
@app.post("/broadcast_corpus")
async def broadcast_corpus(chunk:CorpusChunkModel):
    try:
        coordinator_instance.broadcast_corpus(chunk)
        return {"message": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Endpoint for starting a round of message exchange
@app.post("/init_message_exchange")
async def initiate_message_exchange():
    try:
        print("COORDINATOR RECEIVED INIT MESSAGE EXCHANGE REQUEST")
        result = coordinator_instance.initiate_message_exchange()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for running an iterated language game with the dataset 
@app.post("/run_iterated_game")
async def run_iterated_game(background_tasks: BackgroundTasks, req:GameReq):
    """ To run a game with no evolution, set the `generation_period` to -1.
    If `generation_period` is a positive integer, death fraction must be a float
    between 0 and 1.
    """
    print("[Coordinator] Received iterated game request ", req)
    try:
        # Schedule run_game_endpoint to run in the background
        background_tasks.add_task(coordinator_instance.run_game_endpoint, req.num_iters, generation_period=req.generation_period, death_fraction=req.death_fraction)
        return {"message": "Game running in the background"}
    except Exception as e:
        print("[Coordinator] Error occurred while running iterated game: ", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/all_confirmed")
async def all_confirmed(): 
    try: 
        result = coordinator_instance._all_confirmed()
        return {'all_confirmed': result} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conf_exchange")
async def confirm_exchange(agent:AgentServerModel): 
    try: 
        result = await coordinator_instance.confirm_exchange(agent)
        return result
    except Exception as e:
        print("[Coordinator] Error occurred while confirming exchange: ", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_agent_scores")
async def get_agent_scores():
    try: 
        result = coordinator_instance.get_agent_scores()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutate_agents")
async def mutate_agents(num_recent:int): 
    """ `num_recent` is the number of recent corpus loss scores we want to take 
    into account as we compute performance to determine which agents reproduce
    and which agents die. 
    """
    try: 
        result = coordinator_instance.mutate_agents(num_recent)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    # uvicorn.run("main:app", host=str(CONFIG['host']), port=int(args.port), reload=True)
    uvicorn.run(app, host=str(CONFIG['host']), port=int(args.port))
            
