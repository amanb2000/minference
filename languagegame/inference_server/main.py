""" Run this script to start the inference server.

```
   python3 languagegame/inference_server/main.py --port 8000 [--config CONFIG_FILE]
```
"""


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import argparse
import uvicorn
import json
import threading

from languagegame import InferenceServer
from languagegame.models import LossRequest, GenRequest

# app = FastAPI()
app = FastAPI(root_path="/colossus")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, 
                    help="Path to the external config file. Default is languagegame/inference_server/config.json", 
                    default="languagegame/inference_server/config.json")
parser.add_argument("--port", type=int, default=8000, help="Port to run the inference server on. Default 8000")
args = parser.parse_args()

with open(args.config, 'r') as f: 
    CONFIG = json.load(f)

# Create inference server instance
logic = InferenceServer(args.port,
                        **CONFIG)

# Start the worker thread running 
threading.Thread(target=logic.worker_daemon, daemon=True).start()


# Shutdown logic
@app.on_event("shutdown")
async def shutdown_event():
    # Unregister the server on shutdown
    logic.unregister_from_coordinator()



#################
### ENDPOINTS ###
#################
    
# Endpoint to force a re-registration with the server
@app.get("/reregister")
def register():
    try:
        logic.register_with_coordinator()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to generate text
@app.post("/generate")
def generate(request: GenRequest):
    try:
        response = logic.generate_endpoint(request)
        return response
    except Exception as e:
        print("ERROR IN GENERATION: ", e)
        return {'message': 'Error in generation!', 'error': str(e)}

# Endpoint to compute CE loss
@app.post("/ce_loss")
def ce_loss(request: LossRequest):
    """ CE loss computation endpoint. Computes CE loss on P(corpus_str | context_str)
    Args:
    - request (LossRequest): Input request with a context string and corpus string.
    Returns:
    - Dict with the original request and computed loss.
    """
    try:
        # response = logic.compute_loss(request)
        response = logic.ce_loss_endpoint(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Status endpoint
@app.get("/status")
def status(): 
    try: 
        return logic.get_self_model().model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG['host'], port=args.port)
