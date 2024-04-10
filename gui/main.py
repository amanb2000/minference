""" FastAPI GUI server -- run this to serve the GUI. 
TODO: Set the address for the coordinator in a config.json file. 
"""
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import requests

# scientific computing uwu
import numpy as np

app = FastAPI()

# List of allowed origins. You can also use ["*"] to allow any origin
origins = [
    "http://localhost:7998",  # Adjust this to your needs
    "*"
]

app.mount("/static", StaticFiles(directory="gui/static"), name="static")

templates = Jinja2Templates(directory="gui/templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/coordinator_status/", response_class=HTMLResponse)
async def coordinator_status(request: Request):
    url = "http://localhost:7999/status"
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
    # resp = requests.get(url)
        
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch status from coordinator.")

    data = resp.json()

    # data = {"inference_servers":[{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5}],"agent_servers":[{"uid":"agent-0987d39c","ip_address":"65.19.181.239","port":7770,"neighbors":[{"uid":"agent-a3a0c2c7","ip_address":"65.19.181.239","port":7766,"go_first":False},{"uid":"agent-59f09eab","ip_address":"65.19.181.239","port":7764,"go_first":False}],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"},{"uid":"agent-a3a0c2c7","ip_address":"65.19.181.239","port":7766,"neighbors":[{"uid":"agent-0987d39c","ip_address":"65.19.181.239","port":7770,"go_first":False},{"uid":"agent-ee7ea13e","ip_address":"65.19.181.239","port":7768,"go_first":False}],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"},{"uid":"agent-ee7ea13e","ip_address":"65.19.181.239","port":7768,"neighbors":[{"uid":"agent-a3a0c2c7","ip_address":"65.19.181.239","port":7766,"go_first":False},{"uid":"agent-4bc32885","ip_address":"65.19.181.239","port":7769,"go_first":False}],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"},{"uid":"agent-4bc32885","ip_address":"65.19.181.239","port":7769,"neighbors":[{"uid":"agent-ee7ea13e","ip_address":"65.19.181.239","port":7768,"go_first":False},{"uid":"agent-da09b7b6","ip_address":"65.19.181.239","port":7767,"go_first":False}],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"},{"uid":"agent-da09b7b6","ip_address":"65.19.181.239","port":7767,"neighbors":[{"uid":"agent-4bc32885","ip_address":"65.19.181.239","port":7769,"go_first":False},{"uid":"agent-4067318e","ip_address":"65.19.181.239","port":7765,"go_first":False}],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"},{"uid":"agent-4067318e","ip_address":"65.19.181.239","port":7765,"neighbors":[{"uid":"agent-da09b7b6","ip_address":"65.19.181.239","port":7767,"go_first":False},{"uid":"agent-59f09eab","ip_address":"65.19.181.239","port":7764,"go_first":False}],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"},{"uid":"agent-59f09eab","ip_address":"65.19.181.239","port":7764,"neighbors":[{"uid":"agent-4067318e","ip_address":"65.19.181.239","port":7765,"go_first":False},{"uid":"agent-0987d39c","ip_address":"65.19.181.239","port":7770,"go_first":False}],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}],"action_log":[{"action":"register_inference_server","server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5}},{"action":"register_agent_server","server":{"uid":"agent-0987d39c","ip_address":"65.19.181.239","port":7770,"neighbors":[],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}},{"action":"register_agent_server","server":{"uid":"agent-a3a0c2c7","ip_address":"65.19.181.239","port":7766,"neighbors":[],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}},{"action":"register_agent_server","server":{"uid":"agent-ee7ea13e","ip_address":"65.19.181.239","port":7768,"neighbors":[],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}},{"action":"register_agent_server","server":{"uid":"agent-4bc32885","ip_address":"65.19.181.239","port":7769,"neighbors":[],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}},{"action":"register_agent_server","server":{"uid":"agent-da09b7b6","ip_address":"65.19.181.239","port":7767,"neighbors":[],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}},{"action":"register_agent_server","server":{"uid":"agent-4067318e","ip_address":"65.19.181.239","port":7765,"neighbors":[],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}},{"action":"register_agent_server","server":{"uid":"agent-59f09eab","ip_address":"65.19.181.239","port":7764,"neighbors":[],"inference_server":{"uid":"llm-dd990778","ip_address":"65.19.181.239","port":8000,"llm_name":"gpt2","max_seq_len":1024,"batch_size":5},"status":"Ready"}},{"action":"initialize_game","k":4,"p":0.1}],"agent_status":{"agent-0987d39c":"Ready","agent-a3a0c2c7":"Ready","agent-ee7ea13e":"Ready","agent-4bc32885":"Ready","agent-da09b7b6":"Ready","agent-4067318e":"Ready","agent-59f09eab":"Ready"}}

    return templates.TemplateResponse("coordinator_home.html", {"request": request, "data": data})


async def fetch_agent_status(ip: str, port: int):
    url = f"http://{ip}:{port}/full_status"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None  # Or handle error appropriately

@app.get("/agent/{uid}")
async def agent_page(uid: str, request: Request):
    coordinator_data = requests.get("http://localhost:7999/status").json()
    # print("Coordinator_data: ", coordinator_data)
    agents_data = {agent["uid"]: agent for agent in coordinator_data["agent_servers"]}
    # print("Agents_data: ", agents_data)
    print("Agent uids: ", agents_data.keys())
    print("Ground truth uids: ", [x['uid'] for x in coordinator_data["agent_servers"]])
    print("Desired uid: ", uid)

    agent = agents_data[uid]
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_status = await fetch_agent_status(agent["ip_address"], agent["port"])
    if not agent_status:
        raise HTTPException(status_code=500, detail="Failed to fetch agent status")

    return templates.TemplateResponse("agent_page.html", {"agent": agent_status, "request": request})



@app.get("/agent/{uid}/chat_data/{receiver_uid}")
async def get_chat_data(request: Request, uid: str, receiver_uid: str, chat_type: str = 'chat_log'):
    # Start by getting the agent port 
    print("UID: ", uid)
    print("Receiver UID: ", receiver_uid)
    coordinator_data = requests.get("http://localhost:7999/status").json()
    agents_data = {agent["uid"]: agent for agent in coordinator_data["agent_servers"]}
    agent = agents_data[uid]

    # Here, we're assuming you have a function fetch_agent_data(uid) that fetches the agent data.
    agent_data = await fetch_agent_status(agent['ip_address'], agent['port'])
    assert not (agent_data is None)
    print("Type of chat log: ", type(agent_data['chat_log']))
    print("Type of chat archive: ", type(agent_data['chat_archive']))
    if chat_type == 'chat_log':
        chat_data = agent_data['chat_log'][receiver_uid]
    else:
        chat_data = agent_data['chat_archive'][receiver_uid]

    # Render a partial template for chat data. 
    # This could just be an HTML file that formats the chat_data as desired.
    return templates.TemplateResponse("chat_data.html", {"request": request, "chat_data": chat_data})

@app.post("/gui_initialize_game")
async def gui_initialize_game(request: Request):
    url = "http://localhost:7999/initialize_game"
    data = {"k": 4, "p": 0.1}
    resp = requests.post(url, json=data)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to initialize game.")
    return {"message": "successfully initialized game"}


@app.get("/gui_run_game")
async def gui_run_game(num_iters: int, generation_period: int, death_fraction:float): 
    
    url = "http://localhost:7999/run_iterated_game"
    data = {"num_iters": num_iters, "generation_period": generation_period, "death_fraction": death_fraction}
    print("Request to coordinator node to run game: ", data)
    # Use an async httpx client
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=data)
    
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to run game.")
    
    return {"message": "successfully ran game"}


    # resp = requests.post(url, json=data)
    # if resp.status_code != 200:
    #     raise HTTPException(status_code=resp.status_code, detail="Failed to run game.")
    # return {"message": "successfully ran game"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"error": f"{exc.detail}"}


# Endpoint for visualizing + comparing all agent score data
@app.get("/agent_scores")
def get_agent_scores(request: Request):
    # Fetch data from the coordinator
    response = requests.get("http://localhost:7999/status")
    data = response.json()

    # Extract IP:port information of each agent
    agent_data = [requests.get(f"http://{agent['ip_address']}:{agent['port']}/full_status").json() for agent in data['agent_servers']]

    # Extracting the loss values and the means, stds for each agent over time.
    processed_data = []
    for agent in agent_data:
        loss_values = [entry['loss'] for entry in agent['corpus_log'] if 'corpus_string' in entry]

        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)

        processed_data.append({
            "uid": agent['uid'],
            "loss_values": loss_values,
            "mean_loss": mean_loss,
            "std_loss": std_loss
    })

    # Sort agents by their mean CE loss
    processed_data = sorted(processed_data, key=lambda x: x['mean_loss'])


    # Extracting the loss values and the means, stds for each agent over time.
    datasets = []
    for agent in agent_data:
        loss_values = [entry['loss'] for entry in agent['corpus_log'] if 'corpus_string' in entry]

        datasets.append({
            "label": agent['uid'],
            "data": loss_values,
            # You can assign colors here if you want or handle it in JavaScript
        })

    # Assuming every agent has the same number of loss values, or you've handled it such that they do:
    labels = list(range(1, len(datasets[0]['data']) + 1))

    chart_data = {
        "labels": labels,
        "datasets": datasets
    }


    # Further processing of data and calculations would be here

    # Render the template with the required data
    return templates.TemplateResponse("agent_scores.html", {"data": processed_data, "request": request, "chart_data": chart_data})



if __name__ == "__main__": 
    # port is expected as parseargs-friendly `python3 gui/main.py --port 7998`
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7998, help="Port to run the GUI server on. Default 7998")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)

