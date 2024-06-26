"""
In this script we will test the CE loss endpoint of the API. 

Assumed: We are running Llama-3 8b Instruct on localhost at port 4444. 

python3 languagegame/inference_server/main.py         
    --config configs/min_llama_3_8b_instruct.json     
    --port 4444
"""
import argparse 
import httpx
import json
import numpy as np
import pdb
from tqdm import tqdm 


# argparse: 
# --loss_endpoint: str: The endpoint for the loss function, default http://localhost:4444/ce_loss
# --data: str: The path to the data file, default data/ce_data.json
parser = argparse.ArgumentParser()
parser.add_argument("--loss_endpoint", type=str, default="http://localhost:4444/ce_loss")
parser.add_argument("--data", type=str, default="tests/ce_data.json")
args = parser.parse_args()


# Load the data:
with open(args.data, "r") as f:
    data = json.load(f)


# Send the requests in parallel with httpx and gather 
# the results:
results = []
with httpx.Client() as client:
    for d in tqdm(data):
        r = client.post(args.loss_endpoint, json=d)
        results.append(r.json())


# Check the results:
for i, r in enumerate(results):
    print(f"Test {i+1}")
    print(f"Request: {data[i]}")
    print(f"Response: {r}")
    print("\n")
    print("Test passed.")
    print("\n\n")

# write results to a new json 
with open("tests/ce_results.json", "w") as f:
    json.dump(results, f)