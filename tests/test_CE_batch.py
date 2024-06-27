import argparse 
import httpx
import json
import asyncio
import time

async def send_request(client, endpoint, data):
    response = await client.post(endpoint, json=data)
    return response.json()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_endpoint", type=str, default="http://localhost:4444/ce_loss")
    parser.add_argument("--data", type=str, default="tests/ce_data.json")
    parser.add_argument("--num_sup_iters", type=int, default=1, help="Number of iterations on the --data.")
    args = parser.parse_args()

    # Load the data:
    with open(args.data, "r") as f:
        data_ = json.load(f)
    data = []
    for k in range(args.num_sup_iters): 
        for d in data_: 
            data.append(d)

    # Send the requests in parallel with httpx and gather the results:
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, args.loss_endpoint, d) for d in data]
        
        print("Sending requests...")
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"All requests completed in {end_time - start_time:.2f} seconds")

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

if __name__ == "__main__":
    asyncio.run(main())