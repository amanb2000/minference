""" Manual client to test inference on one of our inference APIs
"""

import requests
import json
import argparse
from languagegame.models import GenRequest



def inference_call(req: GenRequest, API_URL, num_tokens=50):
    response = requests.post(API_URL, json=req.model_dump())

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str, default="http://localhost:4444/generate",
                        help="URL of the inference API to test. Default is http://localhost:4444/generate")
    args = parser.parse_args()
    print("\nRunning client on API located at ", args.api_url)

    while True: 
        system_prompt = input("Enter the system prompt: ")
        input_str = input("Enter the input string: ")
        num_tok = int(input("Enter the number of tokens to generate (default 50): ") or "50")

        req = GenRequest(
            system_prompt=system_prompt,
            input_string=input_str,
            num_tokens=num_tok
        )

        result = inference_call(req, args.api_url, num_tok)
        if result:
            print("\n\n")
            # print("input_string (Request):", result["input_string"])
            print("\n\nGenerated Text:", result["generated"])
            # print("\nNum truncated: ", result["num_truncated"])
