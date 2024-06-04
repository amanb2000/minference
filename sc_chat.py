"""
Self-contained chat program for interacting with minference-based 
LLM APIs. 

Saves the transcripts to a folder chat/ in the current directory, 
unless otherwise specified. 

Usage:
    python sc_chat.py --api_url <api_url> --num_tokens <num_tokens> --auto

    api_url: URL of the inference API to test. Default is http://control.languagegame.io/colossus/generate, which I am freely hosting llama-3 70b instruct as a test. 
    num_tokens: Number of tokens to generate per inference call. Default is 20.
    auto: Include this flag to automatically continue generating `num_tokens` more assistant responses for the rest of time. Default=False

"""

from pydantic import BaseModel
import os
import requests
import json
import argparse
from datetime import datetime
import pdb
# line width 80 pretty printing 
import textwrap

class GenRequest(BaseModel):
    input_string: str
    num_tokens: int = 50
    system_prompt: str = ""
    greedy: bool = False


def inference_call(req: GenRequest, API_URL, num_tokens=50):
    response = requests.post(API_URL, json=req.model_dump())

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def load_system_prompt(file_path):
    # check if file exists -- if not, return default system prompt. 
    if not os.path.exists(file_path):
        return "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. It is designed to help you with your tasks and to provide information in a conversational manner. Enjoy the conversation! The inference script the user is using has no further information on the nature of the chatbot. Tell the user to define this in chat/system_prompt.md if they want to change their system prompt. Any serious user should probably do that."

    with open(file_path, "r") as file:
        return file.read().strip()

def save_chat(chat_history, file_path):
    # with open(file_path, "w") as file:
    #     file.write("\n".join(chat_history))
    """ use one line per list element (string beginning with "User: " or "Assistant: ")
    """
    with open(file_path, "w") as file:
        for line in chat_history:
            file.write(line + f"\n")
        

def load_chat(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read().split("\n")
    return []

def list_chat_files(chat_dir):
    return [file for file in os.listdir(chat_dir) if file.endswith(".log")]

def main(args):
    chat_dir = "chat"
    system_prompt_file = os.path.join(chat_dir, "system_prompt.md")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chat_file = os.path.join(chat_dir, f"chat_{current_time}.log")

    system_prompt = load_system_prompt(system_prompt_file)
    chat_history = []

    print(f"Current chat file: {chat_file}")
    print("Type 'quit' to exit the chat.")
    print("Type 'list' to list available chat files.")
    print("Type 'load <file_name>' to load a specific chat file.")
    print("Press Enter with no input to let the assistant continue generating.")

    while True:
        print("\n\n--------------------------------------------------------------------------------")
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "refresh": 
            # print entire chat history
            print("\n----------------------------------------------------------------------------------------------------".join(chat_history))
        elif user_input.lower() == "list":
            chat_files = list_chat_files(chat_dir)
            print("Available chat files:")
            for file in chat_files:
                print(f"- {file}")
            continue
        elif user_input.lower() == "whoami": 
            print("You are the user in chat ", chat_file, "\n")
            continue
        elif user_input.lower().startswith("load"):
            file_name = user_input.split(" ")[1]
            chat_file = os.path.join(chat_dir, file_name)
            chat_history = load_chat(chat_file)
            print(f"Loaded chat file: {chat_file}")
            continue

        chat_history.append(f"User: {user_input}")
        save_chat(chat_history, chat_file)

        input_string = "\n".join(chat_history)

        req = GenRequest(
            system_prompt=system_prompt,
            input_string=input_string+f"\nAssistant: ",
            num_tokens=args.num_tokens
        )

        result = inference_call(req, args.api_url, args.num_tokens)
        if result:
            generated_text = result["generated"]
            generated_text = " ".join(generated_text.split("\n"))
            generated_text = textwrap.fill(generated_text, width=80)

            chat_history.append(f"Assistant: {generated_text}")
            save_chat(chat_history, chat_file)
            print(f"\n\n--------------------------------------------------------------------------------\nAssistant: {generated_text}")
        else: 
            print("ERROR in results: ", result) 

        if not args.auto: 
            user_input = input(f"")
        while args.auto or (user_input == "" and not args.auto):
            req.num_tokens = args.num_tokens
            # get the current chat history 
            input_string = "\n".join(chat_history)
            req.input_string = input_string
            # print("\n[Sending inference call]")
            result = inference_call(req, args.api_url, args.num_tokens)
            # print("\n[Received inference call result]")
            # print("\t",result)
            # print("request: ", req.model_dump())
            if result:
                generated_text = result["generated"]
                # remove newlines, replace with spaces, set linewidth 80 
                generated_text = " ".join(generated_text.split("\n"))
                # generated_text = textwrap.fill(generated_text, width=80)

                chat_history[-1] += generated_text
                save_chat(chat_history, chat_file)
                print(generated_text, end="", flush=True)
            else: 
                print("ERROR IN GETTING RESULT FROM INFERENCE CALL: ", result)
            if not args.auto: 
                user_input = input("")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str, default="http://control.languagegame.io/colossus/generate",
                        help="URL of the inference API to test. Default is http://control.languagegame.io/colossus/generate")
    parser.add_argument("--num_tokens", type=int, default=20,
                        help="Number of tokens to generate per inference call. Default is 20.")
    parser.add_argument("--auto", action="store_true", 
                        help="Include this flag to automatically continue generating `num_tokens` more assistant responses for the rest of time. Default=False")
    args = parser.parse_args()
    main(args)
