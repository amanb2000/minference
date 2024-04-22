import os
import requests
import json
import argparse
from datetime import datetime
from languagegame.models import GenRequest

def inference_call(req: GenRequest, API_URL, num_tokens=50):
    response = requests.post(API_URL, json=req.model_dump())

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def load_system_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

def save_chat(chat_history, file_path):
    with open(file_path, "w") as file:
        file.write("\n".join(chat_history))

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
        elif user_input.lower() == "list":
            chat_files = list_chat_files(chat_dir)
            print("Available chat files:")
            for file in chat_files:
                print(f"- {file}")
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
            input_string=input_string,
            num_tokens=args.num_tokens
        )

        result = inference_call(req, args.api_url, args.num_tokens)
        if result:
            generated_text = result["generated"]
            chat_history.append(f"Assistant: {generated_text}")
            save_chat(chat_history, chat_file)
            print(f"\n\n--------------------------------------------------------------------------------\nAssistant: {generated_text}")

        while user_input == "":
            req.num_tokens += args.num_tokens
            result = inference_call(req, args.api_url, args.num_tokens)
            if result:
                generated_text = result["generated"]
                chat_history[-1] = f"Assistant: {generated_text}"
                save_chat(chat_history, chat_file)
                print(generated_text, end="", flush=True)
            user_input = input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str, default="http://localhost:4444/generate",
                        help="URL of the inference API to test. Default is http://localhost:4444/generate")
    parser.add_argument("--num_tokens", type=int, default=50,
                        help="Number of tokens to generate. Default is 50")
    args = parser.parse_args()
    main(args)