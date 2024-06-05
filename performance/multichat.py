"""
This script reads prompts from a JSONL file and spawns separate processes to run a chat program for each prompt.
The output of each subprocess is redirected to a separate log file, which is updated in real-time.
Usage: python multichat.py --jsonl_file <path_to_jsonl_file>
JSONL file format:
Each line in the JSONL file should be a valid JSON object with the following keys:
- "prompt": The prompt to be passed to the chat program.
- "log_file": The path to the log file where the output of the chat program will be written.
"""
import argparse
import json
import subprocess
import time
import os
from multiprocessing import Process

def run_chat_program(prompt, log_file):
    command = "cd /Users/aman/Documents/minference && venv/bin/python3 sc_chat.py --auto --num_tokens 20"
    with open(log_file, "w") as log:
        process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=log, stderr=log, text=True, bufsize=1)
        process.stdin.write(prompt + "\n")
        process.stdin.flush()
        process.wait()

def main(args):
    # create log file directory if it does not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    processes = []
    with open(args.jsonl_file, "r") as file:
        for line in file:
            data = json.loads(line.strip())
            prompt = data["prompt"]
            log_file = os.path.join(args.log_dir, data["log_file"])
            print(f"Sending off prompt: {prompt} with log file: {log_file}")
            process = Process(target=run_chat_program, args=(prompt, log_file))
            processes.append(process)
            process.start()
            time.sleep(args.delay)
    
    for process in processes:
        process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run chat program with prompts from JSONL file.")
    parser.add_argument("--jsonl_file", required=True, help="Path to the JSONL file containing prompts and log file paths.")
    parser.add_argument("--log_dir", default="logs", help="Directory where log files will be stored (default: logs).")
    parser.add_argument("--delay", type=int, default=5, help="Delay in seconds between spawning each process (default: 5).")
    args = parser.parse_args()
    main(args)
