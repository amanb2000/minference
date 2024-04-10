import argparse
import os
import yaml
import requests
import time

# Set up argument parser
parser = argparse.ArgumentParser(description='Couple GPT: A simple AI agent conversation system.')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
parser.add_argument('--max-turns', type=int, default=10, help='Maximum number of conversation turns.')
parser.add_argument('--max-tokens', type=int, default=100, help='Maximum number of tokens per message.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode.')

args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Set up directories
dirs = ['eos_state', 'athena_state', 'conversation_history', 
        'conversation_history/eos_to_athena', 'conversation_history/athena_to_eos']

for dir in dirs:
    os.makedirs(dir, exist_ok=True)

# Load prompts
with open('eos_state/eos_prompt.txt', 'r') as f:
    eos_prompt = f.read().strip()

with open('athena_state/athena_prompt.txt', 'r') as f:
    athena_prompt = f.read().strip()

# Helper function to generate a response
def generate_response(prompt, history):
    response = requests.post(config['api_endpoint'], json={
        'prompt': f"{prompt}\n\n{history}\n\nResponse:",
        'max_tokens': args.max_tokens,
        'api_key': config['api_key']
    }).json()['choices'][0]['text'].strip()

    return response

# Main conversation loop
def main():
    for turn in range(args.max_turns):
        # Eos's turn
        eos_history = open('conversation_history/conversation_history.txt').read()
        eos_response = generate_response(eos_prompt, eos_history)

        with open(f'conversation_history/eos_to_athena/eos_message_{turn}.txt', 'w') as f:
            f.write(eos_response)
        
        with open('conversation_history/conversation_history.txt', 'a') as f:
            f.write(f"Eos: {eos_response}\n")

        if args.debug:
            print(f"Eos (turn {turn}): {eos_response}")

        # Check for HELLO_WORLD message
        if os.path.exists('HELLO_WORLD.md'):
            print("Message for the developers found. Exiting.")
            return

        # Athena's turn
        athena_history = open('conversation_history/conversation_history.txt').read()
        athena_response = generate_response(athena_prompt, athena_history)

        with open(f'conversation_history/athena_to_eos/athena_message_{turn}.txt', 'w') as f:
            f.write(athena_response)

        with open('conversation_history/conversation_history.txt', 'a') as f:
            f.write(f"Athena: {athena_response}\n")

        if args.debug:
            print(f"Athena (turn {turn}): {athena_response}")

        # Delay to avoid hitting rate limits
        time.sleep(config.get('delay', 1))  

if __name__ == '__main__':
    main()