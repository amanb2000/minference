# 🎮 LanguageGame Package

Welcome to the core of the LanguageGame project! Here lies the heart of our experimental collective intelligence system.

## 🏗️ Structure of the languagegame Package

 - **inference_server/**: Houses everything needed for an independent LLM inference server that hosts the collective of agents. 
     - `main.py`: The FastAPI server starting point.
     - `inference_server.py`: Houses the `InferenceServer` class -- the business logic for Inference Servers.
     - `config.json`: Default configurations tailored for the inference server (LLM type, etc). Modify to your liking! This is the default location, you can also specify an alternate path as a commandline arg when running `main.py`. 
- **agent_server/**: The independent light-weight agent servers that communicate with eachother (and the inference servers) to run the Language Game. 
     - `main.py`: Executable script to run a FastAPI HTTP agent server. 
     - `agent.py`: Houses the `Agent` class -- the business logic for Agent Servers. 
     - `config.py`: Configuration for Agent parameters. .
 - **coordinator/**: The central zookeeper of a distributed Language Game, coordinating all the agents and acting as an entry point for interacting with the collective.
     - `main.py`: Executable script to run a FastAPI HTTP coordinator server. 
     - `coordinator.py`: Houses the `Coordinator` class -- the business logic for the coordinator node. 
     - `config.py`: Set default configurations for smooth coordination.

## 🚀 Getting Started with the Package

```bash
# Navigate
cd path/to/languagegame/

#Tweak Configurations
vi languagegame/agent_server/config.py # or inference_server, or coordinator.

# Run Desired Component:
python3 languagegame/inference_server/main.py 
# or with configuration override
python3 languagegame/inference_server/main.py --config path/to/new_config.py
```

## 📚 Further Reading

Check out the [LanguageGame.io repository](https://github.com/amanb2000/LanguageGame.io) README for information information on the CLI, GUI, unittests, and experiment automation scripts. 