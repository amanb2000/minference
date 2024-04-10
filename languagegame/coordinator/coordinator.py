"""Business logic for the coordinator node.
"""
from typing import Dict, Any
import uuid
import random
import requests
from languagegame.models import InferenceServerModel, AgentServerModel, CorpusChunkModel, NeighborModel, GameReq, SystemPromptModel
from .dataset import LexFridmanDataLoader
from datasets import load_dataset
from datetime import datetime 
import threading
import random
from openai import AsyncOpenAI

import os

import asyncio 
import httpx
import numpy as np

import pdb


class Coordinator: 
    """ Business logic for the inference server.

    Provides methods for node registration, status checking, heartbeat, 
    and administrating language games.
    """

    def __init__(self, port:int,
                 host:str="0.0.0.0", 
                 dataset:str="lex_fridman", 
                 num_chars_per_chunk:int = 550,
                 random_seed:int = 42): 
        self.port = port 
        self.host = host 

        self.inference_servers = []
        self.agent_servers = []
        self.agent_status = {} # dict of agent status, keyed by UID
        # self.exchange_cond = threading.Condition() # Condition variable for waiting on agents to finish message exchange
        self.exchange_cond = asyncio.Condition()
        self.iterate_game = False # flag for whether we want the worker thread to invoke *another* round of the 

        self.action_log = [] # String log of all actions taken by the coordinator

        self.allowable_datasets = ['lex_fridman', 'none']

        assert dataset in self.allowable_datasets, "Dataset not found"
        self.dataset_name = dataset
        self.num_chars_per_chunk = num_chars_per_chunk
        self.random_seed = random_seed

        self.status = "Uninitialized Topology"


        self.init_openai_api() # grabs the key from ~/.bash_profile, sets up self.oai_client

        self.init_dataset()

    def init_openai_api(self):
        # %% set the API key environment variable
        # os.environ["OPENAI_API_KEY"] = "your-openai-key"
        # Read the .bash_profile file
        bash_profile = os.path.expanduser("~/.bash_profile")
        keys_added = []
        with open(bash_profile) as file:
            for line in file:
                line = line.strip()
                if line.startswith("export OPENAI_API_KEY"):
                    key, value = line.replace("export ", "").split("=", 1)
                    os.environ[key] = value.strip('"\'')  # Update the environment
                    keys_added += [key]
                    print('Added key: ', key, ' with value: ', value)
        assert 'OPENAI_API_KEY' in keys_added, "Could not find OPENAI_API_KEY in ~/.bash_profile. Please add it and try again."
        print("Passing key: ", os.environ['OPENAI_API_KEY'], " to AsyncOpenAI.")
        self.oai_client = AsyncOpenAI(max_retries=10, timeout=45, api_key=os.environ['OPENAI_API_KEY'])
        self.mutation_prompt = """You are an expert prompt engineer generating
        mutated versions of system prompts for LLMs in a multi-LLM distributed
        system. You will be provided with the best prompts discovered to date,
        and you will mutate them in an effort to improve them. Make sure to
        maintain the general structure and most of the text, but change some
        portion of the prompt in an effort to make it better. Your response
        should include the mutated prompt and only the mutated prompt. No other
        text should be in your response. Try adding some new information/novel 
        conversational techniques! This is the agent's 'genetic code', so try 
        adding some new 'genes' that will propagate to the next generations of 
        agents."""

    def init_dataset(self): 
        if self.dataset_name == "lex_fridman": 
            self.dataset = load_dataset("Whispering-GPT/lex-fridman-podcast")
            self.data_loader = LexFridmanDataLoader(self.dataset['train'], 
                                                    num_chars_per_chunk=self.num_chars_per_chunk, 
                                                    random_seed=self.random_seed)
        elif self.dataset_name == "none": 
            self.dataset = None
            self.data_loader = None
            print("WARNING: No dataset provided. You will need to manually broadcast corpus chunks to the agents.")
        else: 
            raise Exception(f"Dataset {self.dataset_name} not implemented yet.")

    async def run_game_endpoint(self, num_iters, generation_period, death_fraction):
        """ API endpoint that just creates a thread to run the real `run_game` 
        function then immediately returns to the user. This frees up the main 
        thread to respond to further requests (especially requests from Agent 
        nodes that must confirm their message exchange) while the message exchange 
        is happening. 
        """
        # threading.Thread(target=self.run_game, args=(num_iters,generation_period,death_fraction,)).start()
        await self.run_game(num_iters, generation_period, death_fraction)
        # _thread = threading.Thread(target=asyncio.run, args=(self.run_game(num_iters,generation_period,death_fraction)))
        # task = asyncio.create_task()

        return {'message': 'success, running game'}


    def _progress_status(self, iter, num_iters, start_time): 
        """ Helper function for printing the current status of the game. 
        """
        cur_time = datetime.now()
        time_elapsed = cur_time - start_time
        time_elapsed = time_elapsed.total_seconds()
        time_per_iter = time_elapsed / (iter+1)
        time_remaining = (num_iters - iter) * time_per_iter

        # format time remaining as hours:minutes
        time_remaining = str(datetime.fromtimestamp(time_remaining))
        return f"{iter+1}/{num_iters} iterations completed. Estimated time remaining: {time_remaining} seconds"

    async def run_game(self, num_iters:int, generation_period:int=-1, death_fraction:float=-1.0):
        """ Runs a game of `num_iters` with the current network of agents and 
        whatever dataloader is currently loaded.

        Each round consists of broadcasting a corpus chunk, then initiating a 
        message exchange, then waiting until all the agents have confirmed. 
        """
        # Checking the inputs
        assert ( (death_fraction >= 0) == (generation_period > 0) ), "Death fraction must be positive if generation period is also positive (and vice versa)"

        print("Running game...")
        print("Self.dataset type: ", type(self.dataset))
        assert not (self.dataset is None), "No dataset provided. Cannot run game."

        self.action_log.append({'action': 'start_game', 'num_iters': num_iters, 'time': datetime.now()})

        start_time = datetime.now()

        self.status = "Running Game -- Iteration 0 of {}".format(num_iters)
        for i in range(num_iters):
            # Check if we are due for a generational culling.
            if i > 0 and generation_period > 0 and i % generation_period == 0:
                print("\n\n=======================")
                print("=== Mutating agents ===")
                print("=======================")
                print("Generation: ", i // generation_period)
                # self.mutate_agents(num_recent=generation_period, death_fraction=death_fraction)
                # asyncio.run(self.mutate_agents(num_recent=generation_period, death_fraction=death_fraction))
                await self.mutate_agents(num_recent=generation_period, death_fraction=death_fraction)

            self.action_log.append({'action': 'start_round', 'round': i, 'time': datetime.now()})
            # first we broadcast the corpus chunk 
            chunk, supplemental_info = self.data_loader.get_next_chunk()
            print("Broadcasting chunk ", chunk[:10], "...")
            if chunk is None:
                print("Reached the end of the dataset.")
                break
            self.broadcast_corpus(CorpusChunkModel(
                uid="chunk-"+uuid.uuid4().hex,
                corpus_string=chunk
            ))

            # then we initiate the message exchange
            print("Initiating message exchange...")
            result = self.initiate_message_exchange()
            print("Result of message exchange: ", result)

            # wait on the self.exchange_condition until all agents have confirmed
            # that they have finished the message exchange
            print("Waiting in condition variable for message exchange to all confirm")
            async with self.exchange_cond:
                while not self._all_confirmed(): 
                    await self.exchange_cond.wait()
            print("All agents have confirmed. Moving on to next round.")
            
            # now we're done! We can do the next round :) 
            print("Done with round ", i)

            self.action_log.append({'action': 'end_round', 'round': i, 'time': datetime.now()})
            self.status = "Running Game -- with {} iters".format(self._progress_status(i, num_iters, start_time))

        self.status = "Ready to Start Game"
        self.action_log.append({'action': 'end_game', 'num_iters': num_iters, 'time': datetime.now()})

    def _all_confirmed(self): 
        # returns true if all the agents have returned a `confirm_exchange` message
        # use this with a mutex though :) 
        for uid in self.agent_status.keys(): 
            if self.agent_status[uid] != "Ready": 
                return False
        return True


    def register_inference_server(self, server:InferenceServerModel): 
        """ Registers an inference server with the coordinator.
        """
        # Check if UID is "None"
        if server.uid is None or server.uid == "None":
            print("Assigning UID to new inference server...")
            server.uid = "llm-"+uuid.uuid4().hex[:8]
            print("Assigned UID: {}".format(server.uid))

        self.inference_servers.append(server)
        print(f"Registered inference server: {server.uid} at {server.ip_address}:{server.port}")

        # Add note in action log
        self.action_log.append({'action': 'register_inference_server', 'server': server.model_dump()})

        return server
    
    def register_agent_server(self, server:AgentServerModel): 
        """ Registers an agent server with the coordinator.
        """
        if len(self.inference_servers) == 0: 
            raise Exception("No inference servers registered. Cannot register agent server.")

        # Check if UID is "None"
        if server.uid is None or server.uid == "None":
            print("Assigning UID to new agent server...")
            server.uid = "agent-"+uuid.uuid4().hex[:8]
            print("Assigned UID: {}".format(server.uid))
        if server.inference_server.uid == "None": 
            # select a random inference server from self.inference_servers
            # TODO: implement a more sophisticated load balancing algorithm
            server.inference_server = random.choice(self.inference_servers)
            server.inference_server = self.inference_servers[0]

        self.agent_servers.append(server)
        self.agent_status[server.uid] = "Ready"
        print(f"Registered agent server: {server.uid} at {server.ip_address}:{server.port}")

        # Add note in action log
        self.action_log.append({'action': 'register_agent_server', 'server': server.model_dump()})

        return server
    
    def unregister_agent_server(self, server:AgentServerModel): 
        """ Unregisters an agent server with the coordinator.
        """
        print("[Coordinator] trying to unregister server ", server)
        print("[Coordinator] Agent server UIDs: ", [s.uid for s in self.agent_servers])
        uid = server.uid
        len_before = len(self.agent_servers)
        self.agent_servers = [s for s in self.agent_servers if s.uid != uid]
        len_after = len(self.agent_servers)

        if len_before == len_after:
            raise Exception("Server with uid {} not found".format(uid))
        print(f"Unregistered agent server: {server.uid} at {server.ip_address}:{server.port}")

        if server.uid in self.agent_status.keys(): 
            self.agent_status.pop(server.uid)

        # Add note in action log
        self.action_log.append({'action': 'unregister_agent_server', 'server': server.model_dump()})

        return server
    
    def unregister_inference_server(self, server:InferenceServerModel):
        """ Unregisters an inference server with the coordinator.
        """
        uid = server.uid
        len_before = len(self.inference_servers)
        self.inference_servers = [s for s in self.inference_servers if s.uid != uid]
        len_after = len(self.inference_servers)

        if len_before == len_after:
            raise Exception("Server with uid {} not found".format(uid))
        print(f"Unregistered inference server: {server.uid} at {server.ip_address}:{server.port}")

        # Add note in action log
        self.action_log.append({'action': 'unregister_inference_server', 'server': server.dict()})

        return server
    

    def get_status(self): 
        """ Returns a dict with the server list and agent list.
        """
        return {
            "status": self.status,
            "inference_servers": self.inference_servers,
            "agent_servers": self.agent_servers,
            "action_log": self.action_log,
            "agent_status": self.agent_status
        }


    def _call_add_neighbor_endpoint(self, agent, neighbor):
        """Send a request to the agent's add_neighbor endpoint with the given neighbor's details."""
        endpoint = f"http://{agent.ip_address}:{agent.port}/add_neighbor"
        neighbor_data = {
            "uid": neighbor.uid,
            "ip_address": neighbor.ip_address,
            "port": neighbor.port
        }
        response = requests.post(endpoint, json=neighbor_data)
        if response.status_code != 200:
            raise Exception(f"Failed to add {neighbor.uid} as a neighbor for {agent.uid}. Response: {response.text}")
        

    def _call_remove_neighbor_endpoint(self, agent: AgentServerModel, neighbor_uid: str):
        """Send a request to the agent's remove_neighbor endpoint with the given neighbor's UID."""
        endpoint = f"http://{agent.ip_address}:{agent.port}/remove_neighbor"
        data = {
            "uid": neighbor_uid
        }
        response = requests.post(endpoint, json=data)
        if response.status_code != 200:
            print(f"Failed to remove {neighbor_uid} as a neighbor for {agent.uid}. Response: {response.text}")


    def initialize_game(self, k=4, p=0.1):
        """Initialize the game by constructing a small-world network of agents:
        A small-world network is a mathematical graph in which most nodes are not neighbors of one another, but the neighbors of any given node are likely to be neighbors of each other. 
        Due to this, most neighboring nodes can be reached from every other node by a small number of hops or steps.
        k is the initial number of neighbors for each agent
        p is the probability of rewiring each edge"""
        num_agents = len(self.agent_servers)
        
        if num_agents < k + 1:
            print("Not enough agents to form the desired network.")
            return
        
        for i in range(len(self.agent_servers)): 
            cur_node = self.agent_servers[i]
            neighbor_to_add = self.agent_servers[(i+1) % num_agents] # ring topology for now

            print(f"Adding neighbor: {i} -> {(i+1)%num_agents}", neighbor_to_add.uid, " to ", cur_node.uid)
            self._call_add_neighbor_endpoint(cur_node, neighbor_to_add) 
            # _call_add_neighbor_endpoint will raise an error if it's not successful
            # updating our local copy 

            cur_node.neighbors.append(NeighborModel(
                uid=neighbor_to_add.uid,
                ip_address=neighbor_to_add.ip_address,
                port=neighbor_to_add.port
            ))

            neighbor_to_add.neighbors.append(NeighborModel(
                uid=cur_node.uid,
                ip_address=cur_node.ip_address,
                port=cur_node.port
            ))

        # add to action log 
        self.status = "Ready to Start Game"
        self.action_log.append({'action': 'initialize_game', 'k': k, 'p': p})


        # Start with a regular ring lattice
        """
        for i, agent in enumerate(self.agent_servers):
            neighbors = [(i + offset) % num_agents for offset in range(-k//2, k//2 + 1) if offset != 0]
            
            for neighbor_index in neighbors:
                neighbor = self.agent_servers[neighbor_index]
                self._call_add_neighbor_endpoint(agent, neighbor)

        # Rewiring step to introduce the small-world property
        for i, agent in enumerate(self.agent_servers):
            if random.random() < p:
                # Choose a non-neighbor agent randomly
                non_neighbors = [a for a in self.agent_servers if a not in agent.neighbors and a != agent]
                new_neighbor = random.choice(non_neighbors)
                
                # Remove a random existing neighbor
                removed_neighbor = random.choice(agent.neighbors)
                self._call_remove_neighbor_endpoint(agent, removed_neighbor['uid'])
                
                # Add the new neighbor
                self._call_add_neighbor_endpoint(agent, new_neighbor)
        """

    def broadcast_corpus(self, chunk: CorpusChunkModel):
        """ Broadcasts `chunk` to all agents in the network. 
        Invokes the agent's /process_corpus endpoint with the 
        chunk.
        """
        # Assign a uid if chunk.uid == "None"
        if chunk.uid is None or chunk.uid == "None":
            print("Assigning UID to corpus chunk...")
            chunk.uid = "chunk-"+uuid.uuid4().hex
            print("Assigned UID: {}".format(chunk.uid))

        for agent in self.agent_servers: 
            endpoint = f"http://{agent.ip_address}:{agent.port}/process_corpus"
            response = requests.post(endpoint, json=chunk.model_dump())
            if response.status_code != 200:
                print(f"Failed to broadcast corpus chunk to {agent.uid}. Response: {response.text}")

        # Add to action log
        self.action_log.append({'action': 'broadcast_corpus', 'chunk_uid': chunk.uid})



    def initiate_message_exchange(self): 
        """ This function commands all the agents to start exchanging messages. 
        Note that the server should wait to receive a `confirm_exchange/` message 
        from all the agents before proceeding to the next round. 
        """
        # Check that the last action was `broadcast_corpus` and not `initiate_message_exchange`. 
        # You may need to traverse the action log to check this.
        if len(self.action_log) == 0: 
            raise Exception("No actions have been performed yet. Cannot initiate message exchange.")

        # Check that all self.agent_status == "Ready"
        for agent in self.agent_servers:
            if self.agent_status[agent.uid] != "Ready": 
                raise Exception("Cannot initiate message exchange. Agent {} is not ready.".format(agent.uid))

        # Loop through the action log in reverse order. 
        # If we see a `broadcast_corpus` action, then we can proceed.
        # If we see an `initiate_message_exchange` action first, then we raise 
        # an error. 
        found_broadcast = False
        for i in range(len(self.action_log)-1, -1, -1): 
            action = self.action_log[i]['action']
            if action == 'broadcast_corpus': 
                found_broadcast = True
                break
            elif action == 'initiate_message_exchange': 
                raise Exception("Cannot initiate message exchange. Last action was `initiate_message_exchange` -- must broadcast new corpus chunk before next message exchange.")
        
        if not found_broadcast: 
            raise Exception("Cannot initiate message exchange. No `broadcast_corpus` action found in action log.")

        print("Coordinator will now call each agent's `init_message_exchange` endpoint.")

        # Send a `initiate_message_exchange` message to all agents
        num_failures = 0
        for agent in self.agent_servers: 
            self.agent_status[agent.uid] = "Exchanging"
            endpoint = f"http://{agent.ip_address}:{agent.port}/init_message_exchange"
            response = requests.get(endpoint)
            if response.status_code != 200:
                print(f"Failed to initiate message exchange with {agent.uid}. Response: {response.text}")
                self.agent_status[agent.uid] = "Error"
                num_failures += 1
            else: 
                print(f"Successfully called {agent.uid}'s `init_message_exchange` endpoint.")
            if response.json()['message'] != "success": 
                num_failures += 1


        if num_failures == len(self.agent_servers): 
            raise Exception("Failed to initiate message exchange with any agents.")
        
        self.action_log.append({'action': 'initiate_message_exchange', 'num_failures': num_failures})

        if num_failures > 0: 
            return {
                "message": "semi-failure",
                "num_failures": num_failures, 
                "num_agents": len(self.agent_servers)
            }
        else:
            return {
                "message": "success",
                "num_failures": num_failures, 
                "num_agents": len(self.agent_servers)
            }

    async def confirm_exchange(self, agent: AgentServerModel): 
        """ This function is called by the agent server to confirm that it has 
        finished the message exchange. 
        """
        assert self.agent_status[agent.uid] == "Exchanging", "Cannot confirm exchange. Agent is not exchanging messages."

        self.agent_status[agent.uid] = "Ready"
        print(f"Agent {agent.uid} is now RE")

        self.action_log.append({'action': 'confirm_exchange', 'agent': agent.uid})
        # assert all the rest of the fields of agent are correct
        async with self.exchange_cond: 
            self.exchange_cond.notify_all()
        return {
            "message": "success"
        }
    



    def get_agent_scores(self, num_recent=10):
        # Fetch data from the coordinator
        data = self.get_status() 
        # print("Self status type: ", type(data))
        # print("Self status data: ", data)
        # return data

        # Extract IP:port information of each agent
        # TODO: Make this async
        agent_data = [requests.get(f"http://{agent.ip_address}:{agent.port}/full_status").json() for agent in data['agent_servers']]

        # print("Agent data: ", agent_data)

        # Extracting the loss values and the means, stds for each agent over time.
        processed_data = []
        for agent in agent_data:
            loss_values = [entry['loss'] for entry in agent['corpus_log'] if 'corpus_string' in entry]
            loss_values = loss_values[-num_recent:]

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

        return processed_data

    def _get_agent_prompt(self, uid:str): 
        """ Sends a request to the agent server with the given UID at its
        get_system_prompt/ endpoint and returns the agent's system prompt
        """
        # 1: get the agent server with the given UID from self.agent_servers
        agent = [a for a in self.agent_servers if a.uid == uid][0]
        # make sure it exists
        assert agent is not None, "[_get_agent_prompt()] Agent with UID {} not found".format(uid)

        # 2: send a request to the agent's get_system_prompt/ endpoint

        """
        prompt = requests.get(f"http://localhost:{agent_port}/get_system_prompt").json()
        prompt = prompt['system_prompt']
        print(f"Current system prompt: {prompt}")
        """

        endpoint = f"http://{agent.ip_address}:{agent.port}/get_system_prompt"
        response = requests.get(endpoint)
        if response.status_code != 200:
            raise Exception(f"Failed to get agent prompt. Response: {response.text}")
        return response.json()['system_prompt']
    
    def _set_agent_prompt(self, uid:str, new_prompt:str): 
        """ Sends a request to the agent server with the given UID at its
        set_system_prompt/ endpoint and returns the agent's system prompt
        """
        # 1: get the agent server with the given UID from self.agent_servers
        agent = [a for a in self.agent_servers if a.uid == uid][0]
        # make sure it exists
        assert agent is not None, "[_set_agent_prompt()] Agent with UID {} not found".format(uid)

        # 2: send a request to the agent's get_system_prompt/ endpoint
        payload = SystemPromptModel(system_prompt=new_prompt).model_dump()
        endpoint = f"http://{agent.ip_address}:{agent.port}/set_system_prompt"
        response = requests.post(endpoint, json=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to get agent prompt. Response: {response.text}")
        print("RESPONSE FROM SET_AGENT_PROMPT: ", response.json())

    async def _mutate_prompt(self, original_prompt:str): 
        """ Uses the self.oai_client to mutate the original_prompt and return
        the mutated prompt. 
        """
        try:
            completion = await self.oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.mutation_prompt},
                    {"role": "user", "content": f"Original system prompt: {original_prompt}"}
                ]
            )
            mutant = completion.choices[0].message.content
        except Exception as e: 
            print("Error in _mutate_prompt during OpenAI Request." + str(e))
            mutant = original_prompt
        return mutant


    async def mutate_agents(self, num_recent, death_fraction): 
        """ Mutates the system prompt of the lowest death_fraction 
        agents based on the top `1-death_fraction` agents. 
        """
        assert self._all_confirmed()
        print("Death fraction: ", death_fraction)
        agent_scores = self.get_agent_scores(num_recent=num_recent)

        self.status = "Mutating Agents"
        # agent_scores is a list of dicts with keys:
        #   - uid
        #   - loss_values (num_recent)
        #   - mean_loss (based on num_recent loss_values)
        #   - std_loss (based on num_recent loss_values)


        # get the top (1-death_fraction) agents
        num_alive = round((1-death_fraction) * len(agent_scores))
        num_dead = len(agent_scores) - num_alive

        print("Num alive: ", num_alive)
        print("Num dead: ", num_dead)

        alive_agent_uids = [agent_scores[i]['uid'] for i in range(num_alive)]
        dead_agent_uids = [agent_scores[i]['uid'] for i in range(num_alive, len(agent_scores))]
        print("Alive agent uids: ", alive_agent_uids)
        print("Dead agent uids: ", dead_agent_uids)

        # Now we loop through the dead agents and set their system prompt as a 
        # mutated version of a randomly selected alive agent. 
        cnt=0
        tasks = []
        old_prompts = []
        for dead_agent_uid in dead_agent_uids: 
            print("TODO: Mutate agent ", dead_agent_uid)
            # select a random alive agent
            alive_agent_uid = random.choice(alive_agent_uids)

            # mutate the system prompt of the dead agent
            # get the alive agent's current system prompt 
            alive_agent_prompt = self._get_agent_prompt(alive_agent_uid)

            # mutate the prompt
            task = self._mutate_prompt(alive_agent_prompt)

            tasks.append(task)
            old_prompts.append(alive_agent_prompt)
            cnt += 1

        # Run all mutate prompt tasks concurrently
        print("Waiting on OpenAI Requests...")
        mutated_prompts = await asyncio.gather(*tasks)
        print("Done waiting on OpenAI Requests!")

        # Process the results
        num_failed = 0
        for dead_agent_uid, mutated_prompt, old_prompt in zip(dead_agent_uids, mutated_prompts, old_prompts):
            if mutated_prompt == old_prompt: 
                num_failed += 1

            self._set_agent_prompt(dead_agent_uid, mutated_prompt)
            cnt += 1

        fraction_failed = float(num_failed) / float(num_dead)
        print("Fraction of requests failed: ", fraction_failed)
        if fraction_failed > 0.9: 
            print("[WARNING -- OPENAI FAILURE TO MUTATE] Too many prompts failed to mutate due to OpenAI API errors. Continuing to next iteration.")

        # add this to the log 
        self.action_log.append({'action': 'mutate_agents', 'num_recent': num_recent, 'death_fraction': death_fraction, 'num_failed': num_failed, 'fraction_failed': fraction_failed})
        self.status = "Ready to Start Game"

        return {
            'status': 'done mutating!', 
            'death_fraction': death_fraction, 
            'num_recent': num_recent, 
            'alive_agent_uids': alive_agent_uids, 
            'dead_agent_uids': dead_agent_uids
        }


            

        
