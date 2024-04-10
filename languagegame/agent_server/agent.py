import os
import datetime
import requests
import public_ip as ip
import json 
import uuid
import pdb
from queue import Queue
from languagegame.models import InferenceServerModel, AgentServerModel, MessageModel, CorpusChunkModel, NeighborModel, LossRequest, ContextChunkModel, GenRequest, SystemPromptModel
import threading

class Agent:
    def __init__(self, 
                 port:int,
                 num_messages_per_iteration:int=7,
                 num_messages_retained:int=2,
                 num_message_tokens:int=50,
                 coordinator_url:str = "http://lancelot.languagegame.io:8000",
                 host:str = "0.0.0.0"):
        # Initialization parameters
        self.num_messages_per_iteration = num_messages_per_iteration
        self.num_messages_retained = num_messages_retained
        self.num_message_tokens = num_message_tokens
        self.neighbors = []  # List of AgentServerTemplates
        self.llm_inference_node = None  # Instance of InferenceServerTemplate
        self.corpus_log = []  # List of dictionaries with context and corpus
        self.chat_log = {}  # Dict with UID as key and list of messages as value
                            # Only holds the log for the current iteration.
        self.chat_archive = {} # Dict with UID as key and list of messages as value
                               # This one is used to for long-term storage. 
        self.action_log = [] # List of dictionaries with actions and timestamps
        self.status = "Ready" # Ready, Exchanging, Error

        self.coordinator_url = coordinator_url
        self.host = host
        self.port = port
        self.ip_addr = str(ip.get())
        self.uid = "None"  # assigned during registration with the coordinator
        self.message_queue = Queue() # TODO: Make this a Queue for thread safety for multi-threaded message handlind.
        self.cond_var = threading.Condition()

        self.init_system_prompt()
        self.register_with_coordinator()

    def full_status(self):
        d = self.__dict__.copy()
        d.pop('cond_var', None)  # remove the condition_variable from the dict
        d.pop('message_queue', None)
        assert 'cond_var' not in d
        return d


    def serialize_state(self):
        """Converts the state of the agent into a JSON string."""
        return json.dumps(self.__dict__)

    def deserialize_state(self, state_str):
        """Loads the state from a JSON string."""
        state = json.loads(state_str)
        for key, value in state.items():
            setattr(self, key, value)

    def save_state_to_disk(self, filename="agent_state.json"):
        """Saves the current state to a file."""
        with open(filename, 'w') as f:
            f.write(self.serialize_state())

    def load_state_from_disk(self, data_dir="data/agent_state.json"):
        """Loads the state from a file, if it exists."""
        filename = data_dir + self.uid
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                state_str = f.read()
                self.deserialize_state(state_str)


    def register_with_coordinator(self): 
        """ Register with coordinator node at self.coordinator address
        """
        print(f"[{self.uid}] Registering with the coordinator at ", self.coordinator_url)
        # print("Agent server details: ", self.get_self_model().model_dump())
        registration_url = f"{self.coordinator_url}/register_agent_server"
        response = requests.post(registration_url, json=self.get_self_model().model_dump())
        if response.status_code == 200:
            print("Successfully registered with the coordinator.")
            self.uid = response.json()["uid"]
            self.llm_inference_node = InferenceServerModel(**response.json()['inference_server'])
            # assert that the rest of the fields match 
            assert self.ip_addr == response.json()["ip_address"]
            assert self.port == response.json()["port"]
        else: 
            raise Exception(f"Failed to register with coordinator. Response: {response.json()}")
        print(f"[{self.uid}] Successfully registered!")
        
    def unregister_from_coordinator(self): 
        print(f"[{self.uid}] Unregistering from the coordinator at ", self.coordinator_url)
        json_to_send = self.get_self_model().model_dump()
        response = requests.post(self.coordinator_url + "/unregister_agent_server", json=json_to_send)
        if response.status_code == 200:
            print("Successfully unregistered from the coordinator.")
        else:
            raise Exception("Failed to unregister from the coordinator. Response = {}".format(response.json()))
    
    def get_self_model(self): 
        ret_val =  AgentServerModel(
            uid=str(self.uid), 
            ip_address=str(self.ip_addr),
            port=self.port,
            neighbors=self.neighbors,
            status=self.status
        )
        if not (self.llm_inference_node is None):
            ret_val.inference_server = self.llm_inference_node
        return ret_val
    
    def get_self_neighbor_model(self):
        # Sent to other LLMs during neighbor handshake. 
        return NeighborModel(
            uid=str(self.uid),
            ip_address=str(self.ip_addr),
            port=self.port,
            go_first=False # the receiving LLM should know that we go first -- we are initiating the handshake.
        )


    def init_system_prompt(self): 
        """ Initializes the default system prompt with the correct number of 
        messages per iteration, number of messages retained, number of message
        tokens, etc. 
        """

        self.system_prompt = f"""You are part of a collective of language models
        that must leverage cooperative communication to perform prediction on
        large-scale text data. Try to exchange useful information with your
        peers. Each exchange will have {self.num_messages_per_iteration}
        messages back and forth per iteration, and the final
        {self.num_messages_retained} will be written to your context that will
        be used to perform prediction on the corpus. Each message has
        {self.num_message_tokens} each. All your peers will receive the same
        corpus chunk at the same time, so try to maximize the collective utility
        of the final {self.num_messages_retained} messages in each iteration's
        chat!"""

    def get_system_prompt(self): 
        return self.system_prompt

    def set_system_prompt(self, new_prompt:SystemPromptModel): 
        self.system_prompt = new_prompt.system_prompt

    
    
    def add_neighbor(self, neighbor:NeighborModel): 
        """ Sends a request to another agent (handshake), and adds the neighbor
        to the list of neighbors if the request is successful.

        Since we are the ones initiating the handshake, we will be the ones 
        to `go_first`. Therefore, the neighbor.go_first field should be True. 
        """
        neighbor.go_first = True
        handshake_url = f"http://{neighbor.ip_address}:{neighbor.port}/receive_neighbor_request"
        response = requests.post(handshake_url, json=self.get_self_neighbor_model().dict())  # Sending our own neighborly details

        # Check the response from the handshake -- only if successful do we 
        if response.status_code == 200 and response.json().get('status') == 'success' and response.json().get('my_uid') == neighbor.uid:
            self.neighbors.append(neighbor)
        elif response.json().get('my_uid') != neighbor.uid:
            # You can either raise an exception or log the failure, depending on your requirements. 
            # We will be conservative and raise an exception. This is sus
            raise Exception(f"UID mismatch during neighbor handshake. Expected {neighbor.uid} at {neighbor.ip_address}:{neighbor.port}, got {response.json().get('my_uid')}")
        else:
            # You can either raise an exception or log the failure, depending on your requirements
            raise Exception(f"Failed to establish handshake with neighbor {neighbor.uid}. Response: {response.json()}")

    def receive_neighbor_request(self, neighbor:NeighborModel): 
        """ Accepts a neighbor request from another agent. 
        Returns a positive result (e.g., an AgentServerModel representing 
        this agent) if successful. 
        """
        if neighbor.uid not in [n.uid for n in self.neighbors] and neighbor.uid != self.uid:
            neighbor.go_first = False # we are receiving, so the other LLM goes first. 
            self.neighbors.append(neighbor)
            return {
                    "status": "success", 
                    "message": "Neighbor added successfully",
                    "my_uid": self.uid
                }
        return {"status": "failure", "message": "Neighbor already exists"}
        

    def remove_neighbor(self, uid: str):
        """Remove a neighbor by its UID."""
        self.neighbors = [n for n in self.neighbors if n['uid'] != uid]

    def get_corpus_context(self): 
        """ NOTE: Step 1-2 could be another function triggered at the end of the
        message exchange process (thought).
         1. Get the most recent 2 messages from each chat log, concat into
         context[i] 
         2. Add context[i] to the corpus log. 
         3. Return the most recent few [context[i-2], corpus[i-2], context[i-1],
         corpus[i-1], context[i]] =: FULL_CONTEXT
        """
        # 1. Get the most recent 2 messages from each chat log, concat into context[i]
        if len(self.corpus_log) > 0 and type(self.corpus_log[-1]) == CorpusChunkModel: 
            # We must aggregate the context now!
            context_entries = []
            for uid, chats in self.chat_log.items():
                recent_messages = chats[-self.num_messages_retained:]
                # context_entry = " ".join([chat['message'] for chat in recent_messages])
                context_entry = self.message_list_to_text(recent_messages)
                context_entries.append(context_entry)
            context_i = " ".join(context_entries)
            self.corpus_log.append(
                ContextChunkModel(
                    context_string = context_i
                )
            )
        elif len(self.corpus_log) == 0: 
            self.corpus_log.append(
                ContextChunkModel(
                    context_string = " "
                )
            )

        # 2. We now take care of adding the context in the process_corpus main 
        # function. 
        # concatenated_context = " ".join(context_entries)

        # 3. Return the most recent few [context[i-2], corpus[i-2], context[i-1], corpus[i-1], context[i]]
        full_context = []
        for entry in self.corpus_log[-3:]:
            if type(entry) == ContextChunkModel:
                full_context.append(entry.context_string)
            elif type(entry) == CorpusChunkModel:
                full_context.append(entry.corpus_string)
            else: 
                print("\n\nTYPE OF ENTRY: ", type(entry))
                raise Exception("Corpus log contains an entry that is neither ContextChunkModel nor CorpusChunkModel: ", entry)

        retval = " ".join(full_context)
        if retval == '': 
            retval = ' '
        return retval


    def process_corpus(self,corpus_chunk:CorpusChunkModel): 
        """ this will be a big function 
         1. FULL_CONTEXT = get_corpus_context()
         2. Hit the loss computation API `/ce_loss` of the parent LLM inference node with {context = FULL_CONTEXT, corpus = corpus[i]}
         3. Receive the loss on the corpus[i] for the given agent 
         4. Store the loss, store the corpus[i] in the corpus log. 
         5. Transmit the score back to the coordinator node. 

        """
        # 1. FULL_CONTEXT = get_corpus_context()

        full_context = self.get_corpus_context()
        print("Calling LLM inference node with context: ", full_context)
        print("Corpus chunk string: ", corpus_chunk.corpus_string)

        # 2. Hit the loss computation API `/ce_loss` of the parent LLM inference node
        if self.llm_inference_node:
            endpoint = f"http://{self.llm_inference_node.ip_address}:{self.llm_inference_node.port}/ce_loss"
            data = LossRequest(
                context_string = full_context,
                corpus_string = corpus_chunk.corpus_string
            )
            response = requests.post(endpoint, json=data.model_dump())
            if response.status_code != 200:
                raise Exception(f"Failed to compute loss. Response from inference node {self.llm_inference_node}: {response.json()}")

            # 3. Receive the loss on the corpus[i] for the given agent
            loss = response.json().get('loss', None)
            # assert loss if of type float 
            if loss is None:
                raise Exception(f"Failed to compute loss. Response from inference node {self.llm_inference_node}: {response.json()}")

            print("Received loss value of : ", loss, " from the LLM inference node.")

            # 4. Store the loss and the corpus[i] in the corpus log
            # First get the previous corpus chunk uid if it exists 
            if len(self.corpus_log) > 0 and type(self.corpus_log[-1]) == CorpusChunkModel: 
                prev_uid = self.corpus_log[-1].uid
            else: 
                prev_uid = "None"

            # self.corpus_log.append(
            #     ContextChunkModel(
            #         context_string = full_context,
            #         loss = loss
            #     )
            # ) 
            assert type(self.corpus_log[-1]) == ContextChunkModel
            self.corpus_log[-1].loss = loss

            self.corpus_log.append(
                CorpusChunkModel(
                    corpus_string = corpus_chunk.corpus_string,
                    uid = corpus_chunk.uid, # assigned by coordinator
                    loss = loss,
                    prev_uid = prev_uid # previous corpus chunk uid (not context)
                )
            )

            # Add to action log
            self.action_log.append({'action': 'process_corpus', 'corpus_chunk': self.corpus_log[-1].model_dump(), 'context_chunk': self.corpus_log[-2].model_dump(), 'loss': loss, 'timestamp': datetime.datetime.now()})
        else: 
            raise Exception("No LLM inference node available to compute loss.")

    def receive_message(self, incoming_message: MessageModel):
        """ Receive an incoming message and store it in the message queue.

        The message queue is processed by a seperate thread. This thread is 
        launched in `languagegame/agent_server/main.py` during the startup 
        of the server. It handles the message queue asynchronously. 

        TODO: Make the handler multithreaded, use httpx instead of requests, 
        etc. 
        """
        print("Received message -- grabbing the condition variable...")
        self.message_queue.put(incoming_message)
        with self.cond_var:
            self.cond_var.notify()  # Notify the message processing thread
        print(f"[{self.uid}] Received message number {incoming_message.message_num} from {incoming_message.sender}")
        return {"status": "Received and queued message for processing."}

    def message_daemon(self): 
        """ Keeps watch on self.message_queue. When a condition variable is 
        signalled, it will come to life and process the requests in the queue. 

        # TODO: Allow this to be multi-threaded. Basically switch from 
        requests -> httpx and use a Queue object rather than a list. 
        """
        print("I am the message daemon. Fear me.")
        while True:
            with self.cond_var:
                while self.message_queue.qsize() == 0:
                    self.cond_var.wait()

                message = self.message_queue.get_nowait()
            # TODO: Replace this with a thread pool submit() call
            if message.sender == self.uid: 
                self.handle_outgoing_message(message)
            elif message.receiver == self.uid: 
                self.handle_incoming_message(message)
            else: 
                raise Exception(f"[Agent {self.uid}@message_daemon()] Message {message} is neither incoming nor outgoing. Very sus.")

    def status_daemon(self): 
        """ Call this method every time the status (self.status: Exchanging -> Ready)
        may need to change. This function will check if the status ought to change. 

        The status must change when the current status is Exchanging and we 
        realize that all our logs in self.chat_logs are full (have length 
        equal to the number of total messages per iteration).

        In that case, we should send the coordinator node a POST request to the 
        confirm_exchange/ endpoint with our `get_self_model()` as the payload.
        Once that comes back with code 200, we should set self.status = Ready.
        """
        if self.status == "Exchanging":
            # Check if all the chat logs are full. 
            for uid, chat_log in self.chat_log.items():
                if len(chat_log) < self.num_messages_per_iteration:
                    return
                
        # If we got to here, we should be done exchanging.
        print(f"Node {self.uid} is done exchanging! Notifying coordinator...")
        self.status = "Ready"
        endpoint = f"{self.coordinator_url}/conf_exchange"
        
        print(f"Node {self.uid} is done exchanging! All chat logs are full!")
        print("Sending a POST request to the coordinator to confirm the exchange.")
        response = requests.post(endpoint, json=self.get_self_model().model_dump())
        if response.status_code != 200: 
            raise Exception(f"Failed to confirm exchange with coordinator. Response: {response.json()}")
        print("Successfully confirmed exchange with coordinator.")
        return


    def handle_outgoing_message(self, outgoing_message: MessageModel):
        """ Send the outgoing message to the specified receiver. 

        CONVENTION: The outgoing message in the queue should be empty default 
        messages that only specify the outgoing UID
        """
        assert outgoing_message.message == "None"
        assert outgoing_message.uid == "None"
        assert outgoing_message.prev_uid == "None"
        assert outgoing_message.sender == self.uid
        assert outgoing_message.receiver != self.uid

        self.send_message(outgoing_message.receiver)

    def handle_incoming_message(self, incoming_message: MessageModel): 
        """ Process the received message in a seperate thread
        """
        # Extract the sender UID from the incoming message
        print("\n\nReceived message: ", incoming_message.dict(), "\n\n")
        sender_uid = incoming_message.sender

        # Check if this sender UID is already present in the chat_log
        if sender_uid not in [n.uid for n in self.neighbors]:
            raise Exception(f"Received message from {sender_uid}, but this UID is not in the list of neighbors. Cannot receive message.")
        if sender_uid not in self.chat_log.keys():
            self.chat_log[sender_uid] = []
            assert incoming_message.message_num == 0
        if sender_uid not in self.chat_archive.keys(): 
            self.chat_archive[sender_uid] = []

        # Check if the message number is 0. If so, we need to move the chat log
        # to the archive and clear the chat log.
        if incoming_message.message_num == 0:
            assert len(self.chat_log[sender_uid]) == 0 or len(self.chat_log[sender_uid]) == self.num_messages_per_iteration
            if len(self.chat_log[sender_uid]) > 0:
                self.chat_archive[sender_uid] += self.chat_log[sender_uid]
                self.chat_log[sender_uid] = []
        else: 
            assert self.chat_log[sender_uid][-1].receiver == sender_uid
            assert self.chat_log[sender_uid][-1].message_num == incoming_message.message_num - 1
        
        # Now we are good to add the message to the chat log and respond. 
        self.chat_log[sender_uid].append(incoming_message)

        # TODO: Make this a thread pool submit() call
        if incoming_message.message_num < self.num_messages_per_iteration - 1:
            self.send_message(sender_uid) # And the cycle continues.
        elif incoming_message.message_num == self.num_messages_per_iteration - 1: 
            # find the neighbor with the specified UID
            print("Received final message, flipping go_first flag!")
            checked = False
            for neighbor in self.neighbors:
                if neighbor.uid == sender_uid:
                    neighbor.go_first = not neighbor.go_first
                    checked=True
                    break
            assert checked == True, f"Very sus behavior -- received a message with {incoming_message.message_num} from {sender_uid}, but could not find the neighbor with that UID in the list of neighbors. Very very weird and bad, someone's trying to mess with us."
            print("Done flipping go_first flag!")
            print("Checking if we are actually done with the exchange...")
            self.status_daemon() # handles the case where we are the last recipient.

    def message_exchange(self):
        """ Entry point for initiating the message exchange process.

        Initiating nodes will send a message to all neighbors with
        neighbor.go_first = True. 

        We will only keep the messages for the current iteration in the 
        chat log. Therefore, we will move the messages from self.chat_log[uid]
        to self.chat_archive[uid] for all neighbors we are newly initializing
        contact with. 

        We will do the same in the receive_message function when we receive a 
        message from a neighbor with message.message_num == 0. 
        """
        assert self.status == "Ready"

        self.status = "Exchanging"

        for neighbor in self.neighbors:
            uid = neighbor.uid
            # Check that go_first is True AND that the log[uid] is either 
            # empty OR ends with a message with message_num == self.num_messages_per_iteration
            if uid not in self.chat_log.keys():
                self.chat_log[uid] = []
            if uid not in self.chat_archive.keys():
                self.chat_archive[uid] = []

            # print("Chat log: ", self.chat_log)

            if neighbor.go_first and (len(self.chat_log[uid]) == 0 or self.chat_log[uid][-1].message_num == self.num_messages_per_iteration-1):
                print("Sending message to ", uid, " with go_first = ", neighbor.go_first)
                # if the length of the message log is greater than zero, we need to 
                # move it to the archive
                if len(self.chat_log[uid]) > 0:
                    assert len(self.chat_log[uid]) == self.num_messages_per_iteration
                    self.chat_archive[uid] += self.chat_log[uid]
                    self.chat_log[uid] = []

                # Send a message to this neighbor
                assert self.uid != uid
                out_message = MessageModel(
                    receiver=str(uid), 
                    sender=str(self.uid), 
                    message = "None", 
                    uid = "None",
                    prev_uid = "None", 
                    message_num = 0
                )
                print(f"Added to outbox: {self.uid} -> {out_message.receiver}")
                # self.send_message(uid) # old synchronous call

                # New async code
                self.message_queue.put(out_message)
                neighbor.go_first = not neighbor.go_first

            with self.cond_var:
                self.cond_var.notify()
                # we will flip the go_first flag for receiver nodes in the receive_message/ 
                # endpoing

    def message_list_to_text(self, message_list):
        """ Converts a list of MessageModel objects into a single string. """
        ret_val = ""
        for message in message_list: 
            assert type(message) == MessageModel
            if message.sender == self.uid: 
                ret_val += f"\n[{message.sender}] (me): {message.message}"
            else:
                ret_val += f"\n[{message.sender}]: {message.message}"
        return ret_val


    def generate_message(self, uid) -> MessageModel:
        """ Constructs the new message for the neighbor at the specified UID. 
        
        The message is sampled from 

            P(message[uid][n] | context[i-1] + corpus[i-1] + message[uid][1:n-1])
        
        where context[i-1] is the context from the previous iteration of corpus 
        processing and corpus[i-1] is the corpus from the most recent iteration 
        corpus processing.
        """
        # Ensure we either have self.chat_log[uid] == [] or
        # self.chat_log[uid][-1].sender == uid
        if len(self.chat_log[uid]) > 0 and self.chat_log[uid][-1].sender != uid:
            raise Exception(f"Message log for {uid} is not empty, but the last message is not from {uid}. Cannot generate message to {uid}.")
    
        # Get the corpus and context from the corpus log
        corpus_i = self.corpus_log[-1]
        assert type(corpus_i) == CorpusChunkModel
        context_i = self.corpus_log[-2]
        assert type(context_i) == ContextChunkModel

        # Get the recent messages from the chat log. 
        recent_messages = self.message_list_to_text(self.chat_log[uid])

        # Assemble the full context: 
        system_prompt = self.get_system_prompt()
        full_context = f"\n[RECENT CONTEXT]\n{context_i.context_string}\n[RECENT CORPUS CHUNK]{corpus_i.corpus_string}\n[CURRENT DISCUSSION WITH {uid}]\n{recent_messages}"
        # Not that the [RECENT CONTEXT] section should contain the text of the last 
        # 2 exchanges the agent had with each neighbor -- including the UIDs! 
        # We should be testing this condition. For now we, will add every gen_request
        # to the agent's log dictionary (modelled after the Coordinator log dictionary).
        full_context += f"\n[{self.uid}] (me): "

        # print("Full context for message generation: ", full_context)

        # Get the message from the LLM inference node
        req = GenRequest(system_prompt=system_prompt, input_string=full_context, num_tokens=self.num_message_tokens)

        # send a POST request to the server
        resp = requests.post(f"http://{self.llm_inference_node.ip_address}:{self.llm_inference_node.port}/generate", json=req.model_dump())
        if resp.status_code != 200:
            raise Exception(f"Failed to generate message. Response: {resp.json()}")

        resp = resp.json()
        generated_text = resp['generated']

        # Add the generation request to the Agent's personal log.
        self.action_log.append({'action': 'generate_request', 'gen_request': req.model_dump(), 'generated_text': generated_text, 'timestamp': datetime.datetime.now()})

        prev_uid = None 
        prev_num = -1
        if len(self.chat_log[uid]) > 0: 
            prev_uid = self.chat_log[uid][-1].uid
            prev_num = self.chat_log[uid][-1].message_num
        elif len(self.chat_archive[uid]) > 0:
            prev_uid = self.chat_archive[uid][-1].uid
        else: 
            prev_uid = "None"

        print("Type of generated_text: ", type(generated_text))
        outgoing_message = MessageModel(
            receiver=str(uid), 
            sender=str(self.uid), 
            message = str(generated_text), 
            uid = str('message-'+str(uuid.uuid4().hex)),
            prev_uid = str(prev_uid), 
            message_num = prev_num + 1
        )

        return outgoing_message 


    def send_message(self, uid):
        # Send the generated message to the specified UID
        for neighbor in self.neighbors:
            if neighbor.uid == uid:
                message = self.generate_message(uid)

                endpoint = f"http://{neighbor.ip_address}:{neighbor.port}/receive_message"
                print(f"Sending message #{message.message_num} to neighbor ", neighbor.uid, "...")
                print("Endpoint: ", endpoint)
                requests.post(endpoint, json=message.model_dump())
                print("Done sending message to neighbor ", neighbor.uid, "!")
                print("Adding message to chat log")
                self.chat_log[uid].append(message)

                # Checking if this was the last message -- in which case we run status_daemon(self)
                if message.message_num == self.num_messages_per_iteration - 1:
                    self.status_daemon()

                break

