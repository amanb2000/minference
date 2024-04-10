""" Shared data object (pydantic models) for the language game.
Function to interface between json and python objects, standardize 
the representation between programs that need to keep track of 
the same data.
"""

from pydantic import BaseModel


class GameInit(BaseModel):
    k: int
    p: float


class GenRequest(BaseModel):
    input_string: str
    num_tokens: int = 50
    system_prompt: str = ""

class GenResponse(BaseModel): 
    input_string: str 
    generated: str 
    num_truncated: int = 0

class LossRequest(BaseModel): 
    context_string: str
    corpus_string: str

class InferenceServerModel(BaseModel): 
    uid: str # Unique identifier for the server
    ip_address: str
    port: int
    llm_name: str
    max_seq_len: int = -1  # Default value, -1 for unknown
    batch_size: int = 1 # default value

class NeighborModel(BaseModel): 
    """ This is used as a minimal representation of the neighboring agents 
    in the AgentServerModel class. 
    """
    uid: str
    ip_address: str
    port: int
    go_first: bool = False # Do I (agent that `has` this neighbor of this) go 
                           # first in the message interaction with the 
                           # neighbor:NeighborModel in question? 


class AgentServerModel(BaseModel): 
    uid: str 
    ip_address: str
    port: int
    neighbors: list[NeighborModel]
    inference_server: InferenceServerModel = InferenceServerModel(
        uid="None", 
        ip_address="None",
        port=-1,
        llm_name="None",
        max_seq_len=-1
    )
    status: str = "Ready" # Ready, Exchanging, Error




class MessageModel(BaseModel): 
    receiver: str  # UID of the receiving agent
    sender: str  # UID of the sending agent
    message: str
    uid: str # uid of this message
    prev_uid: str # uid of the previous message
    message_num: int # which message is this in the exchange? 


class CorpusChunkModel(BaseModel): 
    corpus_string: str
    uid: str = "None" # uuid4
    prev_uid: str = "None" # UID of the previous Corpus Chunk. 
    loss: float = -1.0 # default, -1 for unknown. Populated within each model.

class ContextChunkModel(BaseModel): 
    context_string: str 
    loss: float = -1.0 # default, -1 for unknown. Populated within each model.


class SystemPromptModel(BaseModel): 
    system_prompt: str 
    uid: str = "None" # uuid4

class GameReq(BaseModel): 
    num_iters: int
    generation_period: int = -1
    death_fraction: float = -1.0

