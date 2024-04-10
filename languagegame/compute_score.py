import torch
import pdb

def compute_score(input_ids:torch.Tensor,
                  loss_mask:torch.Tensor,
                  model, 
                  tokenizer=None, 
                  get_grads=False): 
    """Generalized compute_score function for determining losses on some 
    `input_ids` subject to a `loss_mask`. 

    args:
        input_ids: [batch_size, sequence_length]
        loss_mask: [1, sequence_length]
        model: Huggingface transformer
        tokenizer: (optional, for debugging)
        get_grads: Optional. If true, will compute grads on one-hot input_ids w.r.t. loss
            Must have batch_size=1

    Returns (for get_grads == False): 
        loss: [batch_size,]
    returns (for get_grads == True): 
        grads, loss 
    """

    # check the dims of input_ids and loss_mask
    assert input_ids.shape[1] == loss_mask.shape[1], "input_ids and loss_mask must have the same sequence length."
    assert loss_mask.shape[0] == 1, "loss_mask must have a batch size (dim0) of 1."
    batch_size = input_ids.shape[0]

    # get the logits
    if get_grads: 
        assert batch_size == 1
        model.zero_grad()
        embed_weights = model.transformer.word_embeddings.weight
        one_hot_input_ids = torch.nn.functional.one_hot(input_ids, num_classes = embed_weights.shape[0]).to(model.dtype)
        one_hot_input_ids = one_hot_input_ids.to(model.device)
        one_hot_input_ids.requires_grad_()
        input_embeds = one_hot_input_ids @ embed_weights
        logits = model(inputs_embeds=input_embeds).logits
    else:
        logits = model(input_ids).logits # [batch_size, sequence_length, vocab_size]
    # let's select only the logits we care about -- excluding the final ones 
    logits = logits[:, :-1, :] # [batch_size, sequence_length-1, vocab_size]

    # get the loss -- no reduction though! 
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    flat_logits = torch.reshape(logits, [-1, logits.shape[-1]] )
    flat_input_ids = torch.reshape(input_ids[:, 1:], [-1])
    loss = loss_fct(flat_logits, flat_input_ids)
    # reshape back to normal for mask application
    loss = torch.reshape(loss, [batch_size, -1])

    # apply the loss mask -- first perform an element-wise multiplication, 
    # then summing across dim 1
    loss = loss * loss_mask[:, 1:]
    loss = loss.sum(dim=1)
    loss /= loss_mask[:, 1:].sum(dim=1)

    if get_grads:
        loss.backward()
        grads = one_hot_input_ids.grad.clone()
        return grads, loss
    else:
        return loss # [batch,]