# X-LoRA: Mixture of Low-Rank Adapter Experts

X-LoRA is a flexible framework for creating adaptable large language models by dynamically mixing multiple pre-trained low-rank adapters (LoRA). It enables the development of LLMs with diverse capabilities across different domains. 
The link to the paper: https://arxiv.org/abs/2402.07148

## Overview

The primary motivation behind X-LoRA is to integrate various areas of knowledge and expertise into a single language model. This is achieved by combining multiple LoRA adapters, each fine-tuned for specific tasks or domains, at a granular token and layer level.

The approach is inspired by biological principles, where neural network building blocks are reused hierarchically, allowing for the integration of different capabilities.

## Key Features

- **Dynamic Mixing**: X-LoRA employs a scaling strategy to dynamically gate and mix LoRA adapters based on the input's hidden states, enabling new layer-wise combinations of expertise.
- **Granular Adaptation**: The scaling and mixing of LoRA adapters happen at a fine-grained token and layer level, providing a high degree of flexibility.
- **Efficient Training**: Building upon the low-rank adaptation technique, X-LoRA allows for efficient training by only updating the low-rank matrices associated with each adapter.

## Architecture of X-LoRA

The X-LoRA architecture involves:

A base model (e.g., GPT) with frozen weights.

Multiple LoRA adapters, each trained on a specific task or dataset.

An X-LoRA scaling head, which is a neural network that predicts the scalings matrix ⇤ based on the input.

<img width="874" alt="image" src="https://github.com/JiayueLiuBMI/Paper_presentation_DS5690/assets/35744343/07bb7a2c-7e5e-42cb-b90b-21c3b45bf684">


### Pseudocode for X-LoRA

```
/* X-LoRA: Mixture of Low-Rank Adapter Experts */
Input: x, the input sequence
Parameters: θ includes the following:
    W_0 ∈ ℝ^(d×k), the pretrained weight matrix.
    For each adapter i ∈ [1, n]:
        B_i ∈ ℝ^(d×r_i), A_i ∈ ℝ^(r_i×d), the decomposition matrices for adapter i.
        r_i << min(d, k), the rank of adapter i.
    Φ, the X-LoRA scaling head (a neural network).

1 | h_0 ← W_0 x  # Pretrained weight multiplication
2 | s ← Φ(x)  # Predict scalings using the scaling head
3 | for i = 1 to n:
4 |   γ_i ← s[:, i]  # Extract scaling for adapter i
5 |   h_i ← B_i * A_i * (x * γ_i)  # Apply adapter i with scaling
6 |   h_0 ← h_0 + h_i  # Combine adapter output with base output
7 | end for

return h_0
```
The X-LoRA adapter works as follows:

It starts with the output h_0 from the pretrained weight matrix W_0.

The X-LoRA scaling head Φ (a neural network) predicts the scalings s based on the input sequence x.
For each adapter i from 1 to n: a. Extract the scaling γ_i for adapter i from the predicted scalings s. b. Apply the adapter i to the input x, scaling the adapter's output by γ_i. c. Combine the scaled adapter output h_i with the base output h_0.

Return the final output h_0, which is the sum of the base output and the weighted outputs from all adapters.
The key idea behind X-LoRA is that the scaling head Φ learns to predict the appropriate scalings for each adapter based on the input sequence. This allows the model to dynamically mix and combine the outputs of different adapters, leveraging their specialized knowledge for different tasks or data distributions.

## Applications

X-LoRA framework can be applied to various domains by training domain-specific LoRA adapters.
<img width="305" alt="image" src="https://github.com/JiayueLiuBMI/Paper_presentation_DS5690/assets/35744343/c9f42f51-1bf2-4ec0-b1e3-e3cd759a79db">

The paper focuses on developing an X-LoRA model with scientific capabilities, particularly in areas such as:

- Biomaterials
- Protein mechanics
- Materials design

## Getting Started
See the notebook

Cited from https://huggingface.co/lamm-mit/x-lora

### Installation 
```
pip install git+https://github.com/EricLBuehler/xlora.git -U

```

### Example for converting a base LLM to X-LoRA

```
import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM # type: ignore

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="auto",
)

### Convert the model to X-LoRA
model_created = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora.xLoRAConfig(config.hidden_size, xlora_depth=8, device=torch.device("cuda")),
    verbose=True,
    adapters={
        "adapter_1": "./path/to/the/checkpoint_adapter_1/",
        "adapter_2": "./path/to/the/checkpoint_adapter_2/",
        "adapter_n": "./path/to/the/checkpoint_adapter_3/",
    },
)
```
### Example for loading pre-trained X-LoRA model

```
import torch
from xlora.xlora_utils import load_model  # type: ignore

XLoRA_model_name = "lamm-mit/x-lora/X-LoRA"

model, tokenizer = load_model(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    device="cuda:0",
    dtype=torch.bfloat16,
    fine_tune_model_name=XLoRA_model_name,
    adapters={
        # the adapters can be added or removed 
        "adapter_1": "lamm-mit/x-lora/X-LoRA_adapters/1/",
        "adapter_2": "lamm-mit/x-lora/X-LoRA_adapters/2/",
        "adapter_3": "lamm-mit/x-lora/X-LoRA_adapters/3/",
        "adapter_4": "lamm-mit/x-lora/X-LoRA_adapters/4/",
        "adapter_5": "lamm-mit/x-lora/X-LoRA_adapters/5/",
        "adapter_6": "lamm-mit/x-lora/X-LoRA_adapters/6/",
        "adapter_7": "lamm-mit/x-lora/X-LoRA_adapters/7/",
        "adapter_8": "lamm-mit/x-lora/X-LoRA_adapters/8/",
        "adapter_9": "lamm-mit/x-lora/X-LoRA_adapters/9/",
    },
)

```
### Example for using the trained X-LoRA model for inference
```
def generate_response (model, tokenizer, 
                      text_input="What is the best biomaterial for superior strength?",
                      num_return_sequences = 1,
                      temperature = 0.75,  
                      max_new_tokens = 127,
                      num_beams = 1,
                      top_k = 50,
                      top_p = 0.9,
                      repetition_penalty=1.,
                      eos_token_id=2, 
                      add_special_tokens=True,  
                      ):
    inputs = tokenizer(text_input,  add_special_tokens=add_special_tokens)
    with torch.no_grad():
          outputs = model.generate(input_ids = inputs["input_ids"],
                                    attention_mask = inputs["attention_mask"] ,  
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature, 
                                    num_beams=num_beams,
                                    top_k = top_k,
                                    top_p = top_p,
                                    num_return_sequences = num_return_sequences,
                                    eos_token_id=eos_token_id,
                                    pad_token_id = eos_token_id,
                                    do_sample =True, 
                                    repetition_penalty=repetition_penalty,
                                  )
    return tokenizer.batch_decode(outputs[:,inputs["input_ids"].shape[1]:].detach().cpu().numpy(), skip_special_tokens=True)

output_text=generate_response (model, tokenizer, text_input=txt,eos_token_id=eos_token,
                                           num_return_sequences=1, repetition_penalty=1.1,
                                           top_p=0.9, top_k=512, 
                                           temperature=0.5,
                                           max_new_tokens=256)

print (output_text[0])

```

### Train individual LoRA adapters for different areas of expertise.
https://github.com/EricLBuehler/xlora/blob/master/examples/simple.ipynb


## Critical Analysis 

Advantages of the X-LoRA (Mixture of Low-Rank Adapter Experts) approach:

1. **Simple implementation**: The X-LoRA approach provides a straightforward way to implement dynamic adapter mixing in existing large language models (LLMs). The provided code allows for easy integration with models in the Hugging Face ecosystem.

2. **Leverages LLM strength**: The scaling head in X-LoRA learns to exploit the inherent strengths of the base LLM by intelligently mixing and combining the outputs of different adapters, leveraging their specialized knowledge for different tasks or data distributions.

3. **Adaptability**: During training, the X-LoRA model can learn sophisticated combinatorial methods by exposing it to complex samples, such as question-answer pairs or conversational data, allowing the model to learn the best way to mix adapters for different scenarios.

4. **Interpretability**: The X-LoRA approach provides insights into how the model adapts its behavior by analyzing the scaling values and the activation patterns of different adapters for various inputs.

Disadvantages of the X-LoRA approach:

1. **Computational cost**: The X-LoRA approach requires two forward passes: one to calculate the hidden states for the scaling head, and another to apply the predicted scalings to the adapters. This additional computational cost can be a drawback, especially for resource-constrained environments.

2. **Key-value cache management**: For efficient inference in model-serving tools like vLLM, separate key-value caches need to be tracked for the scaling and forward passes, which adds complexity to the implementation.

3. **Potential for over-reliance on adapters**: While the X-LoRA approach allows for dynamic mixing of adapters, there is a risk of over-relying on the adapters and not fully utilizing the base model's capabilities, especially if the adapters are not well-trained or diverse enough.

Despite the disadvantages, the authors argue that the advantage of the X-LoRA approach is its simplicity and compatibility with existing models without the need for internal restructuring. This advantage is particularly valuable given the vast resources available in the Hugging Face ecosystem, allowing for easy adoption and experimentation with the X-LoRA technique across different models and domains.

## Resources
1. API description and examples: https://github.com/EricLBuehler/xlora/tree/master?tab=readme-ov-file
2. LoRA overview video: https://towardsdatascience.com/dive-into-lora-adapters-38f4da488ede
3. X-LoRA overview video: 

## Citation 

