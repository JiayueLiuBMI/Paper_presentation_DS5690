# Paper_presentation_DS5690
ReadMe author:Jiayue Liu


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

1. Train individual LoRA adapters for different areas of expertise.
2. Train the X-LoRA model using a subset of the combined training data from the individual adapters.
3. Use the trained X-LoRA model for inference, allowing it to dynamically mix the different adapters based on the input.
