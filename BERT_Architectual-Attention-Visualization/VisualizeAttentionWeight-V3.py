import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True) # Crucial change

# Input sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
encoded_input = tokenizer(sentence, return_tensors='pt')

# Get the model's output
with torch.no_grad():
    outputs = model(**encoded_input)

# Extract attention weights (choose a layer)
layer_to_visualize = 0
attention_weights = outputs.attentions[layer_to_visualize].detach().numpy()


# Average attention weights across heads
num_heads = attention_weights.shape[1]
average_attention = np.mean(attention_weights, axis=1)

# Prepare for plotting (adjust according to the actual shape from attention_weights)
attention_matrix = average_attention[:,:,0].T # Simplified representation

# Create the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(attention_matrix, cmap='magma', interpolation='nearest')

# Add labels
tokens = tokenizer.tokenize(sentence)
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel('Target Tokens')
plt.ylabel('Source Tokens')
plt.title(f'Averaged Attention Weights - Layer {layer_to_visualize}')

# Add colorbar
plt.colorbar(label='Attention Weight')


# Show plot
plt.tight_layout()
plt.show()
