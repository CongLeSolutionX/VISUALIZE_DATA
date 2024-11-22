import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Model and tokenizer
model_name = "bert-base-uncased"  # Choose a pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Input sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
encoded_input = tokenizer(sentence, return_tensors='pt')

# Get the model's output
with torch.no_grad():
    outputs = model(**encoded_input)

# Extract attention weights (choose a layer - 0 is the first layer)
layer_to_visualize = 0  # Change this to visualize different layers
attention_weights = outputs.attentions[layer_to_visualize][0].detach().numpy()

# Average attention weights across the heads. (optional)
num_heads = attention_weights.shape[1]
average_attention_weights = attention_weights.mean(axis=1)

# Prepare for plotting. (Assuming 12 tokens for our example sentence in average_attention_weights)
attention_matrix = average_attention_weights[:12,:12] # Adjust according to input length


# Create the heatmap
plt.figure(figsize=(10, 8))  # Adjust figure size if needed
plt.imshow(attention_matrix, cmap='magma', interpolation='nearest')

# Add labels
tokens = tokenizer.tokenize(sentence)
plt.xticks(range(len(tokens)), tokens, rotation=90)  # Rotate x-axis labels
plt.yticks(range(len(tokens)), tokens)
plt.xlabel('Target Tokens')
plt.ylabel('Source Tokens')
plt.title(f'Averaged Attention Weights - Layer {layer_to_visualize}')

# Add colorbar
plt.colorbar(label='Attention Weight')

# Show plot
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.show()
