import matplotlib.pyplot as plt
import numpy as np

# Simulate attention weights (replace with actual weights from BERT)
attention_weights = np.array([
    [0.1, 0.8, 0.1],  # Word 1 attends mostly to Word 2
    [0.3, 0.2, 0.5],  # Word 2 attends somewhat to Word 1 and Word 3
    [0.7, 0.2, 0.1]   # Word 3 attends mostly to Word 1
])

# Create a heatmap
plt.imshow(attention_weights, cmap='viridis', interpolation='nearest')

# Add labels and title
words = ['Word 1', 'Word 2', 'Word 3']
plt.xticks(range(len(words)), words)
plt.yticks(range(len(words)), words)
plt.xlabel('Target Words')
plt.ylabel('Source Words')
plt.title('Attention Weights Heatmap')


# Add a colorbar for better interpretation
plt.colorbar(label='Attention Weight')

# Show the plot
plt.show()
