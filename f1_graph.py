import matplotlib.pyplot as plt

# Original and Augmented F1-Score data
original_f1_scores = [0.9629, 0.9825, 0.9878, 0.9899, 0.9923, 0.9918, 0.9930, 0.9938, 0.9942, 0.9943]
augmented_f1_scores = [0.9710, 0.9815, 0.9907, 0.9924, 0.9925, 0.9941, 0.9952, 0.9961, 0.9958, 0.9965]

epochs = list(range(10))  # 0 to 9

# Plotting the F1-Score graph
plt.figure(figsize=(12, 8))
plt.plot(epochs, original_f1_scores, label="F1-Score (Original)", marker='o', linestyle='--')
plt.plot(epochs, augmented_f1_scores, label="F1-Score (Augmented)", marker='o', linestyle='-')

plt.title("Validation F1-Score: Original vs Augmented")
plt.xlabel("Epochs")
plt.ylabel("F1-Score")
plt.legend()
plt.grid(True)
plt.show()
