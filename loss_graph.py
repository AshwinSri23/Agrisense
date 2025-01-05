import matplotlib.pyplot as plt

# Original and Augmented loss data
original_train_losses = [0.8204, 0.1730, 0.1253, 0.0998, 0.0844, 0.0733, 0.0674, 0.0588, 0.0568, 0.0550]
original_valid_losses = [0.1297, 0.0605, 0.0424, 0.0327, 0.0265, 0.0281, 0.0235, 0.0198, 0.0202, 0.0181]
augmented_train_losses = [0.7638, 0.1856, 0.1328, 0.1102, 0.0936, 0.0800, 0.0764, 0.0649, 0.0598, 0.0610]
augmented_valid_losses = [0.0968, 0.0580, 0.0325, 0.0266, 0.0257, 0.0193, 0.0169, 0.0144, 0.0149, 0.0135]

epochs = list(range(10))  # 0 to 9

# Plotting the loss graph
plt.figure(figsize=(12, 8))

# Original data
plt.plot(epochs, original_train_losses, label="Train Loss (Original)", marker='o', linestyle='--')
plt.plot(epochs, original_valid_losses, label="Validation Loss (Original)", marker='o', linestyle='--')

# Augmented data
plt.plot(epochs, augmented_train_losses, label="Train Loss (Augmented)", marker='o', linestyle='-')
plt.plot(epochs, augmented_valid_losses, label="Validation Loss (Augmented)", marker='o', linestyle='-')

plt.title("Training and Validation Loss: Original vs Augmented")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
