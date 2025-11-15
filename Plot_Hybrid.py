import numpy as np
import matplotlib.pyplot as plt

# Load the fitness history from Hybrid v1
history = np.load("npy Files/Hybrid_History.npy")   
plt.figure(figsize=(10, 5))

# If your fitness values are negative, this shows them naturally
plt.plot(history, linewidth=2, label="Hybrid PSO-TLBO v1")

plt.title("Hybrid PSO-TLBO v1 Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (Lower is Better)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("Hybrid_v1_Convergence.png", dpi=300)

plt.show()
