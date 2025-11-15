import numpy as np
from Hybrid_PSO_TLBO import HybridPSOTLBO

# load the same data files your other scripts use
room_pref_matrix = np.load("npy Files/Room_Preference_Matrix.npy")
# double occupancy not needed here; fitness function uses allocate_students internally

hybrid = HybridPSOTLBO(
    population_size=60,
    max_iter=800,
    w=0.72,
    c1=1.4,
    c2=1.6,
    tlbo_interval=25,
    tlbo_fraction=0.20,
    seed=42
)

best_vec, best_score, history = hybrid.run(room_pref_matrix, verbose=True)

np.save("Hybrid_Final_Vector.npy", best_vec)
np.save("Hybrid_Final_Score.npy", np.array([best_score]))
np.save("Hybrid_History.npy", history)

print("\nHybrid run finished.")
print("Best hybrid score:", best_score)
