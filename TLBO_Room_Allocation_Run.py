import numpy as np
from TLBO_Room_Allocation import TLBO

room_pref_matrix = np.load("npy Files/Room_Preference_Matrix.npy")

tlbo = TLBO(population_size=80, max_iter=400)

best_vec, best_score, history = tlbo.run(room_pref_matrix)

np.save("TLBO_Final_Vector.npy", best_vec)
np.save("TLBO_Final_Score.npy", np.array([best_score]))

print("\nBest TLBO Score:", best_score)
