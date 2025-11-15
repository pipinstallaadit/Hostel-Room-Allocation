import numpy as np
import os

# Load correct files
room_preferences_data = np.genfromtxt(
    '/Users/aadit/Desktop/OT MINI PROJECT/Hostel-Room-Allocation/csv Files/Room_Preference_Data.csv',
    dtype=float
)

room_capacity = np.genfromtxt(
    '/Users/aadit/Desktop/OT MINI PROJECT/Hostel-Room-Allocation/csv Files/Room_Capacity_List.csv',
    dtype=float
).astype(int)

total_capacity = np.sum(room_capacity)

room_id = []
double_occupancy_rooms = []

k = 0

for i in range(room_capacity.shape[0]):
    for j in range(room_capacity[i]):
        room_id.append(i)

    if room_capacity[i] == 2:
        double_occupancy_rooms.append([k, k+1])
        k += 1

    k += 1

room_preference = np.zeros((room_preferences_data.shape[0], total_capacity))

for i in range(room_preferences_data.shape[0]):
    room_preference[i] = room_preferences_data[i, room_id]

np.save(os.path.join('npy Files', 'Room_Preference_Matrix.npy'), room_preference)
np.save(os.path.join('npy Files', 'Double_Occupancy_Rooms.npy'), double_occupancy_rooms)

