# Hostel Room Allocation using Metaheuristic Optimization

This repository contains a Python implementation for solving the hostel room allocation problem using various metaheuristic optimization algorithms. The goal is to assign students to hostel rooms in a way that maximizes their overall satisfaction, based on their stated room preferences. The problem is modeled as a single-objective optimization task, where the fitness function aims to minimize the sum of preference scores for all students.

---

## Algorithms Implemented

This project explores and compares three different optimization techniques:

### **🟦 Particle Swarm Optimization (PSO)**

A population-based stochastic optimization technique inspired by the social behavior of bird flocking or fish schooling.
Implemented in:

* `Particle_Swarm_Optimization.py`
* `PSO_Room_Allocation.py`

---

### **🟩 Teaching-Learning-Based Optimization (TLBO)**

An algorithm that simulates the teaching and learning process in a classroom using:

* **Teacher Phase:** Students learn from the best performer
* **Learner Phase:** Students learn from each other

Implemented in:

* `TLBO_Room_Allocation.py`
* `TLBO_Room_Allocation_Run.py`

---

### **🟧 Hybrid PSO-TLBO (Best Performer)**

A novel hybrid approach combining:

* PSO’s exploitation (fine search)
* TLBO’s exploration (global jumps)

The Hybrid periodically injects TLBO-inspired operations into the PSO loop, allowing the optimizer to escape local minima while still converging smoothly.

Implemented in:

* `Hybrid_PSO_TLBO.py`
* `Hybrid_PSO_TLBO_Run.py`

---

## How it Works

The core of the solution involves:

1. **Data Modeling**
   Reads student preferences and room capacities from CSV files and generates:

   * `Room_Preference_Matrix.npy`
   * `Double_Occupancy_Rooms.npy`

2. **Solution Encoding**
   An allocation is represented by a special vector of pop-indices, decoded into a valid student-to-room assignment using `Room_Allocation_Methods.py`.

3. **Fitness Evaluation**
   The fitness = **negative** sum of preference scores.
   Lower (more negative) → better.

4. **Optimization**
   PSO, TLBO, or Hybrid PSO+TLBO iteratively searches for the vector that minimizes dissatisfaction.

---

## Repository Structure

```
.
├── Hybrid_PSO_TLBO.py                # Hybrid algorithm
├── Hybrid_PSO_TLBO_Run.py            # Hybrid runner
├── Particle_Swarm_Optimization.py     # PSO class
├── PSO_Room_Allocation.py             # PSO runner
├── TLBO_Room_Allocation.py            # TLBO class
├── TLBO_Room_Allocation_Run.py        # TLBO runner
├── Prepare_Room_Allocation_Data.py    # Converts CSV → NPY
├── Room_Allocation_Methods.py         # Fitness + allocation logic
├── Room_Allocation_Results.py         # Displays allocation summary
├── Room_Allocation_PSO_Visualization.py # PSO convergence animation
├── Plot_Hybrid.py                     # Hybrid convergence plot
├── csv Files/
│   ├── Room_Capacity_List.csv
│   └── Room_Preference_Data.csv
└── npy Files/
    ├── Room_Preference_Matrix.npy
    └── Double_Occupancy_Rooms.npy
```

---

## Setup and Usage

### **Prerequisites**

Install required libraries:

```bash
pip install numpy matplotlib numba
```

`ffmpeg` is required for saving PSO animations.

---

## Running the Code

### **1. Prepare the Data**

```bash
python Prepare_Room_Allocation_Data.py
```

⚠️ Make sure to update file paths inside this file to match your machine.

---

### **2. Run an Optimization Algorithm**

#### **PSO**

```bash
python PSO_Room_Allocation.py
```

#### **TLBO**

```bash
python TLBO_Room_Allocation_Run.py
```

#### **Hybrid PSO + TLBO**

```bash
python Hybrid_PSO_TLBO_Run.py
```

---

### **3. View the Results**

#### **PSO Allocation Results**

```bash
python Room_Allocation_Results.py
```

#### **PSO Convergence Animation**

```bash
python Room_Allocation_PSO_Visualization.py
```

#### **Hybrid Convergence Plot**

```bash
python Plot_Hybrid.py
```

---

# 📊 Results and Comparison

This project evaluates three algorithms based on the **best preference sum achieved** (lower = better).

### **PSO Results**

* Best score ≈ **–650**
* Converges quickly but traps in local minima
* Limited exploration

### **TLBO Results**

* Best score ≈ **–1400 to –1500**
* Strong exploration
* High variance, sometimes unstable

### **Hybrid PSO-TLBO Results**

* Best score ≈ **–1882**
* Consistently outperforms both PSO and TLBO
* Escapes local minima while maintaining PSO’s stability
* Achieves ~**3× improvement** over PSO

### **📈 Summary Table**

| Algorithm           | Strengths                             | Weaknesses                 | Best Score  |
| ------------------- | ------------------------------------- | -------------------------- | ----------- |
| **PSO**             | Fast, stable                          | Gets stuck in local minima | ~ **–650**  |
| **TLBO**            | Explores well                         | Less stable, large jumps   | ~ **–1500** |
| **Hybrid PSO+TLBO** | Best balance of global + local search | Slightly heavier compute   | ⭐ **–1882** |

**Conclusion:**
💡 The **Hybrid PSO-TLBO** algorithm produces the **best hostel room allocation quality**.

---

# ✍️ Authors & Contributors

### **Original Project Author**

**Souritra Garai**

* Email: [sgarai65@gmail.com](mailto:sgarai65@gmail.com)
* Affiliation: IIT Gandhinagar

### **Extended Optimization & Improvements**

* Added TLBO algorithm
* Designed Hybrid PSO + TLBO
* Implemented convergence plotting
* Enhanced documentation and readability
* Performed performance tuning & benchmarking

---

# 🙌 Acknowledgements

Special thanks to:

* Metaheuristic optimization research community
* Open-source contributors
* Everyone supporting educational and research-oriented software development

