import numpy as np
from Room_Allocation_Methods import calculate_fitness


class TLBO:
    def __init__(self, population_size=60, max_iter=300):
        self.population_size = population_size
        self.max_iter = max_iter

    def initialize_population(self, num_students):
        pop = []
        for _ in range(self.population_size):
            vec = []
            for i in range(num_students - 1):
                vec.append(np.random.randint(0, num_students - i - 1))
            pop.append(np.array(vec, dtype=int))
        return np.array(pop)

    def repair_vector(self, vec):
        """
        Ensures vector[i] ∈ [0, num_students - i - 1] for all i
        """
        n = len(vec)
        for i in range(n):
            max_val = n - i
            if vec[i] < 0:
                vec[i] = 0
            elif vec[i] >= max_val:
                vec[i] = max_val - 1
        return vec

    def teacher_phase(self, population, fitness):
        teacher = population[np.argmin(fitness)]
        mean_vec = np.mean(population, axis=0)

        new_population = []
        for learner in population:
            TF = np.random.randint(1, 3)  # TF = 1 or 2

            diff = np.round((teacher - TF * mean_vec)).astype(int)
            candidate = learner + diff

            candidate = self.repair_vector(candidate)
            new_population.append(candidate)

        return np.array(new_population)

    def learner_phase(self, population, fitness):
        new_population = []
        pop_size = len(population)

        for i, learner in enumerate(population):
            j = np.random.choice([k for k in range(pop_size) if k != i])
            partner = population[j]

            if fitness[j] < fitness[i]:
                candidate = learner + np.round(np.random.rand() * (partner - learner)).astype(int)
            else:
                candidate = learner - np.round(np.random.rand() * (partner - learner)).astype(int)

            candidate = self.repair_vector(candidate)
            new_population.append(candidate)

        return np.array(new_population)

    def run(self, room_pref_matrix=None):
        num_students = room_pref_matrix.shape[0]

        population = self.initialize_population(num_students)
        best_history = []

        for it in range(self.max_iter):
            fitness = np.array([calculate_fitness(ind) for ind in population])

            best_history.append(np.min(fitness))

            # Teacher Phase
            population = self.teacher_phase(population, fitness)

            # Recalculate fitness
            fitness = np.array([calculate_fitness(ind) for ind in population])

            # Learner Phase
            population = self.learner_phase(population, fitness)

            if it % 20 == 0:
                print(f"Iteration {it}: Best Score = {best_history[-1]}")

        # Final selection
        final_fitness = np.array([calculate_fitness(ind) for ind in population])
        best_index = np.argmin(final_fitness)

        return population[best_index], final_fitness[best_index], np.array(best_history)
