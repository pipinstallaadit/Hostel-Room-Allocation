import numpy as np
import random
from Room_Allocation_Methods import calculate_fitness

class HybridPSOTLBO:
    def __init__(
        self,
        population_size=50,
        max_iter=600,
        w=0.7,          # inertia
        c1=1.5,         # cognitive
        c2=1.5,         # social
        tlbo_interval=30,   # apply TLBO injection every X iterations
        tlbo_fraction=0.2,  # fraction of population to apply TLBO to
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.population_size = population_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tlbo_interval = tlbo_interval
        self.tlbo_fraction = tlbo_fraction

    # ---------- Helpers for vector domain ----------
    def init_particle(self, num_students):
        """Initialize a single valid vector for your encoding"""
        return np.array([np.random.randint(0, num_students - i - 1) for i in range(num_students - 1)], dtype=int)

    def repair(self, vec):
        """Ensure vec[i] in [0, num_students - i - 1]. vec may be float or int"""
        n = len(vec)
        # operate in-place for speed
        for i in range(n):
            maxv = n - i
            v = int(np.round(vec[i]))
            if v < 0:
                v = 0
            elif v >= maxv:
                v = maxv - 1
            vec[i] = v
        return vec.astype(int)

    def weighted_diff(self, a, b):
        """Smooth differential update: larger early, smaller later"""
        n = len(a)
        # weights from 1.0 down to 0.2
        w = np.linspace(1.0, 0.2, n)
        # result as float
        return w * (a.astype(float) - b.astype(float))

    # ---------- TLBO routines ----------
    def teacher_injection(self, population, fitness, apply_idx):
        """Apply a focused teacher-step to particles in apply_idx"""
        teacher = population[np.argmin(fitness)]
        mean = np.mean(population, axis=0)
        for idx in apply_idx:
            learner = population[idx].astype(float)
            TF = np.random.randint(1, 3)  # 1 or 2
            diff = self.weighted_diff(teacher, TF * mean)
            # small scaling to avoid big jumps; scale down by random factor
            alpha = np.random.uniform(0.4, 0.9)
            candidate = learner + alpha * diff
            population[idx] = self.repair(candidate)
        return population

    def learner_injection(self, population, fitness, apply_idx):
        """Apply learner pairwise updates among the selected indices"""
        for idx in apply_idx:
            i = idx
            # pick a partner
            choices = [k for k in range(len(population)) if k != i]
            j = random.choice(choices)
            learner = population[i].astype(float)
            partner = population[j].astype(float)
            if fitness[j] < fitness[i]:
                diff = self.weighted_diff(partner, learner)
                beta = np.random.uniform(0.3, 0.8)
                candidate = learner + beta * diff
            else:
                diff = self.weighted_diff(learner, partner)
                beta = np.random.uniform(0.3, 0.8)
                candidate = learner - beta * diff
            population[i] = self.repair(candidate)
        return population

    # ---------- Main run ----------
    def run(self, room_pref_matrix, verbose=True):
        """
        room_pref_matrix is required only to derive num_students shape.
        Fitness must be computed using calculate_fitness(vector) from Room_Allocation_Methods.
        """
        num_students = room_pref_matrix.shape[0]
        dim = num_students - 1

        # Initialize population and velocities
        population = np.array([self.init_particle(num_students) for _ in range(self.population_size)])
        velocities = np.zeros((self.population_size, dim), dtype=float)

        # personal bests
        pbest = population.copy()
        pbest_fitness = np.array([calculate_fitness(ind) for ind in pbest])

        # global best
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].copy()
        gbest_f = pbest_fitness[gbest_idx]

        history = [gbest_f]

        # precompute indices for TLBO injection size
        tlbo_count = max(1, int(self.tlbo_fraction * self.population_size))

        for it in range(1, self.max_iter + 1):
            # PSO velocity & position update
            for i in range(self.population_size):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                cognitive = self.c1 * r1 * (pbest[i].astype(float) - population[i].astype(float))
                social = self.c2 * r2 * (gbest.astype(float) - population[i].astype(float))

                velocities[i] = self.w * velocities[i] + cognitive + social

                # Clip velocities to reasonable bounds to avoid huge jumps.
                # Velocity bounds proportional to remaining range per position
                max_vel = np.array([max(1, (dim - j) * 0.5) for j in range(dim)], dtype=float)
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)

                candidate = population[i].astype(float) + velocities[i]
                # repair candidate to valid integer vector
                population[i] = self.repair(candidate)

            # Evaluate fitness
            fitness = np.array([calculate_fitness(ind) for ind in population])

            # Update personal and global bests
            improved = fitness < pbest_fitness
            pbest[improved] = population[improved]
            pbest_fitness[improved] = fitness[improved]

            cur_best_idx = np.argmin(pbest_fitness)
            cur_best_f = pbest_fitness[cur_best_idx]
            if cur_best_f < gbest_f:
                gbest_f = cur_best_f
                gbest = pbest[cur_best_idx].copy()

            history.append(gbest_f)

            # TLBO injection occasionally
            if (it % self.tlbo_interval) == 0:
                # choose worst-performing ones to replace/explore, plus a few randoms
                worst_idxs = np.argsort(fitness)[-tlbo_count:]
                random_idxs = np.random.choice(self.population_size, tlbo_count, replace=False)
                apply_idxs = list(set(list(worst_idxs) + list(random_idxs)))[:tlbo_count]

                # Teacher phase on selected
                population = self.teacher_injection(population, fitness, apply_idxs)

                # Re-evaluate fitness for injected particles
                for idx in apply_idxs:
                    fitness[idx] = calculate_fitness(population[idx])

                # Learner phase among the same set
                population = self.learner_injection(population, fitness, apply_idxs)

                # re-evaluate and update pbest accordingly
                for idx in apply_idxs:
                    f = calculate_fitness(population[idx])
                    fitness[idx] = f
                    if f < pbest_fitness[idx]:
                        pbest[idx] = population[idx].copy()
                        pbest_fitness[idx] = f

                # update global best if needed
                cur_best_idx = np.argmin(pbest_fitness)
                cur_best_f = pbest_fitness[cur_best_idx]
                if cur_best_f < gbest_f:
                    gbest_f = cur_best_f
                    gbest = pbest[cur_best_idx].copy()

            if verbose and (it % max(1, int(self.max_iter/20)) == 0):
                print(f"Iter {it}/{self.max_iter}  Global best = {gbest_f}")

        return gbest.astype(int), gbest_f, np.array(history)
