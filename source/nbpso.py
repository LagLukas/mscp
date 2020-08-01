import math
import copy
import random
import numpy as np
from numpy import linalg as LA
try:
    from source.set_cover import *
    from source.greedy import GreedyAlgorithm
except Exception as _:
    from set_cover import *
    from greedy import GreedyAlgorithm


class Particle:

    def __init__(self, initial_sol, local_opt, velocity, problem_instance):
        self.greedy = GreedyAlgorithm(problem_instance)
        self.greedy.make_solution_feasible(initial_sol)
        self.solution = initial_sol
        self.personal_best = copy.deepcopy(initial_sol)
        self.local_opt = copy.deepcopy(local_opt)
        self.problem_instance = problem_instance
        self.velocity_vector = velocity

    def flip_propability(self, velocity):
        sigmoid = 1.0 / (1.0 + math.exp(-1.0 * velocity))
        return 2.0 * abs(sigmoid - 0.5)

    def update_position(self):
        for i in range(0, len(self.velocity_vector)):
            prop = self.flip_propability(self.velocity_vector[i])
            if random.random() < prop:
                self.solution.set_vector[i] = 0 if self.solution.set_vector[i] == 1 else 1

    def update_feasibility(self):
        self.solution.is_feasible = None
        self.solution.is_feasible_solution()
        self.greedy.make_solution_feasible(self.solution)
        if self.solution.cost < self.personal_best.cost:
            old = self.personal_best
            del old
            self.personal_best = copy.deepcopy(self.solution)

    def update_velocity(self, omega):
        self.velocity_vector = omega * self.velocity_vector
        self.velocity_vector += NBPSO.PHI_1 * random.random() * (self.local_opt.set_vector - self.solution.set_vector)
        self.velocity_vector += NBPSO.PHI_2 * random.random() * (self.personal_best.set_vector - self.solution.set_vector)
        for i in range(0, len(self.velocity_vector)):
            if self.velocity_vector[i] > NBPSO.V_MAX:
                self.velocity_vector[i] = NBPSO.V_MAX
            elif self.velocity_vector[i] < -1.0 * NBPSO.V_MAX:
                self.velocity_vector[i] = -1.0 * NBPSO.V_MAX


class NBPSO:

    V_MAX = 6
    PHI_1 = 2
    PHI_2 = 2
    NEIGHBOURHOOD = 5
    OMEGA = 0.6
    OMEGA_LOWER = 0.2

    def __init__(self, pop_size, problem_instance, iterations):
        self.name = "NBPSO"
        self.omega = NBPSO.OMEGA
        self.pop_size = pop_size
        self.problem_instance = problem_instance
        self.number_of_sets = problem_instance.problem_instance.shape[0]
        self.iterations = iterations
        self.init_population()
        self.update_local_opts()

    def init_population(self):
        self.population = []
        for i in range(0, self.pop_size):
            sol_vector = np.zeros(self.number_of_sets)
            if i != self.pop_size - 1:
                for i in range(0, self.number_of_sets):
                    if random.random() < 0.5:
                        sol_vector[i] = 1
            solution = Solution(self.problem_instance, sol_vector)
            velocity_vector = np.zeros(self.number_of_sets)
            for i in range(0, self.number_of_sets):
                magnitude = 2 * (random.random() - 0.5) * NBPSO.V_MAX
                velocity_vector[i] = magnitude
            self.population.append(Particle(solution, solution, velocity_vector, self.problem_instance))

    def update_local_opts(self):
        for particle in self.population:
            distances = list(map(lambda x: (x, LA.norm(particle.personal_best.set_vector - x.personal_best.set_vector, 1)), self.population))
            distances = sorted(distances, key=lambda x: x[1])
            neighbourhood = list(map(lambda x: x[0], distances[:NBPSO.NEIGHBOURHOOD + 1]))
            old = particle.local_opt
            del old
            particle.local_opt = copy.deepcopy(min(neighbourhood, key=lambda x: x.personal_best.cost).personal_best)

    def find_approximation(self):
        print("start NBPSO")
        for i in range(0, self.iterations):
            if i % 10 == 0:
                print_now()
                print("finished " + str(i/self.iterations) + " of nbpso")
            for particle in self.population:
                particle.update_velocity(self.omega)
                particle.update_position()
                particle.update_feasibility()
            self.update_local_opts()
            self.omega = NBPSO.OMEGA - ((NBPSO.OMEGA - NBPSO.OMEGA_LOWER) / self.iterations) * i
        print("finished NBPSO")
        return min(self.population, key=lambda x: x.personal_best.cost).personal_best

