import math
import copy
import random
import numpy as np
import time
from numpy import linalg as LA
try:
    from source.set_cover import *
    from source.greedy import GreedyAlgorithm
    from source.population_initializer import PopulationCreator
except Exception as _:
    from set_cover import *
    from greedy import GreedyAlgorithm
    from population_initializer import PopulationCreator

class JPSO_Particle:

    def __init__(self, initial_sol, local_opt, global_opt, problem_instance, pop_creator):
        self.greedy = GreedyAlgorithm(problem_instance)
        self.greedy.make_solution_feasible(initial_sol)
        self.solution = initial_sol
        self.personal_best = copy.deepcopy(initial_sol)
        self.local_opt = copy.deepcopy(local_opt)
        self.global_opt = global_opt
        self.problem_instance = problem_instance
        self.pop_creator = pop_creator

    def jump(self, attractor):
        used_sets = []
        used_sets_attractor = []
        for i in range(0, len(self.solution.set_vector)):
            if self.solution.set_vector[i] == 1:
                used_sets.append(i)
            if attractor.set_vector[i] == 1:
                used_sets_attractor.append(i)
        r = random.randint(1, len(used_sets))
        set_vector = copy.deepcopy(self.solution.set_vector)
        for i in range(0, r):
            if random.uniform(0, 1) < 0.5:
                # delete random set
                index = random.randint(0, len(used_sets) - 1)
                element = used_sets[index]
                set_vector[element] = 0
                del used_sets[index]
            else:
                # add random set of attractor
                index = random.randint(0, len(used_sets_attractor) - 1)
                element = used_sets_attractor[index]
                set_vector[element] = 1
                used_sets.append(index)
        new_solution = Solution(self.problem_instance, set_vector)
        set_vector = new_solution.set_vector
        self.greedy.make_solution_feasible(new_solution)
        # check for redundant sets
        for index in used_sets:
            set_vector[index] = 0
            if not Solution(self.problem_instance, set_vector).is_feasible:
                set_vector[index] = 1
        self.solution = Solution(self.problem_instance, set_vector)
        if self.personal_best.cost > self.solution.cost:
            self.personal_best = self.solution

    def update_position(self):
        alpha = random.uniform(0, 1)
        if alpha < 0.25:
            attractor = self.pop_creator.create_random_instance_beasley()
        elif alpha < 0.5:
            attractor = self.personal_best
        elif alpha < 0.75:
            attractor = self.local_opt
        else:
            attractor = self.global_opt
        self.jump(attractor)


class JPSO:

    NEIGHBOURHOOD = 5

    def __init__(self, pop_size, problem_instance, iterations):
        self.name = "JPSO"
        self.pop_size = pop_size
        self.problem_instance = problem_instance
        self.number_of_sets = problem_instance.problem_instance.shape[0]
        self.pop_creator = PopulationCreator(problem_instance)
        self.iterations = iterations
        self.init_population()
        self.update_local_opts()
        self.best = self.update_glob_opt()

    def init_population(self):
        self.population = []
        for _ in range(0, self.pop_size):
            solution = self.pop_creator.create_random_instance_beasley()
            self.population.append(JPSO_Particle(solution, solution, solution, self.problem_instance, self.pop_creator))

    def update_local_opts(self):
        for particle in self.population:
            distances = list(map(lambda x: (x, LA.norm(particle.personal_best.set_vector - x.personal_best.set_vector, 1)), self.population))
            distances = sorted(distances, key=lambda x: x[1])
            neighbourhood = list(map(lambda x: x[0], distances[:JPSO.NEIGHBOURHOOD + 1]))
            old = particle.local_opt
            del old
            particle.local_opt = copy.deepcopy(min(neighbourhood, key=lambda x: x.personal_best.cost).personal_best)

    def update_neighbourhood(self, particle):
        distances = list(map(lambda x: (x, LA.norm(particle.personal_best.set_vector - x.personal_best.set_vector, 1)), self.population))
        distances = sorted(distances, key=lambda x: x[1])
        neighbourhood = list(map(lambda x: x[0], distances[:JPSO.NEIGHBOURHOOD + 1]))
        old = particle.local_opt
        del old
        particle.local_opt = copy.deepcopy(min(neighbourhood, key=lambda x: x.personal_best.cost).personal_best)

    def update_glob_opt(self):
        best = min(self.population, key=lambda x: x.personal_best.cost).personal_best
        for particle in self.population:
            particle.global_opt = best
        return best

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start JPSO")
        s = time.time()
        old_best = sys.maxsize
        found = 0
        for i in range(0, self.iterations):
            iter_start = time.time()
            for particle in self.population:
                particle.update_position()
                self.update_neighbourhood(particle)
                if self.best.cost > particle.personal_best.cost:
                    self.update_glob_opt()
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + "percent of JPSO")
            iter_end = time.time()
            if self.best.cost != sys.maxsize:
                if self.best.cost < old_best:
                    found = i
                    old_best = self.best.cost
                else:
                    if has_converged(found, i, time.time() - s):
                        break
            self.logger.log_entry(i, self.best.cost, float(iter_end - iter_start))
        print("finished JPSO")
        return self.best

