import numpy as np
import random
import sys
import copy
import time
try:
    from source.set_cover import *
    from source.set_cover import Solution
    from source.greedy import GreedyAlgorithm
except Exception as _:
    from set_cover import *
    from set_cover import Solution
    from greedy import GreedyAlgorithm


class SEIP:

    def __init__(self, problem_instance, iterations):
        self.name = "SEIP"
        self.iterations = iterations
        self.problem = problem_instance
        self.greedy_alg = GreedyAlgorithm(problem_instance)
        self.population = []

    def mu(self, sol):
        return sum(sol.covered_elements)

    def superior(self, sol_a, sol_b):
        if sol_a.cost <= sol_b.cost:
            if self.mu(sol_a) == self.mu(sol_b):
                return True
        return False

    def mutate(self, sol):
        mut_prob = 1.0 / len(sol.set_vector)
        new_sol = copy.deepcopy(sol.set_vector)
        for i in range(0, len(sol.set_vector)):
            if random.random() < mut_prob:
                new_sol[i] = 1 if new_sol[i] == 0 else 0
        return Solution(self.problem, new_sol)

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start SEIP")
        s = time.time()
        old_best = sys.maxsize
        found = 0
        self.population.append(Solution(self.problem, np.zeros(self.problem.problem_instance.shape[0])))
        for i in range(0, self.iterations):
            iter_start = time.time()
            ele = self.population[random.randint(0, len(self.population) - 1)]
            mutated = self.mutate(ele)
            if len(list(filter(lambda x: self.superior(x, mutated), self.population))) == 0:
                dominated_sols = list(filter(lambda x: self.superior(mutated, x), self.population))
                self.population = [x for x in self.population if x not in dominated_sols]
                self.population.append(mutated)
            iter_end = time.time()
            feasible = list(filter(lambda x: x.is_feasible, self.population))
            try:
                best = min(feasible, key=lambda x: x.cost).cost
            except Exception as _:
                best = sys.maxsize
            if best != sys.maxsize:
                if best < old_best:
                    found = i
                    old_best = best
                else:
                    if has_converged(found, i, time.time() - s):
                        break
            self.logger.log_entry(i, best, float(iter_end - iter_start))
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + "percent of SEIP")
        for sol in self.population:
            self.greedy_alg.make_solution_feasible(sol)
        print("finished SEIP")
        feasible = list(filter(lambda x: x.is_feasible, self.population))
        try:
            return min(feasible, key=lambda x: x.cost)
        except Exception as _:
            return sys.maxsize
