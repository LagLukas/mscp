try:
    from source.population_initializer import PopulationCreator
    from source.greedy import GreedyAlgorithm
    from source.set_cover import *
except Exception as _:
    from population_initializer import PopulationCreator
    from greedy import GreedyAlgorithm
    from set_cover import *
import numpy as np
import copy
import random
import sys
import time

class GCAIS:

    BORDER = 200
    # to tackle exploding population
    CROPPING = False

    def __init__(self, problem_instance, iterations):
        self.name = "GCAIS"
        self.population = {}
        self.problem_instance = problem_instance
        self.iterations = iterations
        number_of_sets = self.problem_instance.problem_instance.shape[0]
        sol_vector = np.zeros(number_of_sets)
        self.population[0] = {}
        self.population[0]["sols"] = [Solution(self.problem_instance, sol_vector)]
        self.population[0]["best_covered"] = 0
        self.iter = 0

    def superior(self, sol_a, sol_b):
        if sol_a.cost <= sol_b.cost:
            if sol_a.covered > sol_b.covered:
                return True
        if sol_a.covered >= sol_b.covered:
            if sol_a.cost < sol_b.cost:
                return True
        return False

    def dominated_by_any(self, sol_set, ele):
        for solution in sol_set:
            if self.superior(solution, ele):
                return True
        return False

    def mutate(self, sol):
        mut_prob = 1.0 / len(sol.set_vector)
        new_sol = copy.deepcopy(sol.set_vector)
        for i in range(0, len(sol.set_vector)):
            if random.random() < mut_prob:
                new_sol[i] = 1 if new_sol[i] == 0 else 0
        return Solution(self.problem_instance, new_sol)

    def iteration(self):
        keys = list(map(lambda x: int(x), self.population.keys()))
        # print(len(keys))
        for key in keys:
            # print("started key " + str(key) + " with " + str(len(self.population[key]["sols"])) + " elements")
            to_add = list(map(lambda x: self.mutate(x), self.population[key]["sols"]))
            for ele in to_add:
                try:
                    if self.population[ele.cost]["best_covered"] < ele.covered:
                        self.population[ele.cost]["best_covered"] = ele.covered
                        self.population[ele.cost]["sols"] = [ele]
                    elif self.population[ele.cost]["best_covered"] == ele.covered:
                        if len(self.population[ele.cost]["sols"]) < GCAIS.BORDER or not GCAIS.CROPPING:
                            equals = False
                            for other_ele in self.population[ele.cost]["sols"]:
                                if ele.equals_other_sol(other_ele):
                                    equals = True
                                    break
                            if not equals:
                                self.population[ele.cost]["sols"].append(ele)
                except Exception as _:
                        self.population[ele.cost] = {}
                        self.population[ele.cost]["best_covered"] = ele.covered
                        self.population[ele.cost]["sols"] = [ele]
        keys = list(map(lambda x: int(x), self.population.keys()))
        for key in keys:
            try:
                for i in range(key + 1, max(keys) + 1):
                    if self.population[i]["best_covered"] <= self.population[key]["best_covered"]:
                        del self.population[i]
            except Exception as _:
                pass
        self.iter = self.iter + 1

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start GCAIS")
        old_best = sys.maxsize
        found = 0
        s = time.time()
        for i in range(0, self.iterations):
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + " percent of GCAIS")
            iter_start = time.time()
            self.iteration()
            iter_end = time.time()
            feasible = []
            for key in self.population.keys():
                feasible.extend(list(filter(lambda x: x.is_feasible, self.population[key]["sols"])))
            try:
                best = min(feasible, key=lambda x: x.cost).cost
            except Exception as e:
                best = sys.maxsize
            if best != sys.maxsize:
                if best < old_best:
                    found = self.iter
                    old_best = best
                else:
                    if has_converged(found, self.iter, time.time() - s):
                        break
            self.logger.log_entry(self.iter, best, float(iter_end - iter_start))
        feasible = []
        for key in self.population.keys():
            feasible.extend(list(filter(lambda x: x.is_feasible, self.population[key]["sols"])))
        if len(feasible) == 0:
            all_sets = np.ones(self.problem_instance.problem_instance.shape[0])
            return Solution(self.problem_instance, all_sets)
        print("finished GCAIS")
        return min(feasible, key=lambda x: x.cost)
