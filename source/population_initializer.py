try:
    from source.set_cover import SetCover
    from source.set_cover import Solution
    from source.greedy import GreedyAlgorithm
except Exception as _:
    from set_cover import SetCover
    from set_cover import Solution
    from greedy import GreedyAlgorithm

import numpy as np
import random


class PopulationCreator:

    ON_PROB = 0.5

    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.greedy = GreedyAlgorithm(problem_instance)

    def create_random_instance(self):
        number_of_sets = self.problem.problem_instance.shape[0]
        sol_vector = np.zeros(number_of_sets)
        for i in range(0, number_of_sets):
            if random.random() < PopulationCreator.ON_PROB:
                sol_vector[i] = 1
        solution = Solution(self.problem, sol_vector)
        self.greedy.make_solution_feasible(solution)
        return solution

    def create_population(self, size):
        population = []
        for _ in range(0, size):
            solution = self.create_random_instance()
            population.append(solution)
        return population

    def create_random_instance_beasley(self):
        number_of_elements = self.problem.problem_instance.shape[1]
        number_of_sets = self.problem.problem_instance.shape[0]
        sol_vector = np.zeros(number_of_sets)
        for i in range(0, number_of_elements):
            solution = Solution(self.problem, sol_vector)
            if solution.covered_elements[i] == 0:
                possible_sets = []
                for j in range(0, number_of_sets):
                    if self.problem.problem_instance[j][i] == 1:
                        possible_sets.append(j)
                index = random.randint(0, len(possible_sets) - 1)
                sol_vector[index] = 1
        # short check for redundant sets
        sets_to_check = list(range(0, number_of_sets))
        while (len(sets_to_check) > 0):
            index = random.randint(0, len(sets_to_check) - 1)
            del sets_to_check[index]
            sol_vector[index] = 0
            if not Solution(self.problem, sol_vector).is_feasible:
                sol_vector[index] = 1
        return Solution(self.problem, sol_vector)

    def create_population_beasley(self, size):
        population = []
        for _ in range(0, size):
            solution = self.create_random_instance_beasley()
            population.append(solution)
        return population

    def reverse_cumlative_scheme(self):
        set_index = []
        for element in range(0, self.problem.problem_instance.shape[1]):
            element_covers = []
            for set_id in range(0, self.problem.problem_instance.shape[0]):
                if self.problem.problem_instance[set_id][element] == 1:
                    element_covers.append(set_id)
            index = random.randint(0, len(element_covers) - 1)
            set_index.append(element_covers[index])
        sol_vector = np.zeros(self.problem.problem_instance.shape[0])
        for element in set_index:
            sol_vector[element] = 1
        solution = Solution(self.problem, sol_vector)
        return solution

    def create_population_cro(self, size):
        population = []
        for _ in range(0, size):
            solution = self.reverse_cumlative_scheme()
            population.append(solution)
        return population

