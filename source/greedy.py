try:
    from source.set_cover import SetCover
    from source.set_cover import Solution
except Exception as _:
    from set_cover import SetCover
    from set_cover import Solution
from abc import abstractmethod
import numpy as np


class FeasibleSolutionConstructor:

    @abstractmethod
    def make_solution_feasible(self, solution):
        pass


class GreedyAlgorithm(FeasibleSolutionConstructor):
    '''
    Greedy algorithm that always takes the set next which covers the most, extant elements.
    '''

    def __init__(self, set_cover_instance):
        self.name = "GREEDY"
        self.set_cover_instance = set_cover_instance

    def get_best_next_set(self, table, already_covered):
        '''
        retrieves the index of set that has the most uncovered elements.

        :param table: set cover instance table (rows = sets, columns = elements).
        :param already_covered: vector containing the already covered elements. Indexes
        correspond to the column indexes.

        :return : index
        '''
        max_index = -1
        max_val = -1
        for i in range(0, table.shape[0]):
            element_count = sum([1 if table[i][j] > already_covered[j] else 0 for j in range(0, len(already_covered))])
            if element_count > max_val:
                max_val = element_count
                max_index = i
        return max_index

    def greedy_iteration(self, solution, table):
        biggest_set_index = self.get_best_next_set(table, solution.covered_elements)
        solution.add_set(biggest_set_index)
        return solution.is_feasible

    def make_solution_feasible(self, solution):
        '''
        Applies the greedy algorithm on a infeasible solution to make feasible.
        '''
        while not solution.is_feasible:
            self.greedy_iteration(solution, self.set_cover_instance.problem_instance)

    def get_greedy_solution(self):
        empty_vec = np.zeros(self.set_cover_instance.problem_instance.shape[0])
        approx_sol = Solution(self.set_cover_instance, empty_vec)
        self.make_solution_feasible(approx_sol)
        return approx_sol
