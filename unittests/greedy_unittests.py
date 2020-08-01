from source.set_cover import SetCover
from source.set_cover import Solution
from source.greedy import GreedyAlgorithm
import unittest
import numpy as np


class TestGreedy(unittest.TestCase):

    def test_greedy_cost(self):
        instance = np.zeros((5, 4))
        '''
        test instance:
        1 0 0 1
        1 1 1 0
        0 0 0 1
        0 1 1 0
        0 1 0 0
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        instance[1][2] = 1
        instance[2][3] = 1
        instance[3][1] = 1
        instance[3][2] = 1
        instance[4][1] = 1
        set_cover = SetCover(instance)
        greedy_algo = GreedyAlgorithm(set_cover)
        solution = greedy_algo.get_greedy_solution()
        assert solution.cost == 2

    def test_greedy_cost_partial_sol(self):
        instance = np.zeros((5, 4))
        '''
        test instance:
        1 0 0 1
        1 1 1 0
        0 0 0 1
        0 1 1 0
        0 1 0 0
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        instance[1][2] = 1
        instance[2][3] = 1
        instance[3][1] = 1
        instance[3][2] = 1
        instance[4][1] = 1
        set_cover = SetCover(instance)
        greedy_algo = GreedyAlgorithm(set_cover)
        sol = np.zeros(5)
        my_solution = Solution(set_cover, sol)
        greedy_algo.make_solution_feasible(my_solution)
        assert my_solution.cost == 2
        sol = np.zeros(5)
        sol[3] = 1
        sol[4] = 1
        my_solution = Solution(set_cover, sol)
        greedy_algo.make_solution_feasible(my_solution)
        assert my_solution.cost == 3

    def test_count_dict_is_sorted(self):
        instance = np.zeros((3, 4))
        '''
        test instance:
        1 0 0 1
        1 1 1 0
        0 0 0 1
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        instance[1][2] = 1
        instance[2][3] = 1
        set_cover = SetCover(instance)
        greedy_algo = GreedyAlgorithm(set_cover)
        best_set = greedy_algo.get_best_next_set(set_cover.problem_instance, np.zeros(4))
        assert best_set == 1


if "__main__" == __name__:
    unittest.main()
