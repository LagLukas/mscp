from source.simulated_annealing import SimulatedAnnealing
from source.set_cover import SetCover
from source.set_cover import Solution
from source.greedy import GreedyAlgorithm
import unittest
import numpy as np


class TestSimulatedAnnealing(unittest.TestCase):

    def test_neighbour_func(self):
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
        sa = SimulatedAnnealing(set_cover, solution)
        neighbour = sa.get_random_neighbour()
        assert neighbour.is_feasible

    def test_boltzmann(self):
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
        use_all = np.zeros(5)
        use_all[0] = 1
        use_all[1] = 1
        solution = Solution(set_cover, use_all)
        sa = SimulatedAnnealing(set_cover, solution)
        neighbour = sa.get_random_neighbour()
        prob = sa.boltzmann_propability(neighbour, 2)
        assert 0 <= prob and prob <= 1

    def test_endless_loop(self):
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
        use_all = np.ones(5)
        solution = Solution(set_cover, use_all)
        sa = SimulatedAnnealing(set_cover, 1000, solution)
        sa.find_approximation()


if "__main__" == __name__:
    unittest.main()
