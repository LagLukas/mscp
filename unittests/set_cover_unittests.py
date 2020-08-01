from source.set_cover import SetCover
from source.set_cover import Solution
import unittest
import numpy as np


class TestSetCover(unittest.TestCase):

    def test_add_set(self):
        instance = np.zeros((3, 4))
        '''
        test instance:
        1 0 0 1
        1 1 0 0
        0 0 1 1
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        instance[2][2] = 1
        instance[2][3] = 1
        set_cover = SetCover(instance)
        sol = np.zeros(3)
        sol[1] = 1
        my_solution = Solution(set_cover, sol)
        assert my_solution.is_feasible == False
        my_solution.add_set(2)
        assert my_solution.is_feasible == True

    def test_is_feasible(self):
        instance = np.zeros((3, 4))
        '''
        test instance:
        1 0 0 1
        1 1 0 0
        0 0 1 1
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        instance[2][2] = 1
        instance[2][3] = 1
        set_cover = SetCover(instance)
        sol = np.zeros(3)
        sol[1] = 1
        sol[2] = 1
        my_solution = Solution(set_cover, sol)
        assert my_solution.is_feasible_solution() == True

    def test_is_not_feasible(self):
        instance = np.zeros((3, 4))
        '''
        test instance:
        1 0 0 1
        1 1 0 0
        0 0 1 1
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        instance[2][2] = 1
        instance[2][3] = 1
        set_cover = SetCover(instance)
        sol = np.zeros(3)
        sol[1] = 1
        my_solution = Solution(set_cover, sol)
        assert my_solution.is_feasible_solution() == False

    def test_problem_is_solveable(self):
        instance = np.zeros((3, 4))
        '''
        test instance:
        1 0 0 1
        1 1 0 0
        0 0 1 1
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        instance[2][2] = 1
        instance[2][3] = 1
        in_exception = False
        try:
            set_cover = SetCover(instance)
        except Exception as _:
            in_exception = True
        assert in_exception == False

    def test_problem_is_unsolveable(self):
        instance = np.zeros((3, 4))
        '''
        test instance:
        1 0 0 1
        1 1 0 0
        0 0 1 1
        '''
        instance[0][0] = 1
        instance[0][3] = 1
        instance[1][1] = 1
        instance[1][0] = 1
        in_exception = False
        try:
            set_cover = SetCover(instance)
        except Exception as _:
            in_exception = True
        assert in_exception == True


if "__main__" == __name__:
    unittest.main()
