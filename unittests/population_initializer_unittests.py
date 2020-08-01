from source.population_initializer import PopulationCreator
from source.set_cover import SetCover
import unittest
import numpy as np


class TestPopulationInitializer(unittest.TestCase):

    def test_random_creation(self):
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
        pop_creator = PopulationCreator(set_cover)
        random_sol = pop_creator.create_random_instance()
        assert random_sol.is_feasible

    def test_random_creation_cro(self):
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
        pop_creator = PopulationCreator(set_cover)
        random_sol = pop_creator.reverse_cumlative_scheme()
        assert random_sol.is_feasible


if "__main__" == __name__:
    unittest.main()
