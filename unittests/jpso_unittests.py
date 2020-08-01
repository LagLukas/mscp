from source.jpso import JPSO
from source.jpso import Particle
from source.set_cover import SetCover
from source.set_cover import Solution
from source.population_initializer import PopulationCreator
import unittest
import numpy as np
import copy


class TestJPSO(unittest.TestCase):

    def test_population_size(self):
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
        nbpso = JPSO(50, set_cover, 6)
        assert len(nbpso.population) == 50

    def test_local_optima_update(self):
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
        nbpso = JPSO(6, set_cover, 50)
        other_pop = []
        sol_a = Solution(set_cover, np.ones(5))
        other_pop.append(Particle(sol_a, sol_a, np.ones(5), set_cover, PopulationCreator(set_cover)))
        set_vector = np.zeros(5)
        set_vector[4] = 1
        sol_a = Solution(set_cover, set_vector)
        other_pop.append(Particle(sol_a, sol_a, np.ones(5), set_cover, PopulationCreator(set_cover)))
        set_vector[3] = 1
        sol_a = Solution(set_cover, set_vector)
        other_pop.append(Particle(sol_a, sol_a, np.ones(5), set_cover, PopulationCreator(set_cover)))
        set_vector[2] = 1
        sol_a = Solution(set_cover, set_vector)
        other_pop.append(Particle(sol_a, sol_a, np.ones(5), set_cover, PopulationCreator(set_cover)))
        set_vector[4] = 0
        sol_a = Solution(set_cover, set_vector)
        other_pop.append(Particle(sol_a, sol_a, np.ones(5), set_cover, PopulationCreator(set_cover)))
        optimal = np.zeros(5)
        optimal[0] = 1
        optimal[1] = 1
        sol_a = Solution(set_cover, optimal)
        other_pop.append(Particle(sol_a, sol_a, np.ones(5), set_cover, PopulationCreator(set_cover)))
        nbpso.population = other_pop
        nbpso.update_local_opts()
        for particle in nbpso.population:
            assert particle.local_opt.cost == 2

    def test_nbpso_approx(self):
        instance = np.zeros((5, 4))
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
        jpso = JPSO(10, set_cover, 20)
        solution = jpso.find_approximation()
        assert solution.is_feasible == True
        assert solution.cost == 2


if "__main__" == __name__:
    unittest.main()
