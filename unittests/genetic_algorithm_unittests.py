from source.genetic_algorithm import GeneticAlgorithm
from source.population_initializer import PopulationCreator
from source.set_cover import SetCover
import unittest
import numpy as np
import copy


class TestGeneticAlgorithm(unittest.TestCase):

    def test_selection(self):
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
        ga = GeneticAlgorithm(set_cover, 50, 5, 0.05, 100)
        parent = ga.selection()
        assert parent.is_feasible == True

    def test_crossover(self):
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
        ga = GeneticAlgorithm(set_cover, 50, 1, 0.05, 100)
        parent_a = ga.selection()
        parent_b = ga.selection()
        child_a = copy.deepcopy(parent_a)
        child_b = copy.deepcopy(parent_b)
        ga.crossover(child_a, child_b)
        for i in range(0, len(child_a.set_vector)):
            assert child_a.set_vector[i] == parent_a.set_vector[i] or child_a.set_vector[i] == parent_b.set_vector[i]
            assert child_b.set_vector[i] == parent_a.set_vector[i] or child_b.set_vector[i] == parent_b.set_vector[i]

    def test_mutation(self):
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
        ga = GeneticAlgorithm(set_cover, 50, 1, 0.05, 100)
        parent_a = ga.selection()
        parent_b = ga.selection()
        child_a = copy.deepcopy(parent_a)
        child_b = copy.deepcopy(parent_b)
        ga.crossover(child_a, child_b)
        ga.mutation(child_a)
        for i in range(0, len(child_a.set_vector)):
            assert child_a.set_vector[i] == 1 or child_a.set_vector[i] == 0

    def test_deletion(self):
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
        ga = GeneticAlgorithm(set_cover, 50, 1, 0.05, 100)
        pop_creator = PopulationCreator(set_cover)
        population = pop_creator.create_population(20)
        ga.population.extend(population)
        ga.del_from_pop()
        assert len(ga.population) == ga.pop_size

    def test_ga_approx(self):
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
        ga = GeneticAlgorithm(set_cover, 50, 1, 0.05, 100)
        solution = ga.find_approximation()
        assert solution.is_feasible == True


if "__main__" == __name__:
    unittest.main()
