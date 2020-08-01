from source.cro import CRO
from source.cro import Molecule
from source.set_cover import SetCover
from source.set_cover import Solution
import unittest
import numpy as np


class TestMolecule(unittest.TestCase):

    def test_on_wall_ineffective_collision(self):
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
        full = np.zeros(5)
        full[0] = 1
        full[4] = 1
        sol = Solution(set_cover, full)
        buffer = np.zeros(1)
        buffer[0] = 100
        mol = Molecule(set_cover, sol, buffer)
        mol.kinetic_energy = 100000
        mol.on_wall_ineffective_collision()
        assert buffer[0] > 100
        assert mol.kinetic_energy != 100000

    def test_synthesis(self):
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
        full = np.zeros(5)
        full[0] = 1
        full[4] = 1
        sol = Solution(set_cover, full)
        buffer = np.zeros(1)
        buffer[0] = 100
        mol1 = Molecule(set_cover, sol, buffer)
        full = np.ones(5)
        sol = Solution(set_cover, full)
        mol2 = Molecule(set_cover, sol, buffer)
        mol1.kinetic_energy = -2000
        mol2.kinetic_energy = -2000
        ret_val = mol1.synthesis(mol2)
        assert ret_val is None
        assert mol1.hits == 1
        assert mol2.hits == 1
        mol1.kinetic_energy = 2000
        mol2.kinetic_energy = 2000
        ret_val = mol1.synthesis(mol2)
        assert ret_val is not None

    def test_decomposition(self):
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
        full = np.zeros(5)
        full[0] = 1
        full[4] = 1
        sol = Solution(set_cover, full)
        buffer = np.zeros(1)
        mol = Molecule(set_cover, sol, buffer)
        mol.kinetic_energy = 0
        ret_val = mol.decomposition()
        assert ret_val is None
        assert mol.hits == 1
        mol.kinetic_energy = 10000
        ret_val = mol.decomposition()
        assert ret_val is not None
        assert buffer[0] == 0
        mol.kinetic_energy = -100
        buffer[0] = 100000
        ret_val = mol.decomposition()
        assert ret_val is not None
        assert buffer[0] < 100000

    def test_intermolecular_ineffective_collision(self):
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
        full = np.zeros(5)
        full[0] = 1
        full[4] = 1
        sol = Solution(set_cover, full)
        buffer = np.zeros(1)
        buffer[0] = 100
        mol1 = Molecule(set_cover, sol, buffer)
        full = np.zeros(5)
        full[1] = 1
        full[2] = 1
        sol = Solution(set_cover, full)
        mol2 = Molecule(set_cover, sol, buffer)
        mol1.kinetic_energy = 2000
        mol2.kinetic_energy = 2000
        mol1.intermolecular_ineffective_collision(mol2)
        assert mol1.kinetic_energy != 2000
        assert mol2.kinetic_energy != 2000


class TestCRO(unittest.TestCase):

    def test_infinte_loop(self):
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
        cro_algo = CRO(set_cover, 20, 0.2, 0, 500, 10, 1000)
        sol = cro_algo.find_approximation()
        assert sol.cost == 2


if "__main__" == __name__:
    unittest.main()
