from source.seip import SEIP
from source.set_cover import SetCover
import unittest
import numpy as np


class TestSEIP(unittest.TestCase):

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
        cro_algo = SEIP(set_cover, 1000)
        sol = cro_algo.find_approximation()
        assert sol.cost == 2


if "__main__" == __name__:
    unittest.main()
