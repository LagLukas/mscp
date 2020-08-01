import numpy as np
import sys
import random
import itertools
from datetime import datetime

def has_converged(found, current, duration):
    assert current - found > 0
    assert duration > 0
    if current - found > 2000:
        return True
    if duration > 3600:
        return True
    return False

def print_now():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

class SetCover:
    '''
    Represents an instance of the Minimum Set Cover problem
    '''
    def __init__(self, problem_instance):
        '''
        :param problem_instance: two dimensional numpy array. The rows represent the available
        sets and the columns the possible elements. If problem_instance[i][j] is one then i-th
        set has the j-th element. If it is set to 0 then the set does not have this element.
        '''
        self.problem_instance = problem_instance
        self.is_solveable()

    def is_solveable(self):
        '''
        Checks if there exists a set cover at all. Raises an Exception if not.
        '''
        all_sets = np.ones(self.problem_instance.shape[0])
        solution = Solution(self, all_sets)
        if not solution.is_feasible_solution():
            raise Exception("Set cover instance cannot be solved")


class TightSetCover():

    def __init__(self, min_k, max_k, max_n):
        self.n = max_n
        self.ks = []
        self.number_ele = []
        self.number_sets = []
        for _ in range(0, self.n):
            k = random.randint(min_k, max_k)
            self.ks.append(k)
            self.number_ele.append(2**(k + 1) - 2)
            self.number_sets.append(k + 2)
        shape = (sum(self.number_sets), sum(self.number_ele))
        self.problem_instance = np.zeros(shape)
        set_index_offset = 0
        ele_index_offset = 0
        for i in range(0, self.n):
            tight_offset = 0
            for k in range(0, self.ks[i]):
                # initialize S_k
                row = set_index_offset + k
                for l in range(0, 2**(k + 1)):
                    index = ele_index_offset + l + tight_offset
                    self.problem_instance[row][index] = 1
                tight_offset += 2**(k + 1)
            # T_is...
            t_index = set_index_offset + self.number_sets[i] - 2
            for k in range(0, self.number_ele[i]):
                if k % 2 == 0:
                    self.problem_instance[t_index][k + ele_index_offset] = 1
                else:
                    self.problem_instance[t_index + 1][k + ele_index_offset] = 1
            ele_index_offset += self.number_ele[i]
            set_index_offset += self.number_sets[i]
        self.is_solveable()

    def is_solveable(self):
        '''
        Checks if there exists a set cover at all. Raises an Exception if not.
        '''
        all_sets = np.ones(self.problem_instance.shape[0])
        solution = Solution(self, all_sets)
        if not solution.is_feasible_solution():
            raise Exception("Set cover instance cannot be solved")


class SteinerTriple:

    TRIPLES_27 = [
(2, 3, 4),
(1, 3, 5),
(1, 2, 6),
(5, 6, 7),
(4, 6, 8),
(4, 5, 9),
(1, 8, 9),
(2, 7, 9),
(3, 7, 8),
(1, 4, 7),
(2, 5, 8),
(3, 6, 9),
(11, 12, 13),
(10, 12, 14),
(10, 11, 15),
(14, 15, 16),
(13, 15, 17),
(13, 14, 18),
(10, 17, 18),
(11, 16, 18),
(12, 16, 17),
(10, 13, 16),
(11, 14, 17),
(12, 15, 18),
(20, 21, 22),
(19, 21, 23),
(19, 20, 24),
(23, 24, 25),
(22, 24, 26),
(22, 23, 27),
(19, 26, 27),
(20, 25, 27),
(21, 25, 26),
(19, 22, 25),
(20, 23, 26),
(21, 24, 27),
(1, 10, 19),
(1, 11, 24),
(1, 12, 23),
(1, 13, 25),
(1, 14, 21),
(1, 15, 20),
(1, 16, 22),
(1, 17, 27),
(1, 18, 26),
(2, 10, 24),
(2, 11, 20),
(2, 12, 22),
(2, 13, 21),
(2, 14, 26),
(2, 15, 19),
(2, 16, 27),
(2, 17, 23),
(2, 18, 25),
(3, 10, 23),
(3, 11, 22),
(3, 12, 21),
(3, 13, 20),
(3, 14, 19),
(3, 15, 27),
(3, 16, 26),
(3, 17, 25),
(3, 18, 24),
(4, 10, 25),
(4, 11, 21),
(4, 12, 20),
(4, 13, 22),
(4, 14, 27),
(4, 15, 26),
(4, 16, 19),
(4, 17, 24),
(4, 18, 23),
(5, 10, 21),
(5, 11, 26),
(5, 12, 19),
(5, 13, 27),
(5, 14, 23),
(5, 15, 25),
(5, 16, 24),
(5, 17, 20),
(5, 18, 22),
(6, 10, 20),
(6, 11, 19),
(6, 12, 27),
(6, 13, 26),
(6, 14, 25),
(6, 15, 24),
(6, 16, 23),
(6, 17, 22),
(6, 18, 21),
(7, 10, 22),
(7, 11, 27),
(7, 12, 26),
(7, 13, 19),
(7, 14, 24),
(7, 15, 23),
(7, 16, 25),
(7, 17, 21),
(7, 18, 20),
(8, 10, 27),
(8, 11, 23),
(8, 12, 25),
(8, 13, 24),
(8, 14, 20),
(8, 15, 22),
(8, 16, 21),
(8, 17, 26),
(8, 18, 19),
(9, 10, 26),
(9, 11, 25),
(9, 12, 24),
(9, 13, 23),
(9, 14, 22),
(9, 15, 21),
(9, 16, 20),
(9, 17, 19),
(9, 18, 27),
]

    TRIPLES_45 = [
(3,4,6),
(4,5,7),
(1,5,8),
(1,2,9),
(2,3,10),
(2,5,6),
(1,3,7),
(2,4,8),
(3,5,9),
(1,4,10),
(8,9,11),
(9,10,12),
(6,10,13),
(6,7,14),
(7,8,15),
(7,10,11),
(6,8,12),
(7,9,13),
(8,10,14),
(6,9,15),
(1,13,14),
(2,14,15),
(3,11,15),
(4,11,12),
(5,12,13),
(1,12,15),
(2,11,13),
(3,12,14),
(4,13,15),
(5,11,14),
(1,6,11),
(2,7,12),
(3,8,13),
(4,9,14),
(5,10,15),
(18,19,21),
(19,20,22),
(16,20,23),
(16,17,24),
(17,18,25),
(17,20,21),
(16,18,22),
(17,19,23),
(18,20,24),
(16,19,25),
(23,24,26),
(24,25,27),
(21,25,28),
(21,22,29),
(22,23,30),
(22,25,26),
(21,23,27),
(22,24,28),
(23,25,29),
(21,24,30),
(16,28,29),
(17,29,30),
(18,26,30),
(19,26,27),
(20,27,28),
(16,27,30),
(17,26,28),
(18,27,29),
(19,28,30),
(20,26,29),
(16,21,26),
(17,22,27),
(18,23,28),
(19,24,29),
(20,25,30),
(33,34,36),
(34,35,37),
(31,35,38),
(31,32,39),
(32,33,40),
(32,35,36),
(31,33,37),
(32,34,38),
(33,35,39),
(31,34,40),
(38,39,41),
(39,40,42),
(36,40,43),
(36,37,44),
(37,38,45),
(37,40,41),
(36,38,42),
(37,39,43),
(38,40,44),
(36,39,45),
(31,43,44),
(32,44,45),
(33,41,45),
(34,41,42),
(35,42,43),
(31,42,45),
(32,41,43),
(33,42,44),
(34,43,45),
(35,41,44),
(31,36,41),
(32,37,42),
(33,38,43),
(34,39,44),
(35,40,45),
(1,16,31),
(1,17,39),
(1,18,37),
(1,19,40),
(1,20,38),
(1,21,41),
(1,22,33),
(1,23,35),
(1,24,32),
(1,25,34),
(1,26,36),
(1,27,45),
(1,28,44),
(1,29,43),
(1,30,42),
(2,16,39),
(2,17,32),
(2,18,40),
(2,19,38),
(2,20,36),
(2,21,35),
(2,22,42),
(2,23,34),
(2,24,31),
(2,25,33),
(2,26,43),
(2,27,37),
(2,28,41),
(2,29,45),
(2,30,44),
(3,16,37),
(3,17,40),
(3,18,33),
(3,19,36),
(3,20,39),
(3,21,34),
(3,22,31),
(3,23,43),
(3,24,35),
(3,25,32),
(3,26,45),
(3,27,44),
(3,28,38),
(3,29,42),
(3,30,41),
(4,16,40),
(4,17,38),
(4,18,36),
(4,19,34),
(4,20,37),
(4,21,33),
(4,22,35),
(4,23,32),
(4,24,44),
(4,25,31),
(4,26,42),
(4,27,41),
(4,28,45),
(4,29,39),
(4,30,43),
(5,16,38),
(5,17,36),
(5,18,39),
(5,19,37),
(5,20,35),
(5,21,32),
(5,22,34),
(5,23,31),
(5,24,33),
(5,25,45),
(5,26,44),
(5,27,43),
(5,28,42),
(5,29,41),
(5,30,40),
(6,16,41),
(6,17,35),
(6,18,34),
(6,19,33),
(6,20,32),
(6,21,36),
(6,22,44),
(6,23,42),
(6,24,45),
(6,25,43),
(6,26,31),
(6,27,38),
(6,28,40),
(6,29,37),
(6,30,39),
(7,16,33),
(7,17,42),
(7,18,31),
(7,19,35),
(7,20,34),
(7,21,44),
(7,22,37),
(7,23,45),
(7,24,43),
(7,25,41),
(7,26,40),
(7,27,32),
(7,28,39),
(7,29,36),
(7,30,38),
(8,16,35),
(8,17,34),
(8,18,43),
(8,19,32),
(8,20,31),
(8,21,42),
(8,22,45),
(8,23,38),
(8,24,41),
(8,25,44),
(8,26,39),
(8,27,36),
(8,28,33),
(8,29,40),
(8,30,37),
(9,16,32),
(9,17,31),
(9,18,35),
(9,19,44),
(9,20,33),
(9,21,45),
(9,22,43),
(9,23,41),
(9,24,39),
(9,25,42),
(9,26,38),
(9,27,40),
(9,28,37),
(9,29,34),
(9,30,36),
(10,16,34),
(10,17,33),
(10,18,32),
(10,19,31),
(10,20,45),
(10,21,43),
(10,22,41),
(10,23,44),
(10,24,42),
(10,25,40),
(10,26,37),
(10,27,39),
(10,28,36),
(10,29,38),
(10,30,35),
(11,16,36),
(11,17,43),
(11,18,45),
(11,19,42),
(11,20,44),
(11,21,31),
(11,22,40),
(11,23,39),
(11,24,38),
(11,25,37),
(11,26,41),
(11,27,34),
(11,28,32),
(11,29,35),
(11,30,33),
(12,16,45),
(12,17,37),
(12,18,44),
(12,19,41),
(12,20,43),
(12,21,38),
(12,22,32),
(12,23,36),
(12,24,40),
(12,25,39),
(12,26,34),
(12,27,42),
(12,28,35),
(12,29,33),
(12,30,31),
(13,16,44),
(13,17,41),
(13,18,38),
(13,19,45),
(13,20,42),
(13,21,40),
(13,22,39),
(13,23,33),
(13,24,37),
(13,25,36),
(13,26,32),
(13,27,35),
(13,28,43),
(13,29,31),
(13,30,34),
(14,16,43),
(14,17,45),
(14,18,42),
(14,19,39),
(14,20,41),
(14,21,37),
(14,22,36),
(14,23,40),
(14,24,34),
(14,25,38),
(14,26,35),
(14,27,33),
(14,28,31),
(14,29,44),
(14,30,32),
(15,16,42),
(15,17,44),
(15,18,41),
(15,19,43),
(15,20,40),
(15,21,39),
(15,22,38),
(15,23,37),
(15,24,36),
(15,25,35),
(15,26,33),
(15,27,31),
(15,28,34),
(15,29,32),
(15,30,45)]

    def __init__(self, T27=True):
        if T27:
            TRIPLES = SteinerTriple.TRIPLES_27
        else:
            TRIPLES = SteinerTriple.TRIPLES_45
        rows = len(TRIPLES)
        cols = max(list(map(lambda x: max(x), TRIPLES)))
        problem_matrix = np.zeros((rows, cols))
        counter = 0
        for triple in TRIPLES:
            for ele in triple:
                problem_matrix[counter][ele - 1] = 1
            counter += 1
        self.problem_instance = problem_matrix 
        self.is_solveable()

    def is_solveable(self):
        '''
        Checks if there exists a set cover at all. Raises an Exception if not.
        '''
        all_sets = np.ones(self.problem_instance.shape[0])
        solution = Solution(self, all_sets)
        if not solution.is_feasible_solution():
            raise Exception("Set cover instance cannot be solved")


class Solution:
    '''
    Represents a possible infeasible solution of the Set Cover problem
    '''
    def __init__(self, set_cover_instance, set_vector, is_feasible=None, cost=None):
        '''
        :param set_cover_instance: instance of SetCover
        :param set_vector: numpy vector indicating the sets that the solution holds.
        The i-th entry of set_vector corresponds to the i-th row of the set cover
        table.
        :param is_feasible: indicates if the solution is a possible set cover.
        :param cost: number of sets in the cover.
        '''
        self.set_cover_instance = set_cover_instance
        self.set_vector = set_vector
        self.is_feasible = is_feasible
        self.cost = cost
        self.is_feasible_solution()

    def equals_other_sol(self, other_sol):
        for i in range(0, len(self.set_vector)):
            if self.set_vector[i] != other_sol.set_vector[i]:
                return False
        return True

    def add_set(self, index):
        '''
        Adds the set of the given index to the solution. Afterwards the cost is updated
        and is checked if the solution becomes feasible.

        :param index: index in the set cover table of the set.
        '''
        if self.set_vector[index] == 1:
            return False
        self.set_vector[index] = 1
        self.cost += 1
        self.covered_elements += self.set_cover_instance.problem_instance[index]
        self.covered_elements = [1 if ele > 0 else 0 for ele in self.covered_elements]
        if sum(self.covered_elements) == self.set_cover_instance.problem_instance.shape[1]:
            self.is_feasible = True
        self.covered = sum(self.covered_elements)
        return True

    def get_cost(self):
        if self.is_feasible():
            return self.cost
        else:
            return sys.maxsize

    def is_feasible_solution(self):
        '''
        Also retrieves the covered elements and calculates the solutions cost.
        '''
        if self.is_feasible is not None:
            return self.is_feasible
        available_elements = np.zeros(len(self.set_cover_instance.problem_instance[0]))
        cost = 0
        for i in range(0, len(self.set_vector)):
            if self.set_vector[i] == 1:
                cost += 1
                available_elements += self.set_cover_instance.problem_instance[i]
        self.covered_elements = [1 if ele > 0 else 0 for ele in available_elements]
        self.covered = sum(self.covered_elements)
        if len(available_elements[0 in available_elements]) == 0:
            self.cost = cost
            self.is_feasible = True
            return self.is_feasible
        self.cost = cost
        self.is_feasible = False
        return self.is_feasible
