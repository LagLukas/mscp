import random
import sys
import numpy as np
import copy
import time

try:
    from source.set_cover import *
    from source.greedy import GreedyAlgorithm
    from source.population_initializer import PopulationCreator
except Exception as _:
    from set_cover import *
    from greedy import GreedyAlgorithm
    from population_initializer import PopulationCreator



# neighbourhood operator
def two_exchange_neighbour(sol_vector):
    number_of_bits = len(sol_vector)
    a = int(random.random() * number_of_bits)
    b = int(random.random() * number_of_bits)
    new_sol_vector = copy.deepcopy(sol_vector)
    new_sol_vector[a] = sol_vector[b]
    new_sol_vector[b] = sol_vector[a]
    return new_sol_vector


# other neighbourhood operator
def pertubation_heuristic(sol_vector, problem):
    # cost efficiency vector
    new_sol_vector = copy.deepcopy(sol_vector)
    worst_cost_eff = -1
    worst_cost_eff_val = sys.maxsize
    for i in range(0, len(sol_vector)):
        if sol_vector[i] == 1:
            cost_eff = sum(problem.problem_instance[i])
            if worst_cost_eff_val < cost_eff:
                worst_cost_eff = i
                worst_cost_eff_val = cost_eff
    new_sol_vector[worst_cost_eff] = 0
    solution = Solution(problem, new_sol_vector)
    while not solution.is_feasible:
        repair_efficiencies = []
        for i in range(0, problem.problem_instance.shape[0]):
            eff = solution.covered_elements - problem.problem_instance[i]
            eff = [1 if ele < 0 else 0 for ele in eff]
            repair_efficiencies.append(sum(eff))
        probs = []
        eff_sum = sum(repair_efficiencies)
        for i in range(0, problem.problem_instance.shape[0]):
            probs.append(repair_efficiencies[i] / eff_sum)
        index = np.random.choice(range(0, len(sol_vector)), replace=False, size=1, p=probs)[0]
        new_sol_vector[index] = 1
        solution = Solution(problem, new_sol_vector)
    return new_sol_vector


# decomposition operator
def half_total_exchange(sol_vector):
    number_of_bits = len(sol_vector)
    new1 = np.zeros(number_of_bits)
    sample1 = random.sample(range(0, number_of_bits), int(number_of_bits / 2))
    new2 = np.zeros(number_of_bits)
    sample2 = random.sample(range(0, number_of_bits), int(number_of_bits / 2))
    for i in range(0, len(sample1)):
        new1[sample1[i]] = sol_vector[sample1[i]]
        new2[sample1[i]] = sol_vector[sample2[i]]
    return sample1, sample2


# dec operator
def scp_dec(sol_vector, problem):
    s1 = sol_vector
    s2 = sol_vector
    for _ in range(0, 10):
        s1 = pertubation_heuristic(s1, problem)
        s2 = pertubation_heuristic(s2, problem)
    return s1, s2


# synthesis operator
def probabilistic_select(sol_vector_a, sol_vector_b):
    number_of_bits = len(sol_vector_a)
    new_sol_vector = np.zeros(number_of_bits)
    for i in range(0, number_of_bits):
        if random.random() < 0.5:
            new_sol_vector[i] = sol_vector_a[i]
        else:
            new_sol_vector[i] = sol_vector_b[i]
    return new_sol_vector


# syn operator
def scp_syn(sol_vector_a, sol_vector_b, problem):
    cost_a = sum(sol_vector_a)
    cost_b = sum(sol_vector_b)
    indexes = []
    prob_a = cost_a / (cost_a + cost_b)
    for ele in range(0, problem.problem_instance.shape[1]):
        sol = sol_vector_b
        if random.random() < prob_a:
            sol = sol_vector_a
        for i in range(0, len(sol)):
            if sol[i] == 1 and problem.problem_instance[i][ele] == 1:
                indexes.append(i)
                break
    number_of_bits = len(sol_vector_a)
    new_sol_vector = np.zeros(number_of_bits)
    for ele in indexes:
        new_sol_vector[ele] = 1
    # solution = Solution(problem, new_sol_vector)
    return new_sol_vector


class Molecule:

    INITIAL_KINETIC_ENERGY = 1000
    KE_LOSS_RATE = 0.1

    def __init__(self, problem_instance, solution, buffer):
        self.kinetic_energy = Molecule.INITIAL_KINETIC_ENERGY
        self.hits = 0
        self.min_hits = 0
        self.problem_instance = problem_instance
        self.solution = solution
        self.buffer = buffer
        self.min_pe = self.solution.cost
        self.min_struct = self.solution

    def on_wall_ineffective_collision(self):
        neighbour = pertubation_heuristic(self.solution.set_vector, self.problem_instance)
        other_solution = Solution(self.problem_instance, neighbour)
        self.hits += 1
        if self.solution.cost + self.kinetic_energy >= other_solution.cost:
            a = random.random() * (1 - Molecule.KE_LOSS_RATE) + Molecule.KE_LOSS_RATE
            energy = (self.solution.cost - other_solution.cost + self.kinetic_energy)
            new_ke = energy * a
            self.buffer[0] = self.buffer[0] + energy * (1 - a)
            self.solution = other_solution
            self.kinetic_energy = new_ke
            if self.solution.cost < self.min_pe:
                self.min_struct = copy.deepcopy(self.solution)
                self.min_pe = self.solution.cost
                self.min_hits = self.hits

    def update_during_composition(self, mol_a, mol_b, E_dec):
        delta_3 = random.random()
        mol_a.kinetic_energy = delta_3 * E_dec
        mol_b.kinetic_energy = (1 - delta_3) * E_dec

    def decomposition(self):
        sol_vector_a, sol_vector_b = scp_dec(self.solution.set_vector, self.problem_instance)
        sol_a = Solution(self.problem_instance, sol_vector_a)
        mol_a = Molecule(self.problem_instance, sol_a, self.buffer)
        sol_b = Solution(self.problem_instance, sol_vector_b)
        mol_b = Molecule(self.problem_instance, sol_b, self.buffer)
        if self.solution.cost + self.kinetic_energy >= mol_a.solution.cost + mol_b.solution.cost:
            E_dec = self.solution.cost + self.kinetic_energy - (mol_a.solution.cost + mol_b.solution.cost)
            self.update_during_composition(mol_a, mol_b, E_dec)
        else:
            delta_1 = random.random()
            delta_2 = random.random()
            E_dec = self.solution.cost + self.kinetic_energy + delta_1 * delta_2 * self.buffer[0] - (mol_a.solution.cost + mol_b.solution.cost)
            if E_dec >= 0:
                self.buffer[0] = (1.0 - delta_1 * delta_2) * self.buffer[0]
                self.update_during_composition(mol_a, mol_b, E_dec)
            else:
                self.hits += 1
                del mol_a
                del mol_b
                return None
        return mol_a, mol_b

    def intermolecular_ineffective_collision(self, other_mol):
        neighbour_a = pertubation_heuristic(self.solution.set_vector, self.problem_instance)
        sol_a = Solution(self.problem_instance, neighbour_a)
        mol_a = Molecule(self.problem_instance, sol_a, self.buffer)
        mol_a.hits += 1
        neighbour_b = pertubation_heuristic(other_mol.solution.set_vector, self.problem_instance)
        sol_b = Solution(self.problem_instance, neighbour_b)
        mol_b = Molecule(self.problem_instance, sol_b, self.buffer)
        mol_b.hits += 1
        E_inter = (self.solution.cost + self.kinetic_energy + other_mol.solution.cost + other_mol.kinetic_energy)
        E_inter = E_inter - (mol_a.solution.cost + mol_b.solution.cost)
        if E_inter >= 0:
            delta_4 = random.random()
            mol_a.kinetic_energy = E_inter * delta_4
            mol_b.kinetic_energy = E_inter * (1 - delta_4)
            omega_1 = pertubation_heuristic(neighbour_a, self.problem_instance)
            omega_2 = pertubation_heuristic(neighbour_b, self.problem_instance)
            sol_1 = Solution(self.problem_instance, omega_1)
            sol_2 = Solution(self.problem_instance, omega_2)
            self.solution = sol_1
            self.kinetic_energy = mol_a.kinetic_energy
            if self.solution.cost < self.min_pe:
                self.min_pe = self.solution.cost
                self.min_struct = copy.deepcopy(self.solution)
                self.min_hits = self.hits
            other_mol.solution = sol_2
            other_mol.kinetic_energy = mol_b.kinetic_energy
            if other_mol.solution.cost < other_mol.solution.cost:
                other_mol.min_pe = other_mol.solution.cost
                other_mol.min_struct = copy.deepcopy(self.solution)
                other_mol.min_hits = other_mol.hits

    def synthesis(self, other_mol):
        sol_vector = scp_syn(self.solution.set_vector, other_mol.solution.set_vector, self.problem_instance)
        mol_a = Molecule(self.problem_instance, Solution(self.problem_instance, sol_vector), self.buffer)
        energy_sum = self.solution.cost + self.kinetic_energy + other_mol.solution.cost + other_mol.kinetic_energy
        if energy_sum > mol_a.solution.cost:
            mol_a.kinetic_energy = energy_sum - mol_a.solution.cost
            return mol_a
        else:
            self.hits += 1
            other_mol.hits += 1
            del mol_a
            return None


class CRO:

    def __init__(self, problem_instance, pop_size, mole_coll, buffer_val, alpha, beta, iterations):
        self.name = "CRO"
        self.problem_instance = problem_instance
        self.mole_coll = mole_coll
        self.buffer = np.zeros(1)
        self.buffer[0] = buffer_val
        creator = PopulationCreator(problem_instance)
        sol_pop = creator.create_population_cro(pop_size)
        self.population = list(map(lambda x: Molecule(problem_instance, x, self.buffer), sol_pop))
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.best = min(self.population, key=lambda x: x.min_pe).min_struct

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start CRO")
        old_best = sys.maxsize
        found = 0
        s = time.time()
        for i in range(0, self.iterations):
            iter_start = time.time()
            b = random.random()
            if b > self.mole_coll:
                index = int(random.random() * len(self.population))
                molecule = self.population[index]
                if molecule.hits - molecule.min_hits > self.alpha:
                    ret_val = molecule.decomposition()
                    if ret_val is not None:
                        # del old molecule
                        self.population.pop(index)
                        del molecule
                        self.population.append(ret_val[0])
                        self.population.append(ret_val[1])
                else:
                    molecule.on_wall_ineffective_collision()
            else:
                index_a = int(random.random() * len(self.population))
                index_b = int(random.random() * len(self.population))
                index_b = index_b - 1 if index_a == index_b else index_b
                mole_a = self.population[index_a]
                mole_b = self.population[index_b]
                if mole_a.kinetic_energy <= self.beta and mole_b.kinetic_energy <= self.beta:
                    ret_val = mole_a.synthesis(mole_b)
                    if ret_val is not None:
                        if index_a < index_b:
                            index_b = index_b - 1
                        elif index_a > index_b:
                            temp = index_a
                            index_a = index_b
                            index_b = temp - 1
                        self.population.pop(index_a)
                        if index_b > index_a:
                            self.population.pop(index_b)
                        del mole_a
                        del mole_b
                        self.population.append(ret_val)
                else:
                    mole_a.intermolecular_ineffective_collision(mole_b)
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + "percent of CRO")
            feasible_sols = list(filter(lambda x: x.solution.is_feasible, self.population))
            current_best = min(feasible_sols, key=lambda x: x.min_pe).min_struct
            if current_best.cost < self.best.cost:
                self.best = current_best
                current_best = min(self.population, key=lambda x: x.min_pe).min_struct
            iter_end = time.time()
            if self.best.cost != sys.maxsize:
                if self.best.cost < old_best:
                    found = i
                    old_best = self.best.cost
                else:
                    if has_converged(found, i, time.time() - s):
                        break
            self.logger.log_entry(i, self.best.cost, float(iter_end - iter_start))
        if current_best.cost < self.best.cost:
            self.best = current_best
        print("finished CRO")
        return self.best
