try:
    from source.set_cover import *
    from source.set_cover import Solution
    from source.greedy import GreedyAlgorithm
    from source.population_initializer import PopulationCreator
except Exception as _:
    from set_cover import *
    from set_cover import Solution
    from greedy import GreedyAlgorithm
    from population_initializer import PopulationCreator
import random
import math
import copy
import time as Time

def linear_temperature(time):
    next_temp = 256.0 - time * 0.1
    return 0 if next_temp < 0 else next_temp


def propotional_temperature(time):
    gamma = 0.975
    next_temp = 256.0 * (gamma**time)
    return next_temp

class SimulatedAnnealing:

    BIT_FLIP = 0.1

    def __init__(self, problem_instance, iterations=1000, initial_solution=None, temperature_function=propotional_temperature):
        self.name = "SA"
        self.problem_instance = problem_instance
        self.greedy = GreedyAlgorithm(self.problem_instance)
        if initial_solution is not None:
            self.best_known_solution = initial_solution
        else:
            self.best_known_solution = PopulationCreator(problem_instance).create_random_instance()
        self.current_solution = self.best_known_solution
        self.temperature_function = temperature_function
        self.time = 0
        self.greedy.make_solution_feasible(self.current_solution)
        self.greedy.make_solution_feasible(self.best_known_solution)
        self.iterations = iterations

    def get_random_neighbour(self):
        '''
        Finds a neighbour of the current search solution by bitflipping its entry. Each
        entry is flipped with the propability BIT_FLIP

        :return : random neighbour
        '''
        number_of_sets = len(self.current_solution.set_vector)
        copied_sol_vector = copy.deepcopy(self.current_solution.set_vector)
        for i in range(0, number_of_sets):
            if random.random() < SimulatedAnnealing.BIT_FLIP:
                old_val = copied_sol_vector[i]
                copied_sol_vector[i] = 0 if old_val == 1 else 1
        random_solution = Solution(self.problem_instance, copied_sol_vector)
        self.greedy.make_solution_feasible(random_solution)
        set_vector = random_solution.set_vector
        for i in range(0, len(set_vector)):
            if set_vector[i] == 1:
                set_vector[i] = 0
                if not Solution(self.problem_instance, set_vector).is_feasible:
                    set_vector[i] = 1
        return random_solution

    def boltzmann_propability(self, rand_sol, timestamp):
        try:
            approx_cost = self.best_known_solution.cost
            rand_cost = rand_sol.cost
            temperature = self.temperature_function(timestamp)
            boltzmann_propability = math.exp((approx_cost - rand_cost) / temperature)
            return boltzmann_propability
        except Exception as _:
            return 1.0

    def perform_iteration(self):
        '''
        1. grabs random neighbour of current search solution
        2. if better solution then update approximation
        3. if not then use random neighbour as new search direction with boltzmann propability
        '''
        random_neighbour = self.get_random_neighbour()
        if random_neighbour.cost < self.best_known_solution.cost:
            old_best = self.best_known_solution
            self.best_known_solution = random_neighbour
            del old_best
            old_current = self.current_solution
            self.current_solution = random_neighbour
            del old_current
        else:
            prob = self.boltzmann_propability(random_neighbour, self.time)
            if random.random() < prob:
                old_current = self.current_solution
                self.current_solution = random_neighbour
                del old_current
        self.redundancy_check()

    def redundancy_check(self):
        sol_vector = copy.deepcopy(self.current_solution.set_vector)
        for i in range(0, len(sol_vector)):
            if sol_vector[i] == 1:
                sol_vector[i] = 0
                s = Solution(self.problem_instance, sol_vector)
                if not s.is_feasible_solution():
                    sol_vector[i] = 1
        self.current_solution = Solution(self.problem_instance, sol_vector)

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start SA")
        old_best = sys.maxsize
        found = 0
        s = Time.time()
        while self.time < self.iterations:
            iter_start = Time.time()
            self.perform_iteration()
            self.time += 1
            if self.time % 10 == 0:
                print_now()
                print("finished " + str(self.time / self.iterations) + "percent of SA")
            iter_end = Time.time()
            if self.best_known_solution.cost != sys.maxsize:
                if self.best_known_solution.cost < old_best:
                    found = self.time
                    old_best = self.best_known_solution.cost
                else:
                    if has_converged(found, self.time, Time.time() - s):
                        break
            self.logger.log_entry(self.time, self.best_known_solution.cost, float(iter_end - iter_start))
        print("finished SA")
        return self.best_known_solution
