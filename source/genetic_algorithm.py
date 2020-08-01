try:
    from source.set_cover import *
    from source.population_initializer import PopulationCreator
    from source.greedy import GreedyAlgorithm
except Exception as _:
    from set_cover import *
    from population_initializer import PopulationCreator
    from greedy import GreedyAlgorithm
import math
import random
import copy
import time

class GeneticAlgorithm:

    def __init__(self, problem_instance, population_size, K, mutation_prob, iterations):
        self.name = "GA"
        self.problem_instance = problem_instance
        self.pop_size = population_size
        pop_creater = PopulationCreator(problem_instance)
        self.population = pop_creater.create_population_beasley(population_size)
        best_val = min(list(map(lambda x: x.cost, self.population)))
        self.current_best = list(filter(lambda x: x.cost == best_val, self.population))[0]
        self.k = K
        self.mutation_prob = mutation_prob
        self.number_of_bits = len(self.problem_instance.problem_instance)
        self.greedy = GreedyAlgorithm(problem_instance)
        self.population.append(self.greedy.get_greedy_solution())
        self.iterations = iterations

    def selection(self):
        '''
        returns the best gene that has been found after
        K random selected genes.
        :return : best solution among them
        '''
        N = math.floor(random.uniform(0, 1) * len(self.population))
        best = self.population[N]
        for _ in range(0, self.k - 1):
            N = math.floor(random.uniform(0, 1) * len(self.population))
            if(self.population[N].cost < best.cost):
                best = self.population[N]
        return best

    def crossover(self, sol_a, sol_b):
        '''
        one point crossover
        '''
        split_point = int(random.random() * self.number_of_bits)
        for i in range(0, self.number_of_bits):
            if i <= split_point:
                sol_a.set_vector[i] = sol_b.set_vector[i]
            else:
                sol_b.set_vector[i] = sol_a.set_vector[i]

    def mutation(self, solution):
        '''
        creep mutation
        '''
        for i in range(0, self.number_of_bits):
            if random.random() < self.mutation_prob:
                solution.set_vector[i] = 0 if solution.set_vector[i] == 1 else 0

    def del_from_pop(self):
        '''
        average cost based deletion
        '''
        while len(self.population) > self.pop_size:
            avg = sum(list(map(lambda x: x.cost, self.population))) / len(self.population)
            cost_above_avg = list(filter(lambda x: x.cost >= avg, self.population))
            index = random.randint(0, len(cost_above_avg) - 1)
            self.population.remove(self.population[index])

    def iteration(self):
        parent_a = self.selection()
        parent_b = self.selection()
        child_a = copy.deepcopy(parent_a)
        child_b = copy.deepcopy(parent_b)
        self.crossover(child_a, child_b)
        self.mutation(child_a)
        self.mutation(child_b)
        child_a.is_feasible = None
        child_b.is_feasible = None
        child_a.is_feasible_solution()
        child_b.is_feasible_solution()
        self.greedy.make_solution_feasible(child_a)
        self.greedy.make_solution_feasible(child_b)
        if child_a.cost < self.current_best.cost:
            self.current_best = child_a
        if child_b.cost < self.current_best.cost:
            self.current_best = child_b
        self.population.append(child_a)
        self.population.append(child_b)
        self.del_from_pop()

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start GA")
        s = time.time()
        found = 0
        old_best = sys.maxsize
        for i in range(0, self.iterations):
            iter_start = time.time()
            self.iteration()
            self.mutation_prob *= self.mutation_prob
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + "percent of GA")
            iter_end = time.time()
            if self.current_best.cost != sys.maxsize:
                if self.current_best.cost < old_best:
                    found = i
                    old_best = self.current_best.cost
                else:
                    if has_converged(found, i, time.time() - s):
                        break
            self.logger.log_entry(i, self.current_best.cost, float(iter_end - iter_start))
        print("finished GA")
        return self.current_best
