import random
import numpy as np
from time import time

class ga:
    def __init__(self, dataset, **kwargs):
        self.dataset_hist = np.array([np.array(x) for x in np.array(dataset)[:,0]])
        self.dataset_info = np.array(dataset)[:,1]
        self.mutation_factor = kwargs.get('mutation_factor')
        self.population_size = kwargs.get('population_size')
        if(not self.mutation_factor):
            self.mutation_factor = 0.01
        if(not self.population_size):
            self.population_size = 100

    def __generate_indv(self):
        return np.array([[random.uniform(-1, 1) for x in range(256)] for x in range(3)])

    def __fitness(self, indv):
        # print("__fitness")
        fitness = 0
        for i in range(len(self.dataset_hist)):
            mult = indv * self.dataset_hist[i]
            anomaly = True
            if(sum(sum(mult)) < 0):
                anomaly = False
            if(anomaly == self.dataset_info[i]):
                fitness+=1

        # print(fitness/len(self.dataset_hist))
        return fitness/len(self.dataset_hist)
    
    def __mutation(self, indv):
        for i in range(len(indv)):
            for j in range(len(indv[0])):
                if(random.random() < self.mutation_factor):
                    indv[i][j] = random.uniform(-1, 1)
        return indv

    def run(self):
        seed = random.randint(0, 2**64)
        print("seed =", seed)
        random.seed(seed)
        self.population = []
        print("Generate population")
        for i in range(self.population_size):
            bef = time()
            self.population.append(self.__generate_indv())
            print(i, self.__fitness(self.population[i]), time() - bef)

        print("Mutation")
        for i in range(self.population_size):
            bef = time()
            self.population[i] = self.__mutation(self.population[i])
            print(i, self.__fitness(self.population[i]))
            print(i, self.__fitness(self.population[i]), time() - bef)