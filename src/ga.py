import random
import numpy as np
from time import time
from copy import deepcopy
import os
import json
import matplotlib.pyplot as plt
from threading import Thread
from multiprocessing import Queue
from signal import signal, SIGPIPE, SIG_DFL
import concurrent.futures


def f(dataset, dataset_info, gene):
    fitness = 0
    for i in range(len(dataset)):
        mult = gene * dataset[i]
        anomaly = True
        if(sum(sum(mult)) < 0):
            anomaly = False
        if(anomaly == dataset_info[i]):
            fitness+=1
    return fitness

class ga:
    def __init__(self, dataset, **kwargs):
        self.dataset_hist = np.array([np.array(x) for x in np.array(dataset)[:,0]])
        self.dataset_info = np.array(dataset)[:,1]
        self.mutation_factor = kwargs.get('mutation_factor')
        self.population_size = kwargs.get('population_size')
        self.generations = kwargs.get('generations')
        self.print = kwargs.get('print')
        if(not self.mutation_factor):
            self.mutation_factor = 0.01
        if(not self.population_size):
            self.population_size = 100
        if(not self.generations):
            self.generations = 100
        if(not self.print):
            self.print = False

    def __generate_indv(self):
        gene = np.array([[random.uniform(-1, 1) for x in range(256)] for x in range(3)])
        return {'gene' : gene,
                'fitness' : self.__fitness(gene)}

    def __fitness(self, gene):
        que = Queue()
        n_threads = 8
        fitness = 0
        threads = []
        
        with concurrent.futures.ProcessPoolExecutor(n_threads) as executor:
            future = []
            for i in range(n_threads):
                l = (i*len(self.dataset_hist))//n_threads
                r = ((i+1)*len(self.dataset_hist))//n_threads
                dt = self.dataset_hist[l:r]
                dt_info = self.dataset_info[l:r]
                future.append(executor.submit(f, dt, dt_info, gene))
            fitness += sum([x.result() for x in future])

        """
        for i in range(n_threads):
            l = (i*len(self.dataset_hist))//n_threads
            r = ((i+1)*len(self.dataset_hist))//n_threads
            dataset_now = deepcopy(self.dataset_hist[l:r])
            dataset_info_now = deepcopy(self.dataset_info[l:r])
            threads.append(Thread(target=lambda que, dataset_now, dataset_info_now, gene: que.put(self.__f(dataset_now, dataset_info_now, gene)), args=(que, dataset_now, dataset_info_now, gene)))
            threads[-1].start()
        """

        for t in threads:
            t.join()

        while not que.empty():
            fitness += que.get()
        return fitness/len(self.dataset_hist)
    
    def __mutation(self, indv):
        gene = indv['gene']
        total_cells = len(gene)*len(gene[0])
        amount_to_mutate = round((total_cells*self.mutation_factor)*random.normalvariate(0, 2))
        # print(amount_to_mutate)
        for i in range(amount_to_mutate):
            gene[random.randint(0, len(gene)-1)][random.randint(0, len(gene[0])-1)] = random.uniform(-1, 1)
        indv['fitness'] = self.__fitness(gene)
        return indv

    def __crossover(self, indv1, indv2):
        total_fitness = indv1['fitness'] + indv2['fitness']
        odd_indv1 = indv1['fitness']/total_fitness
        descendent = deepcopy(indv2)
        gene_indv1 = indv1['gene']
        gene_descendent = descendent['gene']
        for i in range(len(gene_descendent)):
            for j in range(len(gene_descendent[0])):
                if(random.random() < odd_indv1):
                    gene_descendent[i][j] = gene_indv1[i][j]
        descendent['fitness'] = self.__fitness(gene_descendent)

        return self.__mutation(descendent)

    def __func(self, indv):
        return -indv['fitness']

    def __selection(self):
        total_fitness = sum([x['fitness'] for x in self.population])
        # print(total_fitness)
        offspring = []
        elitism = []
        # print([x['fitness'] for x in self.population])
        self.population.sort(key=self.__func)
        # print([x['fitness'] for x in self.population])
        for i in range(round(0.1*self.population_size)):
            elitism.append(self.population[i])
        for i in range(self.population_size):
            acu = 0
            rand_val_1 = random.uniform(0, total_fitness)
            rand_val_2 = random.uniform(0, total_fitness)
            indv1 = indv2 = None
            for j in range(self.population_size):
                acu += self.population[j]['fitness']
                if(indv1 == None and acu >= rand_val_1):
                    indv1 = self.population[j]
                if(indv2 == None and acu >= rand_val_2):
                    indv2 = self.population[j]
                if(indv1 != None and indv2 != None):
                    break
            offspring.append(self.__crossover(indv1, indv2))
        self.population = offspring
        self.population.sort(key=self.__func)
        for i in range(round(0.1*self.population_size)):
            del(self.population[-1])
        self.population.extend(elitism)
                    
    def __save_ga_history(self, history, name):
        ga_path = os.path.join('..', 'ga')
        if(not os.path.exists(ga_path)):
            os.makedirs(ga_path)
        with open(os.path.join(ga_path, name + "_" + str(int(time())) + '.json'), 'w') as fl:
            json.dump(history, fl)

    @staticmethod
    def show_history(name):
        ga_path = os.path.join('..', 'ga')
        if(not os.path.exists(ga_path)):
            print("Folder doesn't exist!")
            exit()
        with open(os.path.join(ga_path, name + '.json'), 'r') as fl:
            history = json.load(fl)
        fitness = []
        for x in history:
            if(isinstance(history[x], dict)):
                fitness.append(history[x]['fitness'])
        print(fitness)
        plt.figure(dpi=200)
        plt.plot([max(x) for x in fitness])
        plt.plot([sum(x)/len(x) for x in fitness])
        plt.ylim((0.5, 1))
        plt.xlim((0, 100))
        plt.legend(["Best individual", "Average individual"])
        plt.title("Genetic Algorithm")
        plt.xlabel("Generations")
        plt.ylabel("Precision")
        plt.show()

    def run(self, name):
        seed = random.randint(0, 2**64)
        # seed = 10854117184287281751
        history = {}
        history['seed'] = seed
        history['mutation_factor'] = self.mutation_factor
        history['population_size'] = self.population_size
        history['generations'] = self.generations
        if(self.print):
            print("seed =", seed)
        random.seed(seed)
        self.population = []
        if(self.print):
            print("Generate population")
        for i in range(self.population_size):
            bef = time()
            self.population.append(self.__generate_indv())
            if(self.print):
                print("indv_%0*d" % (len(str(self.population_size)), i+1), self.population[i]['fitness'], time() - bef)

        if(self.print):
            print("Mutation")
        for i in range(self.population_size):
            bef = time()
            self.population[i] = self.__mutation(self.population[i])
            if(self.print):
                print("indv_%0*d" % (len(str(self.population_size)), i+1), self.population[i]['fitness'], time() - bef)

        if(self.print):
            print("Generation 0")
            print("\t", "Best fitness =", max([x['fitness'] for x in self.population]))
        maxi = 0
        for indv in self.population:
            if(indv['fitness'] > maxi):
                maxi = indv['fitness']
                indv_best = indv
        history[0] = {'fitness' : [x['fitness'] for x in self.population],
                    'best_indv' : indv_best}
        for i in range(self.generations):
            if(self.print):
                print("Generation", i + 1)
            bef = time()
            self. __selection()
            if(self.print):
                print("\t", "Time in selection = %.2f s" % (time() - bef))
                print("\t", "Best fitness =", max([x['fitness'] for x in self.population]))
                print("\t", "Average fitness =", sum([x['fitness'] for x in self.population])/self.population_size)
            maxi = 0
            for indv in self.population:
                if(indv['fitness'] > maxi):
                    maxi = indv['fitness']
                    indv_best = indv
            history[i] = {'fitness' : [x['fitness'] for x in self.population],
                        'best_indv' : [x.tolist() for x in indv_best['gene']]}
        self.__save_ga_history(history, name)

        return history