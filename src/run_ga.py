import gen_dummy as gen
import ga
import numpy as np
import random
from copy import deepcopy

def eval_indv(gene, dataset):
    dataset_hist = np.array([np.array(x) for x in np.array(dataset)[:,0]])
    dataset_info = np.array(dataset)[:,1]
    # print("__fitness")
    fitness = 0
    for i in range(len(dataset_hist)):
        mult = gene * dataset_hist[i]
        anomaly = True
        if(sum(sum(mult)) < 0):
            anomaly = False
        if(anomaly == dataset_info[i]):
            fitness+=1
        # if((i+1)%100 == 0):
        #     print("Step %d = %.2f %%" % (i+1, 100*fitness/(i+1)))

    # print(fitness/len(self.dataset_hist))
    return fitness/len(dataset_hist)

dataset_id = "ga_list_img_batch_n_100_r_35_p_0-01_train"
dataset = gen.procImg.get_dataset(dataset_id)

seed = random.randint(0, 2**64)
# seed = 10854117184287281751
random.seed(seed)
ga_now = None
k = 10000
population_size = 200
generations = 200
mutation_factor = 0.01
ga_name = "%s_k_%d_pop_%d_gen_%d_mut_%.4f" % (dataset_id, k, population_size, generations, mutation_factor)
print(ga_name)
ga_now = ga.ga(random.choices(dataset, k=k),
            population_size=population_size,
            mutation_factor=mutation_factor,
            generations=generations,
            print=True)

ga_now.show_history('ga_history_1581732313')

history = ga_now.run(ga_name)
