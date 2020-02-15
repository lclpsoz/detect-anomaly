import proc_img as gen
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

dataset = []
# dataset_ids = ["test", "test2", "test3", "test4_n_100", "test5_n_1000"]
dataset_ids = ["ga_list_img_batch_n_100_r_35_p_0-01_train"]
for id in dataset_ids:
    dataset.extend(gen.procImg.get_dataset(id))

results = []
data = []
lst = []
seed = random.randint(0, 2**64)
# seed = 10854117184287281751
random.seed(seed)
for i in range(5):
    ga_now = None
    ga_now = ga.ga(random.choices(dataset, k=1000),
                population_size=100,
                mutation_factor=0.01,
                generations=20,
                print=True)
    best_gene = ga_now.run()
    print(i+1, "eval = %.3f %%" % (eval_indv(best_gene, dataset)*100))
    results.append(eval_indv(best_gene, dataset)*100)
    lst.append(deepcopy(best_gene))

print(results)