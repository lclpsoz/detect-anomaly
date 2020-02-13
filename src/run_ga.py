import gen_dummy as gen
import ga

dataset = []
dataset_ids = ["test", "test2", "test3"]
for id in dataset_ids:
    dataset.extend(gen.gen_dummy.get_dataset(id))
ga_now = ga.ga(dataset, population_size=10, mutation_factor=0.00001)
ga_now.run()