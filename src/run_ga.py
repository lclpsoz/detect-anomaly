import gen_dummy as gen
import ga

dataset = []
dataset_ids = ["test", "test2", "test3"]
for id in dataset_ids:
    dataset.extend(gen.gen_dummy.get_dataset(id))
ga_now = ga.ga(dataset)
ga_now.run()