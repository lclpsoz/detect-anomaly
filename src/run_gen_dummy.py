import gen_dummy as gen
import matplotlib.pyplot as plt

gen_now = gen.gen_dummy(100, 1, 254, 170, 180, 0.017)
hist = gen_now.get_hist_spot(True)
print(hist)

plt.title("Quantidade de pixels por intensidade de cada canal")
plt.xlabel("Intensidade")
plt.ylabel("Quantidade")
for i in range(len(hist)):
    plt.plot(hist[i], ['r', 'g', 'b'][i])
plt.legend("RGB")
plt.show()

gen_now.generate_dataset(8000, 0.6, "test3")
# print(gen_now.get_population("test"))