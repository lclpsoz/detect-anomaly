import gen_dummy as gen
import matplotlib.pyplot as plt

gen_now = gen.gen_dummy(350, 1, 254, 170, 180, 0.005)
hist = gen_now.get_hist_spot(True)
print(hist)

plt.title("Quantidade de pixels por intensidade de cada canal")
plt.xlabel("Intensidade")
plt.ylabel("Quantidade")
for i in range(len(hist)):
    plt.plot(hist[i], ['R', 'G', 'B'][i])
plt.legend("RGB")
plt.show()