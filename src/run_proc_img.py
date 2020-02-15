import proc_img as gen
import matplotlib.pyplot as plt
from time import time

gen_now = gen.procImg(100, 1, 254, 170, 180, 0.01, [255, 224, 189],
                        [255, 0, 255], 35, [False, False, True])

gen_now.generate_dataset("img_batch_n_100_r_35_p_0-01_train")

exit()

n = 10000
bef = time()
gen_now.generate_img_batch('img_batch_n_100_r_35_p_0-01', n//2, True)
gen_now.generate_img_batch('img_batch_n_100_r_35_p_0-01', n//2, False)
print("time:", time() - bef)
print("time per img:", (time() - bef)/n)

exit()

for i in range(1):
    img = gen_now.generate_img(False)
    gen_now.imshow(img)
    vals_spot = gen_now.get_vals_spot(img)
    plt.suptitle("RGB Histograms")
    for i in range(len(vals_spot)):
        plt.subplot(1, 3, i+1)
        plt.title(["RED", "GREEN", "BLUE"][i] + " Histogram")
        plt.hist(vals_spot[i], bins=255, range=(0, 255), color=['r', 'g', 'b'][i])
    plt.show()
for i in range(1):
    img = gen_now.generate_img(True)
    gen_now.imshow(img)
    vals_spot = gen_now.get_vals_spot(img)
    plt.suptitle("RGB Histograms")
    for i in range(len(vals_spot)):
        plt.subplot(1, 3, i+1)
        plt.title(["RED", "GREEN", "BLUE"][i] + " Histogram")
        if(i==2):
            plt.axvline(170, color='gray')
            plt.axvline(181, color='gray')
            lst = [0 for x in range(256)]
            for x in vals_spot[i]:
                lst[x]+=1 
            plt.axhline(sum(lst)/len(lst), color='black')
        plt.hist(vals_spot[i], bins=255, range=(0, 255), color=['r', 'g', 'b'][i])
    plt.show()

exit()


gen_now = gen.gen_dummy(100, 1, 254, 170, 180, 0.008)
hist = gen_now.get_hist_spot(True)
print(hist)

plt.title("Quantidade de pixels por intensidade de cada canal")
plt.xlabel("Intensidade")
plt.ylabel("Quantidade")
for i in range(len(hist)):
    plt.plot(hist[i], ['r', 'g', 'b'][i])
plt.legend("RGB")
plt.show()

gen_now.generate_dataset(10000, 0.6, "test6_odd_0.008_n_10000")
# print(gen_now.get_population("test"))