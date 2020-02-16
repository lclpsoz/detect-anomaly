import gen_dummy as gen
import matplotlib.pyplot as plt
from time import time

def generate_n_imgs(n, name):
    n = 10000
    bef = time()
    gen_now.generate_img_batch(name, n//2, True)
    gen_now.generate_img_batch(name, n//2, False)
    print("time:", time() - bef)
    print("time per img:", (time() - bef)/n)


def sick_vs_normal_spots():
    fig = plt.figure(dpi=120, figsize=(8, 16))
    plt.suptitle("Sick vs Normal spots")
    for i in range(4):
        a = fig.add_subplot(4, 2, 2*i + 1)
        plt.imshow(gen_now.generate_img(False))
        plt.title("Sick spot")
        a = fig.add_subplot(4, 2, 2*i + 2)
        plt.imshow(gen_now.generate_img(True))
        plt.title("Normal spot")
    plt.show()

def histograms():
    for i in range(1):
        img = gen_now.generate_img(True)
        vals_spot = gen_now.get_vals_spot(img)
        plt.figure(dpi=120, figsize=(30, 12))
        plt.suptitle("RGB Histograms")

        for j in range(2):
            plt.subplot(2, 4, 4*j + 1)
            plt.title("Spot")
            plt.imshow(img)
            for i in range(len(vals_spot)):
                plt.subplot(2, 4, 4*j + i+2)
                if(i==2):
                    plt.axvline(170, color='gray')
                    plt.axvline(181, color='gray')
                    lst = [0 for x in range(256)]
                    for x in vals_spot[i]:
                        lst[x]+=1 
                    plt.axhline(sum(lst)/len(lst), color='black')
                plt.title(["RED", "GREEN", "BLUE"][i] + " Histogram")
                plt.hist(vals_spot[i], bins=255, range=(0, 255), color=['r', 'g', 'b'][i])
        plt.show()


gen_now = gen.procImg(100, 1, 254, 170, 180, 0.01, [255, 224, 189],
                        [255, 0, 255], 35, [False, False, True])

gen_now.generate_dataset('img_batch_n_100_r_35_p_0-01_test')