from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import seaborn as sns
sns.set_theme(style="darkgrid")

model = LinearRegression()

# Generate some random data
# np.random.seed(1221345)
points = 500
lim_n, lim_p = -7, 6
x = np.linspace(lim_n, lim_p, points)
y = np.linspace(lim_n, lim_p, points)
random_noise = np.random.randn(points)  # Noise
x = x + random_noise
y = y - random_noise


def mse(x, y, w, b):
    return ((w * x + b - y)**2 / x.shape[0]).sum()


def cp(x, y, w, b):
    return ( (w * x + b - y)**2 )


w_s, e_s = [], []
def mm():
    """
    Plot and predict data on the go...
    """
    for i in range(2, x.shape[0]):
        if i%10 == 0:
            plt.clf()
            x_t, y_t = x[:i], y[:i]
            model.fit(x_t.reshape(-1, 1), y_t)
            w, b = model.coef_, model.intercept_

            # Cost function line
            plt.subplot(221)
            er = mse(x_t, y_t, w, b)
            w_s.append(w)
            e_s.append(er)
            plt.plot(w_s, e_s, 'g.')
            plt.title("Cost function")
            plt.xlabel("W")
            plt.ylabel("Cost")

            # Linear regression line
            plt.subplot(222)    
            plt.xlim(lim_n-2, lim_p+2)
            plt.ylim(lim_n-2, lim_p+2)
            plt.plot(x_t, y_t, 'r.')
            plt.plot(x_t, w * x_t + b)
            plt.title("Linear regression")
            plt.xlabel("x")
            plt.ylabel('y')

            # Cost over time
            plt.subplot(223)
            plt.plot(range(len(e_s)), e_s)
            plt.title("Cost Over time...")
            plt.xlabel("no of examples")
            plt.ylabel("Cost")

            # 3-D
            plt.subplot(224, projection='3d')
            plt.plot(x_t, y_t, cp(x_t, y_t, w, b), '.')
            plt.title("Cost vs w vs b")
            plt.subplots_adjust(wspace=0.31, hspace=0.5)
            plt.xlabel("w")
            plt.ylabel('b')
            plt.gca().set_zlabel("Cost")
            plt.pause(1e-7)

mm()
plt.show()
        
