import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="darkgrid")


# Generate some random data
# np.random.seed(1221345)
points = 500
lim_n, lim_p = -7, 6
x = np.linspace(lim_n, lim_p, points)
y = np.linspace(lim_n, lim_p, points)
random_noise = np.random.randn(points)  # Noise
x = x + random_noise
y = y - random_noise

# For reproducibility
np.random.seed(123345)

# Our own model
class LinearModel():
    def __init__(self, iters=1000, lr=1e-3, accumulate_grads=False) -> None:
        """
        Simple model simulating Wx + b
        """
        self.coef_ = np.random.random(1) # w
        self.intercept_ = np.random.random(1) # b
        self.iters = iters
        self.lr = lr
        self.dw = 0
        self.db = 0
        self.accumulate_grads = accumulate_grads
        

    def update_params(self):
        self.coef_ = self.coef_ - self.lr * self.dw
        self.intercept_ = self.intercept_ - self.lr * self.db

    def get_derivatives(self, *, x, y):
        """
        Considering cost function -> 1/(2*m) * sum((w*x + b - y)**2)
        """
        if not self.accumulate_grads:
            self.dw, self.db = 0, 0
        temp = (self.coef_ * x + self.intercept_ - y) / x.shape[0]
        self.dw += sum(temp * x)
        self.db += sum(temp)


    def cost(self, x, y, custom=False, w=0, b=0):
        """
        returns mse
        """
        if not custom:
            return sum((1/(2*x.shape[0]) * (self.coef_ * x + self.intercept_ - y)**2))

        return sum((1/(2*x.shape[0]) * (w* x + b - y)**2))
        


    def fit(self, x, y):
        """
        This is for simple Linear Regression
        Args:
            x: data of shape (m, 1)
            y: data of shape (m, 1)

        """
        for _ in range(self.iters):
            self.get_derivatives(x=x, y=y)
            self.update_params()

    def values_for_cost(self, x, y, lims=(-2, 2, -2, 2), points=200):
        ws = np.linspace(lims[0], lims[1], points)
        bs = np.linspace(lims[2], lims[3], points)
        ws, bs = np.meshgrid(ws, bs)
        ws, bs = np.ravel(ws), np.ravel(bs)
        er = np.zeros((points * points, ))
        for i in range(points * points):
            e = self.cost(x, y, ws[i], bs[i])
            er[i] = e
        return ws, bs, er



    def predict(self,):
        """
        Not needed rn...
        """
        pass

model = LinearModel(iters=1, lr=1e-5)
iterations = 10000
er = []
# ws, bs = [], []
def mm():
    """
    Plot and predict data on the go...
    """
    for i in range(1, iterations):
        model.fit(x, y)

        # stats
        er.append(model.cost(x, y))
        # ws.append(model.coef_[0])
        # bs.append(model.intercept_[0])

        if i%500 == 0:
            plt.clf()
            w, b = model.coef_, model.intercept_

            # Linear regression line
            plt.subplot(131)    
            plt.xlim(lim_n-2, lim_p+2)
            plt.ylim(lim_n-2, lim_p+2)
            plt.plot(x, y, 'r.')
            plt.plot(x, w * x + b)
            plt.title("Linear regression")
            plt.xlabel("x")
            plt.ylabel('y')

            # Cost over time
            plt.subplot(132)
            plt.plot(range(len(er)), er)
            plt.title("Cost Over time...")
            plt.xlabel("no of examples")
            plt.ylabel("Cost")

            # # 3-D
            plt.subplot(133, projection='3d')
            ws, bs, c = model.values_for_cost(x, y, (lim_n-1, lim_p+1, lim_n-1, lim_p+1), points=50)
            # plt.plot(ws, bs, er, '.')
            plt.title("Cost vs w vs b")
            plt.plot(ws, bs, c, '.')
            plt.subplots_adjust(wspace=0.31, hspace=0.5)
            plt.xlabel("w")
            plt.ylabel('b')
            plt.gca().set_zlabel("Cost")
            plt.pause(1e-7)

mm()
plt.show()
        
