import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# %matplotlib inline

def uniform(size):
    x = np.linspace(0,1,size)
    return x.reshape(size,1)

def create_data(size):
    x = uniform(size)
    np.random.seed(42)
    y = sin_fun(x)+np.random.normal(scale=0.15, size=x.shape)
    return x,y

def sin_fun(x):
    return np.sin(2*np.pi*x)

def fitting(X_train, y_train, X_test=None, y_test=None):
    fig = plt.figure(figsize=(12, 8))
    for i, order in enumerate([0, 1, 3, 9]):
        plt.subplot(2, 2, i + 1)

        poly = PolynomialFeatures(order)
        X_train_ploy = poly.fit_transform(X_train)
        X_test_ploy = poly.fit_transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_ploy, y_train)
        y_pred = lr.predict(X_test_ploy)

        plt.scatter(X_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(X_test, y_pred, c="r", label="fitting")
        # plt.plot(X_test, y_test, c="g", label="$\sin(2\pi x)$")
        plt.title("M={}".format(order))
        plt.legend()
    plt.show()

def regularization(X_train, y_train, X_test, y_test):
    fig = plt.figure(figsize=(12, 4))
    for i, N in enumerate([10, 100]):
        X_train, y_train = create_data(N)

        plt.subplot(1, 2, i + 1)

        poly = PolynomialFeatures(9)
        X_train_ploy = poly.fit_transform(X_train)
        X_test_ploy = poly.fit_transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_ploy, y_train)
        y_pred = lr.predict(X_test_ploy)

        plt.scatter(X_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(X_test, y_pred, c="r", label="fitting")
        plt.plot(X_test, y_test, c="g", label="$\sin(2\pi x)$")
        plt.title("N={}".format(N))
        plt.legend()
    plt.show()

def ridge_Regression(X_train, y_train, X_test, y_test):
    fig = plt.figure(figsize=(12, 4))
    for i, lamb in enumerate([0.001, 1]):
        X_train, y_train = create_data(10)

        plt.subplot(1, 2, i + 1)

        poly = PolynomialFeatures(9)
        X_train_ploy = poly.fit_transform(X_train)
        X_test_ploy = poly.fit_transform(X_test)

        lr = Ridge(alpha=(lamb / 2))
        lr.fit(X_train_ploy, y_train)
        y_pred = lr.predict(X_test_ploy)

        plt.scatter(X_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(X_test, y_pred, c="r", label="fitting")
        plt.plot(X_test, y_test, c="g", label="$\sin(2\pi x)$")
        plt.title("$\lambda$={}".format(lamb))
        plt.legend()
    plt.show()

def example():
    X_train,y_train = create_data(10)
    X_test = uniform(100)
    y_test = sin_fun(X_test)

    plt.scatter(X_train,y_train,facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(X_test,y_test,c="g",label="$\sin(2\pi x)$")
    plt.ylabel("y",size=20)
    plt.xlabel("x",size=20)
    plt.legend()
    plt.show()

def coeffab(iso, poly):
    iso_list = [1600, 3200, 6400, 12800, 25600]
    a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
    b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
    X_train = np.asarray(iso_list, dtype=np.float32).reshape(-1,1) / 51200
    a_train = np.asarray(a_list, dtype=np.float32)
    b_train = np.asarray(b_list, dtype=np.float32)
    X_test = np.asarray([iso], dtype=np.float32).reshape(-1, 1) / 51200
    order = 3
    poly = PolynomialFeatures(order)
    X_train_ploy = poly.fit_transform(X_train)
    X_test_ploy = poly.fit_transform(X_test)

    lr_a = LinearRegression()
    lr_a.fit(X_train_ploy, a_train)
    a = lr_a.predict(X_test_ploy)
    lr_b = LinearRegression()
    lr_b.fit(X_train_ploy, b_train)
    b = lr_b.predict(X_test_ploy)
    # print('end')
    return a[0], b[0]

def get_fit_curve(X_train, Y_train, order = 3):
    poly = PolynomialFeatures(order)
    X_train_ploy = poly.fit_transform(X_train)
    lr_fit = LinearRegression()
    lr_fit.fit(X_train_ploy, Y_train)
    return lr_fit


def main():
    # example()
    coeffab(211)

if __name__ == '__main__':
    main()

'''
多项式曲线拟合 https://zhuanlan.zhihu.com/p/53056358
'''