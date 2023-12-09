import numpy as np
from scipy.optimize import root


def compute_lambda(density, w, u, d, Ke, method="colebrook"):
    """
    计算输气管道水力摩阻系数
    :param density: 气体密度，kg/m3
    :param w: 气体流速 m/s
    :param u: 气体粘度，Pa.s
    :param d: 圆管管径，m
    :param Ke: 管壁粗糙度，m
    :param method: 摩阻计算公式
    :return:
    """
    Re = density * w * d / u

    def colebrook(lambd):
        left = 1 / np.sqrt(lambd)
        right = -2 * np.log10(Ke / (3.7 * d) + 2.51 / (Re * np.sqrt(lambd)))
        return left - right

    if method == "colebrook":
        lambd = root(colebrook, np.array([0.01]))["x"][0]
    else:
        lambd = 0.01

    return lambd
