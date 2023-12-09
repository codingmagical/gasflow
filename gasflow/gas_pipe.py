import numpy as np
from tqdm import trange
from scipy.linalg import solve
from gasflow.util import compute_lambda
from gasflow.state_equation import BWRSCalculator


class GasPipe:
    def __init__(self, gas, Q, L, D, d, PQ, TQ, T0, K, Ke):
        self.gas = gas  # 管内气体组分
        # 几何参数
        self.L = L  # 管长，m
        self.D = D  # 管外经，m
        self.d = d  # 管内经，m
        self.A = np.pi * (self.d / 2) ** 2  # 管道截面积，m2
        # 水力参数
        self.Ke = Ke  # 管壁粗糙度，m
        self.Q = Q  # 管道起点实际状态下的流量
        self.PQ = PQ  # 起点压力(表)，Pa
        # 热力参数
        self.K = K  # 总传热系数，W/m2.K
        self.T0 = T0  # 平均地温
        self.TQ = TQ  # 起点温度，T

    def compute_params(self, P, T, w):
        actual_calculator = BWRSCalculator(self.gas, P + 101325, T)  # 气体状态不断变化
        density, Z = actual_calculator.compute_density()
        dPdT, dPdp = actual_calculator.compute_derivative()
        dp_dT = -dPdT / dPdp  # kg/m3/K
        dp_dP = 1 / dPdp  # kg/m3/Pa
        lambd = compute_lambda(density, w, 1.76e-5, self.d, self.Ke)  # 摩阻系数

        dHdT = actual_calculator.compute_heat_capacity()  # J/kg.K
        dHdP = 1 / density - (T / density ** 2) * (dPdT / dPdp)  # J/kg/Pa
        b1 = -self.K * np.pi * self.D * (T - self.T0) / (w * self.A * density)
        a31 = w / density * dp_dT
        a32 = w / density * dp_dP

        A = np.array([[dHdT, dHdP, w],
                      [0, 1 / density, w],
                      [a31, a32, 1]], dtype=np.float64)
        b = np.array([b1, - lambd * w ** 2 / (2 * self.D), 0], dtype=np.float64)
        dTdx, dPdx, dwdx = solve(A, b)

        return dTdx, dPdx, dwdx

    def Rk4_one_step(self, P, T, w, step):
        # 求一次导数值
        dTdx1, dPdx1, dwdx1 = self.compute_params(P, T, w)
        # 求二次导数值
        T1 = T + step / 2 * dTdx1
        P1 = P + step / 2 * dPdx1
        w1 = w + step / 2 * dwdx1
        dTdx2, dPdx2, dwdx2 = self.compute_params(P1, T1, w1)
        # 求三次导数值
        T2 = T + step / 2 * dTdx2
        P2 = P + step / 2 * dPdx2
        w2 = w + step / 2 * dwdx2
        dTdx3, dPdx3, dwdx3 = self.compute_params(P2, T2, w2)
        # 求四次导数值
        T3 = T + step * dTdx3
        P3 = P + step * dPdx3
        w3 = w + step * dwdx3
        dTdx4, dPdx4, dwdx4 = self.compute_params(P3, T3, w3)
        # 四节龙格塔库积分求各个变量增值
        Tz = T + step / 6 * (dTdx1 + 2 * dTdx2 + 2 * dTdx3 + dTdx4)
        Pz = P + step / 6 * (dPdx1 + 2 * dPdx2 + 2 * dPdx3 + dPdx4)
        wz = w + step / 6 * (dwdx1 + 2 * dwdx2 + 2 * dwdx3 + dwdx4)

        return Tz, Pz, wz

    def Rk4_steady_simulate(self, step):
        standard_calculator = BWRSCalculator(self.gas, 101325, 293.15)
        standard_density, _ = standard_calculator.compute_density()  # 标准状态密度
        actual_calculator = BWRSCalculator(self.gas, self.PQ + 101325, self.TQ)  # 初始位置实际状态
        actual_density, actual_Z = actual_calculator.compute_density()  # 初始位置实际密度，实际压缩因子
        G = standard_density * self.Q  # 质量流量
        Q_actual = G / actual_density  # 实际体积流量
        w = Q_actual / self.A  # 实际流速

        num_section = int(self.L / step)  # 分割的管段数目
        T_array, P_array, w_array = np.zeros(num_section), np.zeros(num_section), np.zeros(num_section)
        distance = np.zeros(num_section)
        for i in trange(num_section):
            if i == 0:
                Ti, pi, wi = self.Rk4_one_step(self.PQ, self.TQ, w, step)
                distance[0] = step / 1000
            else:
                Ti, pi, wi = self.Rk4_one_step(P_array[i - 1], T_array[i - 1], w_array[i - 1], step)
                distance[i] = distance[i - 1] + step / 1000
            T_array[i] = Ti
            P_array[i] = pi
            w_array[i] = wi
        if num_section * step < self.L:
            Ti, pi, wi = self.Rk4_one_step(P_array[-1], T_array[-1], w_array[-1], self.L - num_section * step)
            np.append(distance, distance[-1] + (self.L - num_section * step) / 1000)
            np.append(T_array, Ti)
            np.append(P_array, pi)
            np.append(w_array, wi)
        distance = np.insert(distance, 0, 0)
        T_array = np.insert(T_array, 0, self.TQ)
        P_array = np.insert(P_array, 0, self.PQ)
        w_array = np.insert(w_array, 0, w)

        return distance, P_array, T_array, w_array
