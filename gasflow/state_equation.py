import os
import numpy as np
import sympy as sp
from scipy.optimize import root

properties = np.array(
    [[190.69, 46.04, 10.05, 0.013, 16.042, 2.39359, -0.002218007, 5.74022e-06, -3.727905e-09, 8.549685000000001e-13],
     [305.38, 48.8, 6.7566, 0.1018, 30.068, 1.10899, -0.000188512, 3.96558e-06, -3.140209e-09, 8.008189e-13],
     [369.89, 42.5, 4.9994, 0.157, 40.94, 0.72265, 0.0007087160000000001, 2.923895e-06, -2.615017e-09,
      7.000448000000001e-13],
     [425.18, 37.97, 3.9213, 0.197, 58.12, 0.4127, 0.002028601, 7.02953e-07, -1.025871e-09, 2.883394e-13],
     [304.09, 73.8, 10.638, 0.21, 44.01, 0.47911, 0.000762195, -3.59392e-07, 8.47438e-11, -5.7752000000000004e-15],
     [126.15, 112.98, 11.098999999999998, 0.035, 28.016, 1.06849, -0.000134096, 2.15569e-07, -7.86319e-11,
      6.985100000000001e-15],
     [np.nan, 12.97, 20.0, 0.0, 2.015, 13.39616, 0.002960131, -3.980745e-06, 2.661667e-09, -6.099862e-13]])
Ai = np.array(
    [0.44369, 1.28438, 0.356306, 0.544979, 0.528629, 0.484011, 0.0705233, 0.504087, 0.0307452, 0.0732828, 0.00645])
Bi = np.array(
    [0.115449, -0.920731, 1.70871, -0.270896, 0.349261, 0.75413, -0.044448, 1.32245, 0.179433, 0.463492, 0.022143])
k = np.array([[0.0, 0.01, 0.023, 0.031, 0.025, 0.05, 0.0],
              [0.01, 0.0, 0.0031, 0.0045, 0.07, 0.048, 0.0],
              [0.023, 0.0031, 0.0, 0.01, 0.01, 0.045, 0.0],
              [0.031, 0.0045, 0.01, 0.0, 0.12, 0.05, 0.0],
              [0.05, 0.048, 0.045, 0.05, 0.0, 0.0, 0.0],
              [0.025, 0.07, 0.1, 0.12, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


class BWRSCalculator:
    def __init__(self, components, P, T):
        """
        BWRS方程
        :param components: 字典，天然气组分百分比
        :param P: 压力， kPa
        :param T: 温度，K
        """
        # 物性条件
        self.P = P
        self.T = T
        # 组分条件转换
        gas_to_index = {"CH4": 0, "C2H6": 1, "C3H8": 2, "n-C4H10": 3, "CO2": 4, "N2": 5, "H2": 6}
        try:
            self.components_index = np.array([gas_to_index[gas] for gas in components.keys()])
            self.frac = np.array(list(components.values()))
            if np.sum(self.frac) != 1.:
                print("请检查气体组分，其百分比和必须为1！")
                exit()
        except KeyError as e:
            print(f"抱歉！气体组分{e}暂时不支持!")
            print(f"目前支持{list(gas_to_index.keys())}!")
            exit()
        # 加载常数
        self.Rm = 8314.510  # 通用气体常数，J/kmol.K
        # self.properties = np.load(os.path.join(data_path, "pure_gas.npy"), allow_pickle=True).astype(
        #     np.float64)  # 纯物质参数
        # self.Ai = np.load(os.path.join(data_path, "Ai.npy"), allow_pickle=True).astype(np.float64)  # 参数Ai
        # self.Bi = np.load(os.path.join(data_path, "Bi.npy"), allow_pickle=True).astype(np.float64)  # 参数Bi
        # self.k = np.load(os.path.join(data_path, "k.npy"), allow_pickle=True).astype(np.float64)  # 交互系数

        self.properties = properties
        self.Ai = Ai
        self.Bi = Bi
        self.k = k

        # BWRS方程计算结果
        self.parameters = None  # 11个参数
        self.M = None  # 摩尔质量
        self.density = None  # 密度
        self.dP_dT, self.dP_dp = None, None  # BWRS方程导数数表达式

    def run(self):
        """计算物性参数"""
        # 计算单物性参数
        Tci = self.properties[self.components_index, 0]  # 临界温度
        pci = self.properties[self.components_index, 2]  # 临界密度
        wi = self.properties[self.components_index, 3]  # 偏心因子
        Ms = self.properties[self.components_index, 4]  # 摩尔质量
        # 特例：H2的Tci与温度有关
        if self.T > 255.35:
            Tci = np.nan_to_num(Tci, nan=47.05)
        elif 199.85 < self.T < 255.35:
            Tci = np.nan_to_num(Tci, nan=35.95)
        else:
            Tci = np.nan_to_num(Tci, nan=27.55)
        B0 = (self.Ai[0] + self.Bi[0] * wi) / pci
        A0 = (self.Ai[1] + self.Bi[1] * wi) * self.Rm * Tci / pci
        C0 = (self.Ai[2] + self.Bi[2] * wi) * self.Rm * Tci ** 3 / pci
        y = (self.Ai[3] + self.Bi[3] * wi) / pci ** 2  # y为Gamma
        b = (self.Ai[4] + self.Bi[4] * wi) / pci ** 2
        a = (self.Ai[5] + self.Bi[5] * wi) * self.Rm * Tci / pci ** 2
        a0 = (self.Ai[6] + self.Bi[6] * wi) / pci ** 3  # a0为alpha
        c = (self.Ai[7] + self.Bi[7] * wi) * self.Rm * Tci ** 3 / pci ** 2
        D0 = (self.Ai[8] + self.Bi[8] * wi) * self.Rm * Tci ** 4 / pci
        d = (self.Ai[9] + self.Bi[9] * wi) * self.Rm * Tci ** 2 / pci ** 2
        E0 = (self.Ai[10] + self.Bi[10] * wi * np.exp(-3.8 * wi)) * self.Rm * Tci ** 5 / pci

        k = self.k[self.components_index][:, self.components_index]

        # 计算混合气体的11个参数
        M = np.dot(self.frac, Ms)  # 摩尔质量 kg/kmol
        A0_final = np.sum(self.frac[:, np.newaxis] * self.frac * np.sqrt(A0[:, np.newaxis] * A0) * (1 - k)) / M ** 2
        B0_final = np.dot(self.frac, B0) / M
        C0_final = np.sum(
            self.frac[:, np.newaxis] * self.frac * np.sqrt(C0[:, np.newaxis] * C0) * (1 - k) ** 3) / M ** 2
        D0_final = np.sum(
            self.frac[:, np.newaxis] * self.frac * np.sqrt(D0[:, np.newaxis] * D0) * (1 - k) ** 4) / M ** 2
        E0_final = np.sum(
            self.frac[:, np.newaxis] * self.frac * np.sqrt(E0[:, np.newaxis] * E0) * (1 - k) ** 5) / M ** 2
        a_final = np.dot(self.frac, np.power(a, 1 / 3)) ** 3 / M ** 3
        b_final = np.dot(self.frac, np.power(b, 1 / 3)) ** 3 / M ** 2
        c_final = np.dot(self.frac, np.power(c, 1 / 3)) ** 3 / M ** 3
        d_final = np.dot(self.frac, np.power(d, 1 / 3)) ** 3 / M ** 3
        a0_final = np.dot(self.frac, np.power(a0, 1 / 3)) ** 3 / M ** 3
        y_final = np.dot(self.frac, np.power(y, 1 / 2)) ** 2 / M ** 2

        self.parameters = np.array([A0_final, B0_final, C0_final, D0_final, E0_final, a_final, b_final, c_final,
                                    d_final, y_final, a0_final])
        self.M = M

    def compute_density(self):
        if self.parameters is None:
            self.run()

        A, B, C, D, E, a, b, c, d, y, a0 = self.parameters
        R = self.Rm / self.M
        p_est = self.P / (R * self.T)  # 用理想气体状态方程计算密度作为初始预测值 => kg/m3

        # 定义方程表达式函数
        def equation(density):
            equation1 = density * R * self.T
            equation2 = (B * R * self.T - A - C / self.T ** 2 + D / self.T ** 3 - E / self.T ** 4) * density ** 2
            equation3 = (b * R * self.T - a - d / self.T) * density ** 3
            equation4 = a0 * (a + d / self.T) * density ** 6
            equation5 = c * density ** 3 / self.T ** 2 * (1 + y * density ** 2) * np.exp(-y * density ** 2)

            return self.P - (equation1 + equation2 + equation3 + equation4 + equation5)

        density = root(equation, np.array([p_est]))['x'][0]  # kg/m3
        Z = self.P / (density * R * self.T)  # 压缩因子

        self.density = density

        return density, Z

    def BWRS_equation(self, p, T):
        """用于sympy求导求积分的BWRS"""
        A, B, C, D, E, a, b, c, d, y, a0 = self.parameters
        R = self.Rm / self.M

        equation1 = p * R * T
        equation2 = (B * R * T - A - C / T ** 2 + D / T ** 3 - E / T ** 4) * p ** 2
        equation3 = (b * R * T - a - d / T) * p ** 3
        equation4 = a0 * (a + d / T) * p ** 6
        equation5 = c * p ** 3 / T ** 2 * (1 + y * p ** 2) * sp.exp(-y * p ** 2)
        P = equation1 + equation2 + equation3 + equation4 + equation5

        return P

    def compute_heat_capacity(self):
        if self.parameters is None:
            self.run()
        if self.density is None:
            self.compute_density()

        # 计算低压状态下的比热容
        coef = self.properties[self.components_index, 5:]
        cp0_array = (coef[:, 0] + 2 * coef[:, 1] * self.T + 3 * coef[:, 2] * self.T ** 2 + 4 * coef[:, 3] * self.T ** 3
                     + 5 * coef[:, 4] * self.T ** 4) * 1000
        cp0 = np.dot(self.frac, cp0_array)
        # 低压下近似理想气体，由迈耶公式
        cv0 = cp0 - self.Rm / self.M
        # 由热力学关系求实际气体的定容比热容
        p, T = sp.symbols("p T")
        P = self.BWRS_equation(p, T)
        dP_dT = sp.diff(P, T)  # Pa/K,
        dP_dp = sp.diff(P, p)  # Pa/kg/m3
        dP_dT2 = sp.diff(dP_dT, T)
        func = - T / p ** 2 * dP_dT2
        cv = cv0 + sp.integrate(func, (p, 0, self.density)).evalf(subs={"T": self.T, "p": self.density})
        # 再转化为定压比热容
        dPdT = dP_dT.evalf(subs={"T": self.T, "p": self.density})
        dPdp = dP_dp.evalf(subs={"T": self.T, "p": self.density})
        cp = cv + self.T / self.density ** 2 * dPdT ** 2 / dPdp

        self.dP_dT, self.dP_dp = dP_dT, dP_dp

        return cp

    def compute_derivative(self):
        if self.parameters is None:
            self.run()
        if self.density is None:
            self.compute_density()
        if self.dP_dT is None or self.dP_dp is None:
            p, T = sp.symbols("p T")
            P = self.BWRS_equation(p, T)
            dP_dT = sp.diff(P, T)
            dP_dp = sp.diff(P, p)
            self.dP_dT, self.dP_dp = dP_dT, dP_dp

        dPdT = self.dP_dT.evalf(subs={"T": self.T, "p": self.density})
        dPdp = self.dP_dp.evalf(subs={"T": self.T, "p": self.density})

        return dPdT, dPdp


if __name__ == '__main__':
    components = {"CH4": 0.995, "C2H6": 0.005}
    pressure = 10e6
    temperature = 293.15
    calculator = BWRSCalculator(components, pressure, temperature)
    dPdT, dPdp = calculator.compute_derivative()
    print(1 / dPdT)
