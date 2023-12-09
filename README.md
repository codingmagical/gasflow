# gaspy
A python library for calculating natural gas properties and gas pipeline simulation
# 简单案例
import scienceplots
import matplotlib.pyplot as plt
from gasflow.gas_pipe import GasPipe
from gasflow.state_equation import BWRSCalculator

# 1.计算气体物理性质
components = {"CH4": 1}
calculator = BWRSCalculator(components, 101325, 273.15)
density, Z = calculator.compute_density()  # 密度，kg/m3 压缩因子
print(density)
cp = calculator.compute_heat_capacity()  # 定压比热容， J/kg.K
print(cp)

# 2.简单输气管道仿真
params = {
    "gas": {"CH4": 0.995, "C2H6": 0.005},  # 气体组分
    "Q": 34.8e8 / 350 / 24 / 3600,  # 标况下流量m3/s
    "L": 6000,  # 管长，m
    "D": 1.219,  # 管外经，m
    "d": 1.219 - 2 * 0.018,  # 管内径，m
    "PQ": 8e6,  # 起点压力
    "TQ": 273.15 + 50,  # 起点温度，K
    "T0": 273.15,  # 平均地温
    "K": 1.2,  # 总传热系数 W/m2.K
    "Ke": 0.02 * 10 ** -3  # 管壁粗糙度 m
}
pipe = GasPipe(**params)
distance, P_array, T_array, w_array = pipe.Rk4_steady_simulate(1000)

with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax1 = plt.subplots()
    ax1.plot(distance, P_array, label="pressure", color="red")
    ax1.set(xlabel='Distance(km)')
    ax1.set(ylabel='pressure')
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (K)')
    ax2.plot(distance, T_array, label="Temperature", color="blue")
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.savefig("fig1.png", dpi=300)
    plt.show()
