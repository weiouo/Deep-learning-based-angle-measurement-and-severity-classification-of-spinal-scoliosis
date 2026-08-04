import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

# 一組乾淨、理想的點
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, -1, 3, -0.5, 2])


x_dense = np.linspace(0, 5, 200)

# Polynomial Fit (degree 3)
poly_coeffs = np.polyfit(x, y, 3)
poly_y = np.polyval(poly_coeffs, x_dense)

# Cubic Spline
cs = CubicSpline(x, y)
spline_y = cs(x_dense)

# LOESS
x_dense = np.linspace(min(x), max(x), 200)
y_interp = np.interp(x_dense, x, y)  # 插值填補，模擬真實資料點
loess_result = lowess(y_interp, x_dense, frac=0.4)
loess_x, loess_y = loess_result[:, 0], loess_result[:, 1]


# Univariate Spline (smooth factor可調，s=0代表強制通過所有點)
uspline = UnivariateSpline(x, y, s=0.5)
uspline_y = uspline(x_dense)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Sample Points", color="black")
#plt.plot(x_dense, poly_y, label="Polynomial (deg=3)", linestyle='-')
#plt.plot(x_dense, spline_y, label="Cubic Spline", linestyle='-')
plt.plot(loess_x, loess_y, label="LOWESS", linestyle='-')
#plt.plot(x_dense, uspline_y, label="Univariate Spline", linestyle='-')
plt.title("Conceptual Comparison of Curve Fitting Methods")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
