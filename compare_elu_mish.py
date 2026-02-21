import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))  # x * tanh(softplus(x))

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


x = np.linspace(-6, 6, 1000)

y_mish = mish(x)
y_elu = elu(x)

# Plot
plt.figure(figsize=(6, 4), dpi=150)

plt.plot(x, y_mish, label="Mish", linewidth=2)
plt.plot(x, y_elu, label=r"ELU ($\alpha=1.0$)", linewidth=2)

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlabel("Input")
plt.ylabel("Activation Output")
plt.title("Comparison of Mish and ELU Activation Functions")

plt.legend()
plt.grid(True, linestyle=":", linewidth=0.5)
plt.tight_layout()

plt.savefig('elu_mish.png')