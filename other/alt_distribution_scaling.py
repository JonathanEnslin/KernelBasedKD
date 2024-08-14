import numpy as np
import matplotlib.pyplot as plt

def softmax(x, temperature=1.0):
    """Compute the softmax with temperature"""
    exp_x = np.exp(x / temperature)
    return exp_x / np.sum(exp_x)

def scaled_softmax(x, temperature=1.0):
    """Compute the modified softmax using the absolute value of logits, with temperature"""
    abs_x = np.abs(x)
    exp_x = np.exp(abs_x / temperature)
    softmax_vals = exp_x / np.sum(exp_x)
    return np.sign(x) * softmax_vals

def tanh_scaling(x, scale=1.0):
    """Apply tanh-based scaling to the logits"""
    return np.tanh(scale * x)

# Generate some sample data
np.random.seed(42)
logits = np.random.randn(5)  # 5 random logits
temperatures = [1.0, 0.5, 2.0, 4.0]  # Different temperatures for comparison
scale_factors = [0.5, 1.0, 2.0, 4.0]  # Different scales for tanh

# Apply normal softmax
softmax_normal = softmax(logits)

# Apply softmax with different temperatures
softmax_temp_low = softmax(logits, temperature=temperatures[1])
softmax_temp_high = softmax(logits, temperature=temperatures[2])
softmax_temp_very_high = softmax(logits, temperature=temperatures[3])

# Apply modified softmax
scaled_softmax_normal = scaled_softmax(logits)

# Apply modified softmax with different temperatures
scaled_softmax_temp_low = scaled_softmax(logits, temperature=temperatures[1])
scaled_softmax_temp_high = scaled_softmax(logits, temperature=temperatures[2])
scaled_softmax_temp_very_high = scaled_softmax(logits, temperature=temperatures[3])

# Apply tanh scaling with different scales
tanh_scaled_low = tanh_scaling(logits, scale=scale_factors[0])
tanh_scaled_normal = tanh_scaling(logits, scale=scale_factors[1])
tanh_scaled_high = tanh_scaling(logits, scale=scale_factors[2])
tanh_scaled_very_high = tanh_scaling(logits, scale=scale_factors[3])

# Plotting the results
plt.figure(figsize=(18, 12))

# Plot original logits
plt.subplot(2, 3, 1)
plt.plot(logits, marker='o', linestyle='-', color='b')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Original Logits")
plt.xlabel("Index")
plt.ylabel("Logit Value")

# Plot softmax output with normal temperature
plt.subplot(2, 3, 2)
plt.plot(softmax_normal, marker='o', linestyle='-', color='b')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Softmax (Temperature = 1.0)")
plt.xlabel("Index")
plt.ylabel("Probability")

# Plot softmax output with lower, higher, and very high temperatures
plt.subplot(2, 3, 3)
plt.plot(softmax_temp_low, marker='o', linestyle='-', color='r', label=f'Temperature = {temperatures[1]}')
plt.plot(softmax_temp_high, marker='o', linestyle='-', color='g', label=f'Temperature = {temperatures[2]}')
plt.plot(softmax_temp_very_high, marker='o', linestyle='-', color='purple', label=f'Temperature = {temperatures[3]}')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Softmax with Temperature")
plt.xlabel("Index")
plt.ylabel("Probability")
plt.legend()

# Plot scaled softmax output
plt.subplot(2, 3, 4)
plt.plot(scaled_softmax_normal, marker='o', linestyle='-', color='b')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Scaled Softmax (Temperature = 1.0)")
plt.xlabel("Index")
plt.ylabel("Scaled Probability")

# Plot scaled softmax output with lower, higher, and very high temperatures
plt.subplot(2, 3, 5)
plt.plot(scaled_softmax_temp_low, marker='o', linestyle='-', color='r', label=f'Temperature = {temperatures[1]}')
plt.plot(scaled_softmax_temp_high, marker='o', linestyle='-', color='g', label=f'Temperature = {temperatures[2]}')
plt.plot(scaled_softmax_temp_very_high, marker='o', linestyle='-', color='purple', label=f'Temperature = {temperatures[3]}')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Scaled Softmax with Temperature")
plt.xlabel("Index")
plt.ylabel("Scaled Probability")
plt.legend()

# Plot tanh scaled output with different scales
plt.subplot(2, 3, 6)
plt.plot(tanh_scaled_low, marker='o', linestyle='-', color='r', label=f'Scale = {scale_factors[0]}')
plt.plot(tanh_scaled_normal, marker='o', linestyle='-', color='b', label=f'Scale = {scale_factors[1]}')
plt.plot(tanh_scaled_high, marker='o', linestyle='-', color='g', label=f'Scale = {scale_factors[2]}')
plt.plot(tanh_scaled_very_high, marker='o', linestyle='-', color='purple', label=f'Scale = {scale_factors[3]}')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Tanh Scaling with Different Scales")
plt.xlabel("Index")
plt.ylabel("Scaled Value")
plt.legend()

plt.tight_layout()
plt.show()


# generate dat as a random array of 6 elements in range [-1, 1]
dat = np.random.rand(7) * 2 - 1

# Pass through sign func to get signs
signs = np.sign(dat)

dat = np.abs(dat)

print(dat)

# Apply exponential scaling with different scales
exp_scaled_low = np.power(dat, 0.5)
exp_scaled_normal = np.power(dat, 1.0)
exp_scaled_high = np.power(dat, 2.0)
exp_scaled_very_high = np.power(dat, 4.0)

# Multiply by the signs to restore original direction
exp_scaled_low *= signs
exp_scaled_normal *= signs
exp_scaled_high *= signs
exp_scaled_very_high *= signs

# Plotting the exponential scaling results
plt.figure(figsize=(18, 6))

plt.subplot(2, 2, 1)
plt.plot(dat * signs, marker='o', linestyle='-', color='b', label='Original Data')
plt.plot(exp_scaled_low, marker='o', linestyle='-', color='r', label='Scale = 0.5')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Exponential Scaling (Scale = 0.5)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(dat * signs, marker='o', linestyle='-', color='b', label='Original Data')
plt.plot(exp_scaled_normal, marker='o', linestyle='-', color='g', label='Scale = 1.0')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Exponential Scaling (Scale = 1.0)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(dat * signs, marker='o', linestyle='-', color='b', label='Original Data')
plt.plot(exp_scaled_high, marker='o', linestyle='-', color='purple', label='Scale = 2.0')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Exponential Scaling (Scale = 2.0)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(dat * signs, marker='o', linestyle='-', color='b', label='Original Data')
plt.plot(exp_scaled_very_high, marker='o', linestyle='-', color='orange', label='Scale = 4.0')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Exponential Scaling (Scale = 4.0)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()

