import numpy as np
import matplotlib.pyplot as plt




# Generate two hemispheres of the 4D quaternion unit sphere projected into 3D (ignore q0)
phi = np.linspace(0, np.pi, 50)
theta = np.linspace(0, 2*np.pi, 50)
phi, theta = np.meshgrid(phi, theta)

# Construct quaternion points q = [q0, q1, q2, q3], project only (q1,q2,q3)
q0 = np.cos(phi/2)
q1 = np.sin(phi/2) * np.cos(theta)
q2 = np.sin(phi/2) * np.sin(theta)
q3 = np.zeros_like(q0)

# Plot two points: q and -q
q = np.array([0.9, 0.3, 0.1, -0.2])
q = q / np.linalg.norm(q)
q_neg = -q

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere (hemisphere projection)
ax.plot_surface(q1, q2, q0, alpha=0.5, color='blue')
ax.plot_surface(-q1, -q2, -q0, alpha=0.5, color='red')

# Plot the quaternions (ignoring q0 in visualization)
ax.scatter(q[1], q[2], q[0], color='blue', s=100, label="q")
ax.scatter(q_neg[1], q_neg[2], q_neg[0], color='red', s=100, label="-q")

# Draw a line connecting q and -q
ax.plot([q[1], q_neg[1]], [q[2], q_neg[2]], [q[0], q_neg[0]], 'k--')

# Labels
ax.set_xlabel("q1")
ax.set_ylabel("q2")
ax.set_zlabel("q0")
ax.set_title("Quaternion Double Covering: q and -q represent the same rotation")
ax.legend()


# Create some example quaternion points (projected into 3D for visualization)
np.random.seed(0)
N = 200
points = np.random.randn(N, 3)
points /= np.linalg.norm(points, axis=1, keepdims=True)  # normalize to unit sphere

# Reference vector (like previous quaternion)
ref = np.array([1, 0, 0])

# Compute dot product to split hemispheres
dots = points @ ref
same_side = points[dots >= 0]
opposite_side = points[dots < 0]

# Plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection="3d")

# Unit sphere wireframe
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.3)

# Plot points: same hemisphere (blue), opposite hemisphere (red)
ax.scatter(same_side[:,0], same_side[:,1], same_side[:,2], color="blue", label="Same Hemisphere")
ax.scatter(opposite_side[:,0], opposite_side[:,1], opposite_side[:,2], color="red", label="Opposite Hemisphere")

# Reference vector
ax.quiver(0,0,0, ref[0],ref[1],ref[2], color="green", linewidth=2, label="Reference Quaternion")

ax.set_title("Analogy: Hemisphere Split by Dot Product Check")
ax.legend()
ax.set_box_aspect([1,1,1])


plt.show()

#%%
def normalize_quaternion_with_history(q_new, q_ref):
    # Standard shortest path
    if np.dot(q_new, q_ref) < 0:
        q_new = -q_new

    # At 180°: dot ≈ 0 → use axis alignment with previous
    if abs(np.dot(q_new, q_ref)) < 1e-4:
        # Extract axis from q_ref (previous)
        axis_ref = q_ref[1:] / np.linalg.norm(q_ref[1:]) if q_ref[0] < 0.9 else q_ref[1:]

        # Extract axis from q_new
        axis_new = q_new[1:] / np.linalg.norm(q_new[1:]) if q_new[0] < 0.9 else q_new[1:]

        # If axes are opposite, flip q_new
        if np.dot(axis_ref, axis_new) < 0:
            q_new = -q_new

    return q_new / np.linalg.norm(q_new)