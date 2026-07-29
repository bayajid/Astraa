import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def normalize(v):
    return v / np.linalg.norm(v)


def triad(v1, v2, b1, b2):

    # reference triad
    t1 = normalize(v1)
    t2 = normalize(np.cross(v1, v2))
    t3 = np.cross(t1, t2)

    # body triad
    b1n = normalize(b1)
    b2n = normalize(np.cross(b1, b2))
    b3n = np.cross(b1n, b2n)

    T = np.vstack((t1, t2, t3)).T
    B = np.vstack((b1n, b2n, b3n)).T

    Rmat = B @ T.T

    return R.from_matrix(Rmat)


# -----------------------------------
# Simulation setup
# -----------------------------------

true_rot = R.from_euler('xyz', [20, 10, 30], degrees=True)

angles_deg = np.linspace(60,120,50)

# Vector error magnitudes (radians)
delta_list = [0.01, 0.05, 0.1]   # 10 mrad, 50 mrad, 100 mrad


# -----------------------------------
# Plot 1: Attitude error vs separation
# -----------------------------------

plt.figure(figsize=(8,5))

for delta in delta_list:

    att_errors = []

    for a in angles_deg:

        alpha = np.radians(a)

        v1 = np.array([1,0,0])
        v2 = normalize(np.array([np.cos(alpha), np.sin(alpha),0]))

        b1 = true_rot.apply(v1)
        b2 = true_rot.apply(v2)

        q_nominal = triad(v1, v2, b1, b2)

        axis = normalize(np.array([0,0,1]))
        error_rot = R.from_rotvec(axis * delta)

        v1_bad = error_rot.apply(v1)

        q_bad = triad(v1_bad, v2, b1, b2)

        dq = q_bad * q_nominal.inv()

        err = np.linalg.norm(dq.as_rotvec())

        att_errors.append(err)

    att_errors = np.array(att_errors) * 1000  # convert to mrad
    delta_mrad = delta * 1000

    plt.plot(
        angles_deg,
        att_errors,
        linewidth=3,
        label=f"vector error = {delta_mrad:.0f} mrad"
    )

plt.xlabel("Angle between reference vectors (deg)")
plt.ylabel("Attitude error (mrad)")
plt.title("TRIAD Sensitivity to Reference Vector Error")
plt.grid(True)
plt.legend(title="Input vector error")

plt.tight_layout()


# -----------------------------------
# Plot 2: Monte Carlo distribution
# -----------------------------------

delta = 0.1   # choose a value to analyze (100 mrad)

N = 500
errors = []

v1 = np.array([1,0,0])
v2 = np.array([0,1,0])

for i in range(N):

    axis = normalize(np.random.randn(3))
    error_rot = R.from_rotvec(axis * delta)

    b1 = true_rot.apply(v1)
    b2 = true_rot.apply(v2)

    q_nominal = triad(v1, v2, b1, b2)

    v1_bad = error_rot.apply(v1)

    q_bad = triad(v1_bad, v2, b1, b2)

    dq = q_bad * q_nominal.inv()

    errors.append(np.linalg.norm(dq.as_rotvec()))

errors = np.array(errors) * 1000  # mrad


plt.figure(figsize=(8,5))

plt.hist(errors, bins=30)

plt.xlabel("Attitude error (mrad)")
plt.ylabel("Count")
plt.title(f"Monte Carlo TRIAD Error (vector error = {delta*1000:.0f} mrad)")

plt.grid(True)

plt.tight_layout()

plt.show()