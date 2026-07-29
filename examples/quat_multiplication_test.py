import numpy as np

def quat_mul(q1, q2):  # Scalar Last
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def multiply_quat(q_1, q_2):  # from Kipras Aastropynaric
    q1 = q_2[0]
    q2 = q_2[1]
    q3 = q_2[2]
    q4 = q_2[3]
    Q2 = np.array([
        [q4, q3, -q2, q1],
        [-q3, q4, q1, q2],
        [q2, -q1, q4, q3],
        [-q1, -q2, -q3, q4]    
    ])
    q_comp = Q2 @ q_1
    return q_comp

def quat_mul_scalar_first(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # scalar part
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])

# Generate some random test quaternions
np.random.seed(0)
for i in range(5):
    q1 = np.random.randn(4)
    q2 = np.random.randn(4)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    
    prod1 = quat_mul(q1, q2)
    prod2 = multiply_quat(q1, q2)
    prod3 = quat_mul_scalar_first(q1, q2)
    
    print(f"Test {i+1}:")
    print("prod1 =", prod1)
    print("prod2 =", prod2)
    print("difference =", np.linalg.norm(prod1 - prod2))
    print("-"*40)
    print('Scalar 1st')
    print("difference =", np.linalg.norm(prod3 - prod2))
    print("-"*40)