import numpy as np

def F(X):
    x1, x2, x3 = X
    return np.array([
        3*x1 - np.cos(x2*x3) - 1/2,
        x1*2 - 81*(x2 + 0.1)**2 + np.sin(x3) + 1.06,
        np.exp(-x1*x2) + 20*x3 + (10*np.pi - 3)/3
    ])

def JF(X):
    x1, x2, x3 = X
    return np.array([
        [3, x3*np.sin(x2*x3), x2*np.sin(x2*x3)],
        [2*x1, -162*(x2 + 0.1), np.cos(x3)],
        [-x2*np.exp(-x1*x2), -x1*np.exp(-x1*x2), 20]
    ])

def newton_Rn(F, JF, x0, eps=1e-8, max_iter=50):
    x = x0.astype(float)
    for k in range(max_iter):
        Fx = F(x)
        J = JF(x)
        dx = np.linalg.solve(J, -Fx)
        x = x + dx
        if np.linalg.norm(dx) < eps:
            return (x, k)
    return (x, max_iter)

# Exemple d'utilisation
x0 = np.array([0.1, 0.1, -0.1])
solution, nb_iter = newton_Rn(F, JF, x0)
print(f"\nSolution approchée : {solution} en {nb_iter} itérations.")
