import numpy as np
import math
import os
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from scipy.integrate import quad

# -------------------------- UTILITIES -------------------------- #

def init_vector(dim, norm):
    vec = 2 * np.random.rand(dim) - 1.
    return vec * norm / math.sqrt(dim)

def init_arms(dim, norm, num):
    return np.array([init_vector(dim, norm) for _ in range(num)])

def project_to_unit_ball(theta):
    norm = np.linalg.norm(theta)
    return theta / norm if norm > 1 else theta

# ---------------------- REWARD FUNCTION ------------------------ #

def reward_function(chosen_arm, theta_star, flag, df=3, scale=1):
    expected_payoff = func(chosen_arm.dot(theta_star))
    noise = t_dist.rvs(df=df) * scale
    return expected_payoff + noise if flag == 0 else -expected_payoff + noise

# ---------------------- OPTIMIZATION --------------------------- #

def z_s(u, y_s, sigma_s, f):
    return (y_s - f(u)) / sigma_s

def objective_function(theta, lambda_k, B, k, t, phi_s, y_s, sigma_s, tau_s, f):
    loss = lambda_k * k / 2 * np.linalg.norm(theta)**2
    for s in range(1, t + 1):
        integral_func = lambda u: (tau_s[s] * z_s(u, y_s[s], sigma_s[s], f)) / np.sqrt(tau_s[s]**2 + z_s(u, y_s[s], sigma_s[s], f)**2)
        result, _ = quad(integral_func, 0, np.dot(phi_s[s], theta))
        loss -= result / sigma_s[s]
    return loss

def gradient_function(theta, lambda_k, B, k, t, phi_s, y_s, sigma_s, tau_s, f):
    grad = lambda_k * k * theta
    for s in range(1, t + 1):
        z = z_s(np.dot(phi_s[s], theta), y_s[s], sigma_s[s], f)
        grad -= phi_s[s] * (tau_s[s] * z) / (sigma_s[s] * np.sqrt(tau_s[s]**2 + z**2))
    return grad

def constraint12(theta):
    return 1 - np.linalg.norm(theta)

# ---------------------- ALGORITHM CORE -------------------------- #

def gadaoful(
    T, B, L, k, K, dim, norm, actions, corruption, sigma, bmu, func,
    repeat_index, repeat_time
):
    number = repeat_index - repeat_time * int(repeat_index / repeat_time)
    print("Corruption rounds allowed:", corruption)
    print("Experiment ID:", number)

    cur_crr = 1
    lambda_ = 20
    tau_0 = 1
    m_0 = 1.5
    alpha = 1
    H = np.eye(dim) * lambda_
    theta = np.zeros(dim)
    beta = 0.6
    sigma_min = 1 / np.sqrt(T)

    sigma_ = [None] * (T + 1)
    w_ = [None] * (T + 1)
    tau_ = [None] * (T + 1)
    y_ = [None] * (T + 1)
    var_ = [None] * (T + 1)
    phi_ = [None] * (T + 1)
    REGRET = 0
    TOTALREGRET = []

    for t in range(1, T + 1):
        noise = t_dist.rvs(df=3, size=actions) * sigma
        flag = 1 if cur_crr < corruption else 0
        if flag:
            cur_crr += 1

        decision = init_arms(dim, norm, actions)
        reward = np.zeros(actions)
        optimal_reward = float("-inf")

        for arm in range(actions):
            reward_s = noise[arm] + np.dot(decision[arm], bmu)
            reward[arm] = reward_s if flag == 0 else (noise[arm] - np.dot(decision[arm], bmu))
            optimal_reward = max(optimal_reward, np.dot(decision[arm], bmu))

        best_i, max_dot = None, float('-inf')
        for i in range(actions):
            def obj(theta_): return -np.dot(decision[i], theta_)
            cons = [
                {'type': 'ineq', 'fun': lambda theta_: B - np.linalg.norm(theta_)},
                {'type': 'ineq', 'fun': lambda theta_: beta - np.sqrt(np.dot(np.dot(theta_ - theta, H), theta_ - theta))}
            ]
            result = minimize(obj, np.zeros_like(theta), constraints=cons)
            if -result.fun > max_dot:
                max_dot = -result.fun
                best_i = i

        phi_[t] = decision[best_i]
        REGRET += func(optimal_reward) - func(np.dot(phi_[t], bmu))
        phi_H_inv_norm = np.sqrt(np.dot(np.dot(phi_[t], np.linalg.inv(H)), phi_[t]))
        y_[t] = reward_function(phi_[t], bmu, flag)
        var_[t] = sigma
        sigma_candidates = [var_[t], sigma_min, phi_H_inv_norm / m_0, alpha * (phi_H_inv_norm ** 0.5)]
        sigma_[t] = max(sigma_candidates)
        w_[t] = phi_H_inv_norm / sigma_[t]
        tau_[t] = tau_0 * np.sqrt(1 + w_[t] ** 2) / w_[t]

        theta = minimize(
            fun=objective_function,
            x0=theta,
            args=(lambda_, B, k, t, phi_, y_, sigma_, tau_, func),
            jac=gradient_function,
            constraints=[{'type': 'ineq', 'fun': constraint12}]
        ).x

        H += np.outer(phi_[t], phi_[t]) / (sigma_[t] ** 2)

        if t % 10 == 0:
            print(f"Step {t} | REGRET = {REGRET:.4f}")
        TOTALREGRET.append(REGRET)

    return TOTALREGRET

# ---------------------- ENTRY POINT ---------------------------- #

def func(x):
    return x

if __name__ == "__main__":
    dim = 10
    sigma = 1
    corruption = 0
    T = 1000
    repeat = 10
    repeat_time = repeat
    actions = 20
    norm = 1
    bmu = np.ones(dim) / math.sqrt(dim)

    total_regret = gadaoful(
        T=T, B=1, L=1, k=1, K=1, dim=dim, norm=norm, actions=actions,
        corruption=corruption, sigma=sigma, bmu=bmu, func=func,
        repeat_index=0, repeat_time=repeat_time
    )
