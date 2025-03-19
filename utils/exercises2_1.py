from scipy.integrate import solve_ivp
from scipy.integrate import quad
from torch.distributions.multivariate_normal import MultivariateNormal
import torch

class Soft_LQR:
    def __init__(self, H, M, C, D, R, sigma, T, N, tau, gamma):
        """
        初始化 soft LQR 类

        Parameters:
        H, M, C, D, R: 线性二次调节器的矩阵
        sigma: 噪声项
        T: 终止时间
        N_steps: 时间步长
        tau: strength of entropic regularization
        gamma: strength of variance of prior normal density 
        time_grid: 时间网格
        """
        self.H = H
        self.M = M
        self.C = C
        self.D = D
        self.R = R
        self.sigma = sigma
        self.T = T
        self.N = N
        self.time_grid = torch.linspace(0, T, N+1)
        self.tau = tau
        self.gamma = gamma
        self.S_values = self.solve_riccati_ode()

    def riccati_ode(self, t, S_flat):
        """Riccati ODE 求解函数，转换为向量形式"""
        S = torch.tensor(S_flat, dtype=torch.float32).reshape(2,2) # 2x2 矩阵
        D_term = self.D + self.tau / (2 * (self.gamma**2)) * torch.eye(2)
        S_dot = S.T @ self.M @ torch.linalg.inv(D_term) @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return S_dot.flatten()

    def solve_riccati_ode(self):
        """使用 solve_ivp 求解 Riccati ODE"""
        S_T = self.R.flatten()  # 终止条件 S(T) = R
        indices = torch.arange(self.time_grid.size(0) - 1, -1, -1)  # 生成倒序索引
        time_grid_re = torch.index_select(self.time_grid, 0, indices)
        sol = solve_ivp(self.riccati_ode, [self.T, 0], S_T, t_eval = time_grid_re, atol = 1e-10, rtol = 1e-10)  # 逆向求解
        S_matrices = sol.y.T[::-1].reshape(-1, 2, 2)  # 转换回矩阵格式
        return dict(zip(tuple(self.time_grid.tolist()), S_matrices))

    def get_nearest_S(self, t):
        """找到最近的 S(t)"""
        nearest_t = self.time_grid[torch.argmin(torch.abs(self.time_grid - t))]
        return self.S_values[nearest_t.tolist()]
    
    def value_function(self, t, x):
        """计算新的 v(t, x) = x^T S(t) x + ∫[t,T] tr(σσ^T S(r)) dr + (T-t)C_{D,tau, gamma}"""
        # 第一部分：x^T S(t) x
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype = torch.float32)
        value = x.T @ S_t @ x
        
        # 第二部分：积分项 ∫[t,T] tr(σσ^T S(r)) dr
        def integrand(r):
            S_r = self.get_nearest_S(r)
            return torch.trace(self.sigma @ self.sigma.T @ S_r)
        
        integral, _ = quad(integrand, t, self.T)  # 使用数值积分计算积分项
        value += integral

        # 第三部分：(T-t)C_{D,tau, gamma}
        # C_{D,tau, gamma} = -tau ln(tau^{m/2}/gamma^{m} * det(∑)^{1/2}), ∑-1 = D+tau/(2*gamma^2)I
        var_matirx = self.D + self.tau / (2 * (self.gamma ** 2)) * torch.eye(2)
        inv_matrix = torch.linalg.inv(var_matirx)
        det_matrix = torch.det(inv_matrix)
        C = - self.tau * torch.log((self.tau / self.gamma ** 2) * torch.sqrt(det_matrix))
        entropic = (self.T - t) * C
        value += entropic

        return value
    
    def optimal_control(self, t, x):
        """计算最优控制分布 pi(·|t, x) = N(-(D+tau/(2*gamma^2)I)^(-1) M^T S(t) x, tau(D+tau/(2*gamma^2)I))"""
        S_t = self.get_nearest_S(t)
        S_t = torch.tensor(S_t, dtype=torch.float32)
        # mean
        inv_term = self.D + self.tau / (2 * (self.gamma ** 2)) * torch.eye(2)
        mean_control = -torch.linalg.inv(inv_term) @ self.M.T @ S_t @ x
        # covarian
        cov_control = self.tau * inv_term
        # distribution
        control_dist = MultivariateNormal(mean_control, cov_control)
        return control_dist