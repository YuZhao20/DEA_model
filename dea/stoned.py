"""
StoNED (Stochastic Non-smooth Envelopment of Data) Model
Based on Benchmarking package stoned function
Combines DEA and SFA approaches
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional, Dict
import pandas as pd
import warnings


class StoNEDModel:
    """
    StoNED (Stochastic Non-smooth Envelopment of Data) Model
    
    Combines the axiomatic and non-parametric frontier (DEA aspect)
    with a stochastic noise term (SFA aspect).
    
    Based on Kuosmanen and Kortelainen (2012).
    """
    
    def __init__(self, inputs: np.ndarray, output: np.ndarray):
        """
        Initialize StoNED Model
        
        Parameters:
        -----------
        inputs : np.ndarray
            Input matrix of shape (n_dmus, n_inputs)
        output : np.ndarray
            Output vector of shape (n_dmus,) - single output only
        """
        self.inputs = np.array(inputs)
        self.output = np.array(output).flatten()
        self.n_dmus, self.n_inputs = self.inputs.shape
        
        if len(self.output) != self.n_dmus:
            raise ValueError("Number of DMUs must be the same for inputs and output")
        
        if self.output.ndim > 1:
            raise ValueError("Output must be a 1D array (single output only)")
    
    def solve(self, rts: str = 'vrs', cost: bool = False, mult: bool = False,
              method: str = 'MM') -> Dict:
        """
        Solve StoNED Model
        
        Parameters:
        -----------
        rts : str
            Returns to scale: 'vrs', 'drs', 'crs', 'irs'
        cost : bool
            If True, estimate cost function; if False, production function
        mult : bool
            If True, multiplicative error term; if False, additive error term
        method : str
            'MM' for Method of Moments, 'PSL' for Pseudo Likelihood
        
        Returns:
        --------
        results : dict
            Dictionary containing:
            - residualNorm: Norm of residual
            - solutionNorm: Norm of solution
            - error: Error flag
            - coef: Beta matrix (coefficients)
            - residuals: Residuals
            - fit: Fitted values
            - eff: Efficiency scores
            - front: Frontier reference points
            - sigma_u: Estimated sigma_u
        """
        rts = rts.lower()
        if rts not in ['vrs', 'drs', 'crs', 'irs']:
            raise ValueError(f"Unknown returns to scale: {rts}")
        
        if method not in ['MM', 'PSL', 'NONE']:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine intercept and multiplicative model
        inter = 0 if rts == 'crs' else 1
        
        # Force multiplicative for certain RTS
        if rts in ['crs', 'drs', 'irs']:
            mult = True
            if not mult:
                warnings.warn(f"Multiplicative model induced for {rts} returns to scale")
        
        # Create Z matrix and B vector
        # Z matrix has structure: [alpha_1, ..., alpha_n, beta_1_1, ..., beta_n_1, ..., beta_1_m, ..., beta_n_m]
        # where alpha_i is intercept for DMU i, beta_i_j is coefficient for input j of DMU i
        if mult:
            # Multiplicative: Y = b'X * exp(e) -> log(Y) = log(b'X) + e
            # For multiplicative, we transform to: 1 = (b'X)/Y * exp(e)
            if inter == 1:
                # First n columns: intercepts (alpha)
                Z = np.diag(1.0 / self.output)
                # Next m*n columns: beta coefficients
                for i in range(self.n_inputs):
                    Z = np.column_stack([Z, np.diag(self.inputs[:, i] / self.output)])
            else:
                # No intercept, only beta coefficients
                Z = None
                for i in range(self.n_inputs):
                    if Z is None:
                        Z = np.diag(self.inputs[:, i] / self.output)
                    else:
                        Z = np.column_stack([Z, np.diag(self.inputs[:, i] / self.output)])
            B = np.ones(self.n_dmus)
        else:
            # Additive: Y = b'X + e
            if inter == 1:
                # First n columns: intercepts (alpha)
                Z = np.eye(self.n_dmus)
                # Next m*n columns: beta coefficients
                for i in range(self.n_inputs):
                    Z = np.column_stack([Z, np.diag(self.inputs[:, i])])
            else:
                # No intercept, only beta coefficients
                Z = None
                for i in range(self.n_inputs):
                    if Z is None:
                        Z = np.diag(self.inputs[:, i])
                    else:
                        Z = np.column_stack([Z, np.diag(self.inputs[:, i])])
            B = self.output
        
        # Create constraints
        # Non-negativity constraints for beta
        n_vars = Z.shape[1]
        non_neg_beta = np.hstack([np.zeros((self.n_inputs * self.n_dmus, self.n_dmus)),
                                  np.eye(self.n_inputs * self.n_dmus)])
        
        # RTS constraints
        if inter == 1:
            if rts == 'vrs':
                sign_cons = non_neg_beta
            elif rts == 'drs':
                rts_cons = np.hstack([(1 if not cost else -1) * np.eye(self.n_inputs * self.n_dmus),
                                     np.zeros((self.n_inputs * self.n_dmus, self.n_dmus))])
                sign_cons = np.vstack([non_neg_beta, rts_cons])
            elif rts == 'irs':
                rts_cons = np.hstack([(-1 if not cost else 1) * np.eye(self.n_inputs * self.n_dmus),
                                     np.zeros((self.n_inputs * self.n_dmus, self.n_dmus))])
                sign_cons = np.vstack([non_neg_beta, rts_cons])
        else:
            sign_cons = np.eye(self.n_inputs * self.n_dmus)
        
        # Concavity/convexity constraints
        alpha_list = []
        beta_list = []
        
        for i in range(self.n_dmus):
            # Alpha matrix
            a_mat = np.eye(self.n_dmus)
            a_mat[:, i] = a_mat[:, i] - 1
            alpha_list.append(a_mat)
            
            # Beta matrix
            b_mat = np.zeros((self.n_dmus, self.n_inputs * self.n_dmus))
            for j in range(self.n_inputs):
                b_mat[:, i + self.n_dmus * j] = -self.inputs[i, j]
                for k in range(self.n_dmus):
                    b_mat[k, k + j * self.n_dmus] = b_mat[k, k + j * self.n_dmus] + self.inputs[i, j]
            beta_list.append(b_mat)
        
        if inter == 1:
            sspot_cons = np.hstack([np.vstack(alpha_list), np.vstack(beta_list)])
        else:
            sspot_cons = np.vstack(beta_list)
        
        # Cost function: convexity instead of concavity
        if cost:
            sspot_cons = -sspot_cons
        
        # Combine constraints
        G = np.vstack([sign_cons, sspot_cons])
        h = np.zeros(G.shape[0])
        
        # Remove zero rows
        zero_rows = np.all(G == 0, axis=1)
        G = G[~zero_rows, :]
        h = h[~zero_rows]
        
        # Quadratic programming problem
        # min 0.5 * x' * D * x - d' * x
        # s.t. G' * x >= h
        # Check for zeros in output (for multiplicative model)
        if mult and np.any(self.output <= 0):
            raise ValueError("Output must be positive for multiplicative model")
        
        D = Z.T @ Z
        # Make D positive definite
        D_max = np.max(np.abs(D))
        if D_max > 0:
            D = D + np.eye(D.shape[0]) * 1e-10 * D_max
        else:
            D = D + np.eye(D.shape[0]) * 1e-10
        d = Z.T @ B
        
        # Solve QP using scipy.optimize.minimize
        # Convert to standard form: min 0.5 * x' * D * x - d' * x
        # s.t. -G' * x <= -h (for minimize)
        def objective(x):
            return 0.5 * x @ D @ x - d @ x
        
        constraints = []
        for i in range(G.shape[0]):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: -G[idx, :] @ x + h[idx]
            })
        
        # Initial guess
        x0 = np.ones(n_vars) * 0.1
        
        # Bounds
        bounds = [(0, None)] * n_vars
        
        # Solve
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            warnings.warn(f"Optimization may not have converged: {result.message}")
        
        z_solution = result.x
        solution_norm = np.sum((Z @ z_solution - B)**2)
        
        # Calculate residuals and fitted values
        # beta_matrix: each row is for one DMU, columns are [alpha, beta_1, ..., beta_m] if inter==1
        # or [beta_1, ..., beta_m] if inter==0
        n_cols = n_vars // self.n_dmus
        beta_matrix = z_solution.reshape(self.n_dmus, n_cols)
        
        if inter == 0:
            # No intercept: y_hat = beta_matrix @ X^T (each row of beta_matrix is [beta_1, ..., beta_m])
            y_hat = np.array([beta_matrix[i, :] @ self.inputs[i, :] for i in range(self.n_dmus)])
        else:
            # With intercept: y_hat = alpha + beta_matrix @ X^T
            # First column is intercept, rest are beta coefficients
            y_hat = np.array([beta_matrix[i, 0] + beta_matrix[i, 1:] @ self.inputs[i, :] 
                             for i in range(self.n_dmus)])
        
        if mult:
            # Avoid log of zero or negative
            y_hat_positive = np.maximum(y_hat, 1e-10)
            resid = np.log(self.output) - np.log(y_hat_positive)
            fitted = self.output / np.exp(resid)
            fitted = np.maximum(fitted, 1e-10)  # Ensure positive
        else:
            resid = self.output - y_hat
            fitted = self.output - resid
        
        # Estimate inefficiency (sigma_u, sigma_v)
        if method == 'MM':
            M2 = np.mean((resid - np.mean(resid))**2)
            M3 = np.mean((resid - np.mean(resid))**3)
            
            if cost:
                M3 = -M3
            
            if M3 > 0:
                warnings.warn("Wrong skewness? Third moment of residuals is greater than 0")
            
            # Estimate sigma_u and sigma_v
            if M3 < 0:
                sigma_u = (abs(M3) / (np.sqrt(2 / np.pi) * (1 - 4 / np.pi)))**(1/3)
                sigma_v_sq = M2 - ((np.pi - 2) / np.pi) * sigma_u**2
                if sigma_v_sq > 0:
                    sigma_v = np.sqrt(sigma_v_sq)
                else:
                    sigma_v = np.sqrt(M2)  # Fallback
                    warnings.warn("Negative variance in MM estimation, using fallback")
            else:
                # If M3 is positive or zero, cannot estimate properly
                sigma_u = np.nan
                sigma_v = np.nan
            lambda_param = sigma_u / sigma_v if (not np.isnan(sigma_v) and sigma_v > 0) else np.nan
        
        elif method == 'PSL':
            # Pseudo likelihood estimation
            def neg_log_likelihood(lambda_param, e):
                sc = np.sqrt(2 * lambda_param**2 / np.pi / (1 + lambda_param**2))
                si = np.sqrt(np.mean(e**2) / (1 - sc**2))
                mu = si * sc
                ep = e - mu
                likelihood = (-len(ep) * np.log(si) +
                             np.sum(norm.logcdf(-ep * lambda_param / si)) -
                             0.5 * np.sum(ep**2) / si**2)
                return -likelihood
            
            if cost:
                sol = minimize(lambda lmd: neg_log_likelihood(lmd, -resid), x0=1.0, method='BFGS')
            else:
                sol = minimize(lambda lmd: neg_log_likelihood(lmd, resid), x0=1.0, method='BFGS')
            
            if not sol.success:
                warnings.warn(f"PSL optimization may not have converged: {sol.message}")
            
            lambda_param = sol.x[0]
            sc = np.sqrt(2 * lambda_param**2 / np.pi / (1 + lambda_param**2))
            sig_sq = np.mean(resid**2) / (1 - sc**2)
            sigma_v = np.sqrt(sig_sq / (1 + lambda_param**2))
            sigma_u = sigma_v * lambda_param
        else:  # METHOD == 'NONE'
            sigma_u = np.nan
            sigma_v = np.nan
            lambda_param = np.nan
        
        # Calculate efficiency scores
        if not np.isnan(sigma_u) and not np.isnan(sigma_v) and sigma_v > 0:
            sum_sigma2 = sigma_u**2 + sigma_v**2
            
            # Composite error
            if cost:
                comp_err = resid + sigma_u * np.sqrt(2 / np.pi)
            else:
                comp_err = resid - sigma_u * np.sqrt(2 / np.pi)
            
            # Conditional mean
            mu_star = -comp_err * sigma_u**2 / sum_sigma2
            sigma_star_sq = sigma_u**2 * sigma_v**2 / sum_sigma2
            sigma_star = np.sqrt(sigma_star_sq) if sigma_star_sq > 0 else 0.0
            
            # Conditional mean of inefficiency
            # Avoid division by zero
            comp_err_scaled = comp_err / np.sqrt(sigma_v**2 / sum_sigma2) if sum_sigma2 > 0 else comp_err
            norm_cdf_val = norm.cdf(comp_err_scaled)
            norm_pdf_val = norm.pdf(comp_err_scaled)
            
            # Handle edge cases
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(norm_cdf_val < 1.0 - 1e-10,
                               norm_pdf_val / (1 - norm_cdf_val),
                               np.inf)
                cond_mean = mu_star + (sigma_u * sigma_v / np.sqrt(sum_sigma2)) * ratio
            
            if mult:
                eff_score = np.exp(-cond_mean)
                if cost:
                    frontier_points = fitted * np.exp(-sigma_u * np.sqrt(2 / np.pi))
                else:
                    frontier_points = fitted * np.exp(sigma_u * np.sqrt(2 / np.pi))
            else:
                eff_score = 1 - cond_mean / self.output
                # Clip efficiency scores to reasonable range
                eff_score = np.clip(eff_score, 0, 2)
                if cost:
                    frontier_points = fitted - sigma_u * np.sqrt(2 / np.pi)
                else:
                    frontier_points = fitted + sigma_u * np.sqrt(2 / np.pi)
        else:
            eff_score = np.full(self.n_dmus, np.nan)
            frontier_points = np.full(self.n_dmus, np.nan)
        
        # Calculate slack norm
        slack = G @ z_solution - h
        slack_norm = -np.sum(slack[slack < 0])
        
        return {
            'residualNorm': slack_norm,
            'solutionNorm': solution_norm,
            'error': not result.success,
            'coef': beta_matrix,
            'sol': z_solution,
            'residuals': resid,
            'fit': fitted,
            'yhat': y_hat,
            'eff': eff_score,
            'front': frontier_points,
            'sigma_u': sigma_u
        }
    
    def evaluate_all(self, rts: str = 'vrs', cost: bool = False,
                    mult: bool = False, method: str = 'MM') -> pd.DataFrame:
        """Evaluate all DMUs"""
        results = self.solve(rts, cost, mult, method)
        
        df = pd.DataFrame({
            'DMU': range(1, self.n_dmus + 1),
            'Efficiency': results['eff'],
            'Fitted': results['fit'],
            'Frontier': results['front'],
            'Residual': results['residuals']
        })
        
        return df

