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
            'MM' for Method of Moments, 'PSL' for Pseudo Likelihood,
            'AUTO' to try MM first and fall back to PSL if skewness is wrong

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

        method = method.upper()
        if method not in ['MM', 'PSL', 'AUTO', 'NONE']:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine intercept and multiplicative model
        inter = 0 if rts == 'crs' else 1
        
        # Force multiplicative for certain RTS (as per R code lines 54-59)
        original_mult = mult
        if rts in ['crs', 'drs', 'irs']:
            mult = True
            if not original_mult:
                warnings.warn(f"Multiplicative model induced for {rts} returns to scale")
        
        # Create Z matrix and B vector
        # Following R code lines 66-97 exactly
        # Z matrix structure: [alpha_1, ..., alpha_n, beta_1_1, ..., beta_n_1, ..., beta_1_m, ..., beta_n_m]
        # where alpha_i is intercept for DMU i, beta_i_j is coefficient for input j of DMU i
        if mult:
            # Multiplicative: Y = b'X * exp(e) -> 1 = (b'X)/Y * exp(e)
            if inter == 1:
                # First n columns: intercepts (alpha) - diag(1/Y)
                Z = np.diag(1.0 / self.output)
                # Next m*n columns: beta coefficients - diag(X[,i]/Y) for each input i
                for i in range(self.n_inputs):
                    Z = np.column_stack([Z, np.diag(self.inputs[:, i] / self.output)])
            else:  # inter == 0
                # No intercept, only beta coefficients
                # Start with first input
                Z = np.diag(self.inputs[:, 0] / self.output)
                for i in range(1, self.n_inputs):
                    Z = np.column_stack([Z, np.diag(self.inputs[:, i] / self.output)])
            B = np.ones(self.n_dmus)
        else:  # MULT == 0
            # Additive: Y = b'X + e
            if inter == 1:
                # First n columns: intercepts (alpha) - identity matrix
                Z = np.eye(self.n_dmus)
                # Next m*n columns: beta coefficients - diag(X[,i]) for each input i
                for i in range(self.n_inputs):
                    Z = np.column_stack([Z, np.diag(self.inputs[:, i])])
            else:  # inter == 0
                # No intercept, only beta coefficients
                # Start with first input
                Z = np.diag(self.inputs[:, 0])
                for i in range(1, self.n_inputs):
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
        
        # Check for numerical issues in Z
        if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            raise ValueError("Z matrix contains NaN or Inf values")
        
        D = Z.T @ Z
        # Make D positive definite (as per R code line 193)
        D_max = np.max(np.abs(D))
        if D_max > 0:
            D = D + np.eye(D.shape[0]) * 1e-10 * D_max
        else:
            D = D + np.eye(D.shape[0]) * 1e-10
        
        # Check for numerical issues
        if np.any(np.isnan(D)) or np.any(np.isinf(D)):
            raise ValueError("D matrix contains NaN or Inf values")
        
        d = Z.T @ B
        
        # Solve QP using scipy.optimize.minimize
        # Standard form: min 0.5 * x' * D * x - d' * x
        # s.t. G @ x >= h (which is -G @ x <= -h for minimize)
        def objective(x):
            return 0.5 * x @ D @ x - d @ x
        
        # Constraint: G @ x >= h, which means -G @ x <= -h for scipy
        constraints = []
        for i in range(G.shape[0]):
            # Create closure to capture i
            def make_constraint(idx):
                return lambda x: G[idx, :] @ x - h[idx]
            constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
        
        # Initial guess - use least squares solution as starting point
        try:
            # Try to get a reasonable initial guess
            x0 = np.linalg.lstsq(Z, B, rcond=None)[0]
            # Ensure non-negative
            x0 = np.maximum(x0, 0.01)
        except:
            x0 = np.ones(n_vars) * 0.1
        
        # Bounds
        bounds = [(0, None)] * n_vars
        
        # Solve with multiple methods if first fails
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-9})
        
        if not result.success:
            # Try with trust-constr if SLSQP fails
            try:
                result = minimize(objective, x0, method='trust-constr', bounds=bounds,
                                constraints=constraints, options={'maxiter': 1000})
            except:
                pass
        
        if not result.success:
            warnings.warn(f"Optimization may not have converged: {result.message}")
        
        z_solution = result.x
        solution_norm = np.sum((Z @ z_solution - B)**2)
        
        # Calculate residuals and fitted values
        # beta_matrix: reshape according to R code: matrix(z_cnls$X, ncol = length(z_cnls$X) / n)
        # This means: n_cols = n_vars / n_dmus, and reshape by column (Fortran order)
        n_cols = n_vars // self.n_dmus
        # Reshape by column (like R's matrix function)
        beta_matrix = z_solution.reshape(n_cols, self.n_dmus).T
        
        if inter == 0:
            # No intercept: y_hat = diag(beta_matrix @ X^T)
            # Each row of beta_matrix is [beta_1, ..., beta_m] for one DMU
            y_hat = np.diag(beta_matrix @ self.inputs.T)
        else:
            # With intercept: y_hat = diag(beta_matrix @ [1, X]^T)
            # First column is intercept, rest are beta coefficients
            X_with_intercept = np.column_stack([np.ones(self.n_dmus), self.inputs])
            y_hat = np.diag(beta_matrix @ X_with_intercept.T)
        
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
        use_psl_fallback = False

        if method in ['MM', 'AUTO']:
            M2 = np.mean((resid - np.mean(resid))**2)
            M3 = np.mean((resid - np.mean(resid))**3)

            if cost:
                M3 = -M3

            if M3 > 0:
                if method == 'AUTO':
                    # Silently fall back to PSL
                    use_psl_fallback = True
                else:
                    warnings.warn("Wrong skewness? Third moment of residuals is greater than 0")

            # Estimate sigma_u and sigma_v
            if M3 < 0 and not use_psl_fallback:
                sigma_u = (abs(M3) / (np.sqrt(2 / np.pi) * (1 - 4 / np.pi)))**(1/3)
                sigma_v_sq = M2 - ((np.pi - 2) / np.pi) * sigma_u**2
                if sigma_v_sq > 0:
                    sigma_v = np.sqrt(sigma_v_sq)
                else:
                    if method == 'AUTO':
                        use_psl_fallback = True
                    else:
                        sigma_v = np.sqrt(M2)  # Fallback
                        warnings.warn("Negative variance in MM estimation, using fallback")
            elif not use_psl_fallback:
                # If M3 is positive or zero, cannot estimate properly
                if method == 'AUTO':
                    use_psl_fallback = True
                else:
                    sigma_u = np.nan
                    sigma_v = np.nan

            if not use_psl_fallback:
                lambda_param = sigma_u / sigma_v if (not np.isnan(sigma_v) and sigma_v > 0) else np.nan

        if method == 'PSL' or use_psl_fallback:
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

        if method == 'NONE':
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
            # Following R code line 311-314 (Keshvari/Kuosmanen 2013 correction)
            mu_star = -comp_err * sigma_u**2 / sum_sigma2
            sigma_star_sq = sigma_u**2 * sigma_v**2 / sum_sigma2
            
            # Conditional mean of inefficiency
            # R code: cond_mean <- mu_star + (sigma_u * sigma_v) *
            #         (dnorm(Comp_err / sigma_v * sigma_v) / (1 - pnorm(Comp_err / sigma_v * sigma_v)))
            # Note: sigma_v * sigma_v in denominator, not sigma_v^2 / sum_sigma2
            comp_err_scaled = comp_err / (sigma_v**2)
            norm_cdf_val = norm.cdf(comp_err_scaled)
            norm_pdf_val = norm.pdf(comp_err_scaled)
            
            # Handle edge cases
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(norm_cdf_val < 1.0 - 1e-10,
                               norm_pdf_val / (1 - norm_cdf_val),
                               np.inf)
                cond_mean = mu_star + (sigma_u * sigma_v) * ratio
            
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
        
        # Calculate slack norm (as per R code lines 202-204)
        # Check for numerical issues before computing slack
        if np.any(np.isnan(z_solution)) or np.any(np.isinf(z_solution)):
            slack_norm = np.nan
        else:
            try:
                slack = G @ z_solution - h
                slack_norm = -np.sum(slack[slack < 0])
            except:
                slack_norm = np.nan
        
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
                    mult: bool = False, method: str = 'AUTO') -> pd.DataFrame:
        """Evaluate all DMUs using AUTO method (tries MM, falls back to PSL)"""
        results = self.solve(rts, cost, mult, method)
        
        df = pd.DataFrame({
            'DMU': range(1, self.n_dmus + 1),
            'Efficiency': results['eff'],
            'Fitted': results['fit'],
            'Frontier': results['front'],
            'Residual': results['residuals']
        })
        
        return df

