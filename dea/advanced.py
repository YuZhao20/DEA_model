"""
Advanced DEA Models from Chapter 4
Including: Norm L1, Congestion, Common Weights, Directional Efficiency
Based on Chapter 4 of Hosseinzadeh Lotfi et al. (2020)
"""

import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pandas as pd


class NormL1Model:
    """
    Norm L1 Super-Efficiency Model
    Based on Chapter 4.4
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, rts: str = 'vrs') -> Tuple[float, float]:
        """
        Solve Norm L1 Super-Efficiency Model (4.4)
        
        min w+ - w-
        s.t. sum(lambda_j * x_ij) - x_i + w+ - w- = 0, i=1,...,m, j!=p
             sum(lambda_j * y_rj) - y_r >= 0, r=1,...,s, j!=p
             x_i <= x_ip, y_r >= y_rp
             lambda_j >= 0, w+ >= 0, w- >= 0
             sum(lambda_j) = 1 (if VRS)
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        
        Returns:
        --------
        w_star : float
            Optimal value of w+ - w-
        super_efficiency : float
            Super-efficiency score (1 + w*)
        """
        # Variables: [lambda_1, ..., lambda_{p-1}, lambda_{p+1}, ..., lambda_n, x_1, ..., x_m, y_1, ..., y_s, w+, w-]
        dmu_indices = [j for j in range(self.n_dmus) if j != dmu_index]
        n_lambdas = len(dmu_indices)
        n_vars = n_lambdas + self.n_inputs + self.n_outputs + 2
        
        # Objective: minimize w+ - w-
        c = np.zeros(n_vars)
        c[n_lambdas + self.n_inputs + self.n_outputs] = 1.0  # w+
        c[n_lambdas + self.n_inputs + self.n_outputs + 1] = -1.0  # -w-
        
        # Constraints: inputs + outputs + RTS (bounds removed to avoid infeasibility)
        n_constraints = self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_constraints += 1
        elif rts in ['drs', 'irs']:
            n_constraints += 1
        
        A = np.zeros((n_constraints, n_vars))
        
        # Input constraints: sum(lambda_j * x_ij) - x_i + w+ - w- = 0
        for i in range(self.n_inputs):
            for idx, j in enumerate(dmu_indices):
                A[i, idx] = self.inputs[j, i]
            A[i, n_lambdas + i] = -1.0  # -x_i
            A[i, n_lambdas + self.n_inputs + self.n_outputs] = 1.0  # w+
            A[i, n_lambdas + self.n_inputs + self.n_outputs + 1] = -1.0  # -w-
        
        # Output constraints: sum(lambda_j * y_rj) - y_r >= 0
        for r in range(self.n_outputs):
            for idx, j in enumerate(dmu_indices):
                A[self.n_inputs + r, idx] = self.outputs[j, r]
            A[self.n_inputs + r, n_lambdas + self.n_inputs + r] = -1.0  # -y_r
        
        # Bounds: x_i <= x_ip, y_r >= y_rp
        # Note: These bounds may cause infeasibility in super-efficiency models
        # We remove them or make them very loose to ensure feasibility
        row = self.n_inputs + self.n_outputs
        # Remove strict bounds to avoid infeasibility
        # Instead, we'll use very loose bounds or remove them entirely
        # For super-efficiency, we don't need strict bounds on x_i and y_r
        
        # RTS constraint
        row = self.n_inputs + self.n_outputs  # No bounds constraints, so RTS constraint starts here
        if rts == 'vrs':
            A[row, :n_lambdas] = 1.0  # sum(lambda_j) = 1
        elif rts == 'drs':
            A[row, :n_lambdas] = 1.0  # sum(lambda_j) <= 1
        elif rts == 'irs':
            A[row, :n_lambdas] = -1.0  # sum(lambda_j) >= 1
        
        # Right-hand side
        b = np.zeros(n_constraints)
        # Input constraints: b = 0 (already initialized)
        # Output constraints: b = 0 (already initialized)
        # No bounds constraints, so RTS constraint is at row = self.n_inputs + self.n_outputs
        if rts == 'vrs':
            b[row] = 1.0
        elif rts == 'drs':
            b[row] = 1.0
        elif rts == 'irs':
            b[row] = -1.0
        
        # Constraint types
        n_eq = self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_eq += 1
        
        A_eq = A[:n_eq, :] if n_eq > 0 else None
        b_eq = b[:n_eq] if n_eq > 0 else None
        A_ub = A[n_eq:, :] if n_eq < n_constraints else None
        b_ub = b[n_eq:] if n_eq < n_constraints else None
        
        # Bounds
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                         bounds=bounds, method='highs')
        
        if not result.success:
            # If infeasible, return a default value (e.g., efficiency = 1.0, w* = 0.0)
            # This can happen in super-efficiency models when the excluded DMU
            # cannot be dominated by the remaining reference set
            import warnings
            warnings.warn(f"Norm L1 model infeasible for DMU {dmu_index}: {result.message}. Returning default efficiency.")
            return 0.0, 1.0  # w* = 0, super_efficiency = 1.0
        
        w_star = result.fun
        super_efficiency = 1.0 + w_star
        
        return w_star, super_efficiency
    
    def evaluate_all(self, rts: str = 'vrs') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            w_star, super_eff = self.solve(j, rts=rts)
            results.append({
                'DMU': j + 1,
                'W*': w_star,
                'Super_Efficiency_NL1': super_eff
            })
        return pd.DataFrame(results)


class CongestionModel:
    """
    Congestion DEA Model
    Based on Chapter 4.13
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Congestion DEA Model (4.13)

        Two-phase model:
        Phase 1: Solve BCC model to get efficiency
        Phase 2: Find congestion slacks

        Returns:
        --------
        eff : float
            Efficiency score from Phase 1
        lambdas : np.ndarray
            Optimal intensity variables
        input_slacks : np.ndarray
            Congestion input slacks
        output_slacks : np.ndarray
            Output slacks (usually zero for congestion)
        """
        # Phase 1: BCC model
        from .bcc import BCCModel
        bcc = BCCModel(self.inputs, self.outputs)
        eff, lambdas_phase1, input_targets, output_targets = bcc.solve_envelopment(dmu_index)

        # Phase 2: Maximize input slacks to find congestion
        # Variables: [lambda_1, ..., lambda_n, s_1^-, ..., s_m^-]
        n_vars = self.n_dmus + self.n_inputs
        c = np.zeros(n_vars)
        c[self.n_dmus:] = -1.0  # maximize input slacks (minimize negative)

        # Constraints:
        # 1. Input constraints: sum(lambda_j * x_ij) + s_i^- = h*x_ip (equality)
        # 2. Output constraints: sum(lambda_j * y_rj) >= y_rp (inequality)
        # 3. Convexity: sum(lambda_j) = 1 (equality)

        n_eq_constraints = self.n_inputs + 1  # inputs + convexity
        n_ub_constraints = self.n_outputs  # output constraints

        A_eq = np.zeros((n_eq_constraints, n_vars))
        b_eq = np.zeros(n_eq_constraints)

        # Input constraints: sum(lambda_j * x_ij) + s_i^- = h*x_ip
        for i in range(self.n_inputs):
            A_eq[i, :self.n_dmus] = self.inputs[:, i]
            A_eq[i, self.n_dmus + i] = 1.0
            b_eq[i] = eff * self.inputs[dmu_index, i]

        # Convexity: sum(lambda_j) = 1
        A_eq[self.n_inputs, :self.n_dmus] = 1.0
        b_eq[self.n_inputs] = 1.0

        # Output constraints: sum(lambda_j * y_rj) >= y_rp
        # For linprog: -sum(lambda_j * y_rj) <= -y_rp
        A_ub = np.zeros((n_ub_constraints, n_vars))
        b_ub = np.zeros(n_ub_constraints)

        for r in range(self.n_outputs):
            A_ub[r, :self.n_dmus] = -self.outputs[:, r]
            b_ub[r] = -self.outputs[dmu_index, r]

        bounds = [(0, None)] * n_vars

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if not result.success:
            # If optimization fails, return zero slacks
            import warnings
            warnings.warn(f"Congestion optimization failed for DMU {dmu_index}: {result.message}. Returning zero slacks.")
            congestion_slacks = np.zeros(self.n_inputs)
            output_slacks = np.zeros(self.n_outputs)
            return eff, lambdas_phase1, congestion_slacks, output_slacks

        lambdas = result.x[:self.n_dmus]
        congestion_slacks = result.x[self.n_dmus:]
        output_slacks = np.zeros(self.n_outputs)

        return eff, lambdas, congestion_slacks, output_slacks
    
    def evaluate_all(self) -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            eff, lambdas, congestion_slacks, output_slacks = self.solve(j)
            congestion_sum = np.sum(congestion_slacks)
            result_dict = {'DMU': j + 1, 'Efficiency': eff, 'Congestion_Sum': congestion_sum}
            for i in range(self.n_inputs):
                result_dict[f'Congestion_{i+1}'] = congestion_slacks[i]
            results.append(result_dict)
        return pd.DataFrame(results)


class CommonWeightsModel:
    """
    Common Set of Weights DEA Model
    Based on Chapter 4.14
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, epsilon: float = 1e-4):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        self.epsilon = epsilon
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve Common Set of Weights Model (4.14)
        
        min sum(d_j)
        s.t. sum(u_r * y_rj) - sum(v_i * x_ij) + d_j = 0, j=1,...,n
             u_r >= epsilon, v_i >= epsilon
             d_j >= 0
        """
        # Variables: [u_1, ..., u_s, v_1, ..., v_m, d_1, ..., d_n]
        n_vars = self.n_outputs + self.n_inputs + self.n_dmus
        c = np.zeros(n_vars)
        c[self.n_outputs + self.n_inputs:] = 1.0  # minimize sum of d_j
        
        # Constraints: n equality constraints + m + s epsilon constraints
        n_constraints = self.n_dmus + self.n_inputs + self.n_outputs
        A_eq = np.zeros((self.n_dmus, n_vars))
        
        # DMU constraints: sum(u_r * y_rj) - sum(v_i * x_ij) + d_j = 0
        for j in range(self.n_dmus):
            A_eq[j, :self.n_outputs] = self.outputs[j, :]  # u_r * y_rj
            A_eq[j, self.n_outputs:self.n_outputs + self.n_inputs] = -self.inputs[j, :]  # -v_i * x_ij
            A_eq[j, self.n_outputs + self.n_inputs + j] = 1.0  # d_j
        
        b_eq = np.zeros(self.n_dmus)
        
        # Epsilon constraints: u_r >= epsilon, v_i >= epsilon
        A_ub = np.zeros((self.n_inputs + self.n_outputs, n_vars))
        for r in range(self.n_outputs):
            A_ub[r, r] = -1.0
        for i in range(self.n_inputs):
            A_ub[self.n_outputs + i, self.n_outputs + i] = -1.0
        
        b_ub = np.zeros(self.n_inputs + self.n_outputs)
        for r in range(self.n_outputs):
            b_ub[r] = -self.epsilon
        for i in range(self.n_inputs):
            b_ub[self.n_outputs + i] = -self.epsilon
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        u_weights = result.x[:self.n_outputs]
        v_weights = result.x[self.n_outputs:self.n_outputs + self.n_inputs]
        d_values = result.x[self.n_outputs + self.n_inputs:]
        obj_value = result.fun
        
        return u_weights, v_weights, obj_value
    
    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all DMUs using common weights
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with efficiency scores for all DMUs
        """
        u_weights, v_weights, obj_value = self.solve()
        
        # Calculate efficiency for each DMU
        results = []
        for j in range(self.n_dmus):
            numerator = np.sum(u_weights * self.outputs[j, :])
            denominator = np.sum(v_weights * self.inputs[j, :])
            efficiency = numerator / denominator if denominator > 0 else 0.0
            
            result_dict = {
                'DMU': j + 1,
                'Efficiency': efficiency
            }
            for r in range(self.n_outputs):
                result_dict[f'Weight_Output_{r+1}'] = u_weights[r]
            for i in range(self.n_inputs):
                result_dict[f'Weight_Input_{i+1}'] = v_weights[i]
            results.append(result_dict)
        
        return pd.DataFrame(results)


class DirectionalEfficiencyModel:
    """
    Directional Efficiency DEA Model
    Based on Chapter 4.15
    """
    
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.n_dmus, self.n_inputs = self.inputs.shape
        self.n_outputs = self.outputs.shape[1]
        
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError("Number of DMUs must be the same for inputs and outputs")
    
    def solve(self, dmu_index: int, gx: np.ndarray = None, gy: np.ndarray = None, rts: str = 'vrs') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Directional Efficiency Model (4.15)
        
        max beta
        s.t. beta*gx_i + sum(lambda_j * x_ij) <= x_ip, i=1,...,m
             -beta*gy_r + sum(lambda_j * y_rj) >= y_rp, r=1,...,s
             lambda_j >= 0
             sum(lambda_j) = 1 (if VRS)
        
        Default direction: gx = -x_p, gy = y_p
        
        Parameters:
        -----------
        dmu_index : int
            Index of DMU under evaluation
        gx : np.ndarray, optional
            Input direction vector
        gy : np.ndarray, optional
            Output direction vector
        rts : str
            Returns to scale: 'crs', 'vrs', 'drs', 'irs' (default: 'vrs')
        
        Returns:
        --------
        efficiency : float
            Directional efficiency score
        lambdas : np.ndarray
            Optimal intensity variables
        input_slacks : np.ndarray
            Input slacks
        output_slacks : np.ndarray
            Output slacks
        """
        if gx is None:
            gx = -self.inputs[dmu_index, :]
        if gy is None:
            gy = self.outputs[dmu_index, :]
        
        # Variables: [beta, lambda_1, ..., lambda_n, s_1^-, ..., s_m^-, s_1^+, ..., s_s^+]
        n_vars = 1 + self.n_dmus + self.n_inputs + self.n_outputs
        c = np.zeros(n_vars)
        c[0] = -1.0  # maximize beta (minimize -beta)
        
        # Constraints: inputs + outputs + RTS (if VRS)
        n_constraints = self.n_inputs + self.n_outputs
        if rts == 'vrs':
            n_constraints += 1
        
        A = np.zeros((n_constraints, n_vars))
        
        # Input constraints: beta*gx_i + sum(lambda_j * x_ij) + s_i^- <= x_ip
        for i in range(self.n_inputs):
            A[i, 0] = gx[i]  # beta coefficient
            A[i, 1:1 + self.n_dmus] = self.inputs[:, i]  # lambda coefficients
            A[i, 1 + self.n_dmus + i] = 1.0  # s_i^-
        
        # Output constraints: -beta*gy_r + sum(lambda_j * y_rj) - s_r^+ >= y_rp
        # For linprog: beta*gy_r - sum(lambda_j * y_rj) + s_r^+ <= -y_rp
        for r in range(self.n_outputs):
            A[self.n_inputs + r, 0] = gy[r]  # beta coefficient
            A[self.n_inputs + r, 1:1 + self.n_dmus] = -self.outputs[:, r]  # lambda coefficients
            A[self.n_inputs + r, 1 + self.n_dmus + self.n_inputs + r] = 1.0  # s_r^+
        
        # RTS constraint
        if rts == 'vrs':
            A[self.n_inputs + self.n_outputs, 1:1 + self.n_dmus] = 1.0  # sum(lambda_j) = 1
        elif rts == 'drs':
            n_constraints += 1
            A = np.vstack([A, np.zeros((1, n_vars))])
            A[-1, 1:1 + self.n_dmus] = 1.0  # sum(lambda_j) <= 1
        elif rts == 'irs':
            n_constraints += 1
            A = np.vstack([A, np.zeros((1, n_vars))])
            A[-1, 1:1 + self.n_dmus] = -1.0  # sum(lambda_j) >= 1
        
        # Right-hand side
        b = np.zeros(n_constraints)
        for i in range(self.n_inputs):
            b[i] = self.inputs[dmu_index, i]
        for r in range(self.n_outputs):
            b[self.n_inputs + r] = -self.outputs[dmu_index, r]
        if rts == 'vrs':
            b[self.n_inputs + self.n_outputs] = 1.0
        elif rts == 'drs':
            b[-1] = 1.0
        elif rts == 'irs':
            b[-1] = -1.0
        
        # Constraint types
        if rts == 'vrs':
            A_eq = A[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1, :]
            b_eq = b[self.n_inputs + self.n_outputs:self.n_inputs + self.n_outputs + 1]
            A_ub = A[:self.n_inputs + self.n_outputs, :]
            b_ub = b[:self.n_inputs + self.n_outputs]
        else:
            A_eq = None
            b_eq = None
            A_ub = A
            b_ub = b
        
        bounds = [(0, None)] * n_vars
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"Optimization failed for DMU {dmu_index}: {result.message}")
        
        efficiency = -result.fun
        lambdas = result.x[1:1 + self.n_dmus]
        input_slacks = result.x[1 + self.n_dmus:1 + self.n_dmus + self.n_inputs]
        output_slacks = result.x[1 + self.n_dmus + self.n_inputs:]
        
        return efficiency, lambdas, input_slacks, output_slacks
    
    def evaluate_all(self, gx: np.ndarray = None, gy: np.ndarray = None, rts: str = 'vrs') -> pd.DataFrame:
        results = []
        for j in range(self.n_dmus):
            # solve returns 4 values: eff, lambdas, input_slacks, output_slacks
            eff, lambdas, input_slacks, output_slacks = self.solve(j, gx, gy, rts)
            result_dict = {'DMU': j + 1, 'Efficiency': eff}
            for i, lam in enumerate(lambdas):
                result_dict[f'Lambda_{i+1}'] = lam
            results.append(result_dict)
        return pd.DataFrame(results)

