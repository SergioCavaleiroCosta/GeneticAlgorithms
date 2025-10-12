"""
Auto-generated equations from PySR training
============================================

This module contains the best discovered equations from symbolic regression.
Each function implements the best-accuracy model for its respective target.

Generated on: 2025-10-11T18:03:25.914859
Source: sr_expansive_burner multi-target training
Training config: 1000 iterations, 100 population_size, 30 maxsize, 1e-4 parsimony
"""

import numpy as np
from typing import Union, Dict, Any

# Model metadata
MODEL_INFO = {
    "20251010_212538_v2sJPa": {
        "equation": "(tanh((((phi * phi) / 0.1555563) ^ ar) * (eps_1 / (u_avg * 0.5385095))) ^ ((a_1 * ((phi * 1.3230652) - ar)) * (((sqrt(u_avg) * -0.019072663) / lt_0) - -0.42732748))) ^ (1.2448077 ^ ((sqrt(eps_1) * a_1) * 0.07756946))",
        "complexity": 40,
        "loss": 0.0015706882,
        "model_type": "flame_classifier",
        "target_variable": "alpha",
        "timestamp": "20251010_212538"
    },
    "20251011_002313_5y7mPP": {
        "equation": "((-4.4448166 / lt_1) + (((((ar + 52.32463) * phi) + ((abs((0.31438735 / lt_1) - (((phi * 60.045013) + -26.869356) / u_avg)) - (a_1 * 0.00877954)) * eps_0)) * -22.093985) + 3580.742)) * phi",
        "complexity": 34,
        "loss": 1189.238,
        "model_type": "flame_surrogate",
        "target_variable": "peak_temp",
        "timestamp": "20251011_002313"
    },
    "20251011_004118_5J9e4I": {
        "equation": "abs((((19.618101 / (eps_0 ^ ar)) + abs(((10227.975 / ar) - -747.21985) * (((u_avg / (eps_1 ^ 2.849905)) - (phi * 1.3682126)) * phi))) * phi) * (eps_0 + ((-117.0736 / a_1) + -1.2461119)))",
        "complexity": 35,
        "loss": 758035.9,
        "model_type": "flame_surrogate",
        "target_variable": "pressure_drop",
        "timestamp": "20251011_004118"
    },
    "20251011_010719_oVKpBh": {
        "equation": "(((0.121049255 - (u_avg * ((a_1 * -3.095461e-5) - 0.030999534))) / phi) + -0.20286718) / sqrt(eps_1 / (0.07169062 - abs((0.0684803 - lt_1) * ((ar * (0.27696922 / phi)) - ((eps_1 + phi) / u_avg)))))",
        "complexity": 35,
        "loss": 2.2367136e-05,
        "model_type": "flame_surrogate",
        "target_variable": "x_peak_temp",
        "timestamp": "20251011_010719"
    },
    "20251011_012252_mQd6jn": {
        "equation": "(((((u_avg - 0.074130625) + (((0.6118055 - phi) * eps_0) * 1.218382)) / ar) - 0.08245268) * ((4.354683 ^ phi) - (((u_avg / (phi ^ 6.6918902)) / ((lt_1 / a_1) / 1.9612003e-6)) + -3.6320436))) + 0.36264047",
        "complexity": 35,
        "loss": 0.023053152,
        "model_type": "flame_surrogate",
        "target_variable": "outlet_velocity",
        "timestamp": "20251011_012252"
    },
    "20251011_014513_z5mtkr": {
        "equation": "tanh(sqrt(ar * (0.033107847 / sqrt(u_avg))) - (abs((u_avg / (phi - ((tanh(lt_1 / -0.33472285) + 0.530116) / (eps_1 ^ 0.110800155)))) + ((phi - (lt_1 * 7.651003)) / -0.18806067)) * 0.014762474))",
        "complexity": 34,
        "loss": 0.00013471884,
        "model_type": "flame_surrogate",
        "target_variable": "radiative_efficiency",
        "timestamp": "20251011_014513"
    },
    "20251011_020310_HcETM0": {
        "equation": "((((10.657699 / lt_0) / eps_1) + (((1907.9698 - ((u_avg * ((-1.929231 / lt_1) - ((242.21515 / ar) / ar))) / (eps_1 ^ (eps_0 - -3.15043)))) + (-9.665854 / lt_0)) / eps_0)) + -2365.6792) * u_avg",
        "complexity": 35,
        "loss": 262663.34,
        "model_type": "no_flame_surrogate",
        "target_variable": "pressure_drop",
        "timestamp": "20251011_020310"
    },
    "20251011_022928_p5nFcU": {
        "equation": "tanh((0.52097124 / ((ar + ((((((lt_1 / eps_1) - 0.0248695) / (eps_0 + -0.23611252)) * (-0.55701584 / eps_1)) - k_0) * 0.022242285)) - 0.7066063)) - (-0.17581312 - (ar * -0.018146198))) * (u_avg + -0.0050839707)",
        "complexity": 34,
        "loss": 0.000109852124,
        "model_type": "no_flame_surrogate",
        "target_variable": "outlet_velocity",
        "timestamp": "20251011_022928"
    }
}

def evaluate_equation(equation_str: str, variables: Dict[str, float]) -> float:
    """
    Safely evaluate a PySR equation string with given variables.
    
    Args:
        equation_str: The equation string from PySR
        variables: Dictionary mapping variable names to values
    
    Returns:
        The evaluated result
    """
    # Replace PySR operators and functions with Python/numpy equivalents
    equation_str = equation_str.replace("^", "**")  # Power operator
    equation_str = equation_str.replace("tanh", "np.tanh")
    equation_str = equation_str.replace("sqrt", "np.sqrt")
    equation_str = equation_str.replace("log", "np.log")
    equation_str = equation_str.replace("exp", "np.exp")
    equation_str = equation_str.replace("abs", "np.abs")
    
    # Create a safe namespace for evaluation
    namespace = {"np": np, "__builtins__": {}}
    namespace.update(variables)
    
    try:
        return eval(equation_str, namespace)
    except Exception as e:
        raise ValueError(f"Error evaluating equation '{equation_str}': {e}")


def predict_alpha_v2sJPa(variables: Dict[str, float]) -> float:
    """
    Predict alpha using flame_classifier model.
    
    Target Variable: alpha
    Equation: (tanh((((phi * phi) / 0.1555563) ^ ar) * (eps_1 / (u_avg * 0.5385095))) ^ ((a_1 * ((phi * 1.3230652) - ar)) * (((sqrt(u_avg) * -0.019072663) / lt_0) - -0.42732748))) ^ (1.2448077 ^ ((sqrt(eps_1) * a_1) * 0.07756946))
    Complexity: 40
    Loss: 1.570688e-03
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted alpha
    """
    equation = "(tanh((((phi * phi) / 0.1555563) ^ ar) * (eps_1 / (u_avg * 0.5385095))) ^ ((a_1 * ((phi * 1.3230652) - ar)) * (((sqrt(u_avg) * -0.019072663) / lt_0) - -0.42732748))) ^ (1.2448077 ^ ((sqrt(eps_1) * a_1) * 0.07756946))"
    return evaluate_equation(equation, variables)

def predict_peak_temp_5y7mPP(variables: Dict[str, float]) -> float:
    """
    Predict peak_temp using flame_surrogate model.
    
    Target Variable: peak_temp
    Equation: ((-4.4448166 / lt_1) + (((((ar + 52.32463) * phi) + ((abs((0.31438735 / lt_1) - (((phi * 60.045013) + -26.869356) / u_avg)) - (a_1 * 0.00877954)) * eps_0)) * -22.093985) + 3580.742)) * phi
    Complexity: 34
    Loss: 1.189238e+03
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted peak_temp
    """
    equation = "((-4.4448166 / lt_1) + (((((ar + 52.32463) * phi) + ((abs((0.31438735 / lt_1) - (((phi * 60.045013) + -26.869356) / u_avg)) - (a_1 * 0.00877954)) * eps_0)) * -22.093985) + 3580.742)) * phi"
    return evaluate_equation(equation, variables)

def predict_pressure_drop_5J9e4I(variables: Dict[str, float]) -> float:
    """
    Predict pressure_drop using flame_surrogate model.
    
    Target Variable: pressure_drop
    Equation: abs((((19.618101 / (eps_0 ^ ar)) + abs(((10227.975 / ar) - -747.21985) * (((u_avg / (eps_1 ^ 2.849905)) - (phi * 1.3682126)) * phi))) * phi) * (eps_0 + ((-117.0736 / a_1) + -1.2461119)))
    Complexity: 35
    Loss: 7.580359e+05
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted pressure_drop
    """
    equation = "abs((((19.618101 / (eps_0 ^ ar)) + abs(((10227.975 / ar) - -747.21985) * (((u_avg / (eps_1 ^ 2.849905)) - (phi * 1.3682126)) * phi))) * phi) * (eps_0 + ((-117.0736 / a_1) + -1.2461119)))"
    return evaluate_equation(equation, variables)

def predict_x_peak_temp_oVKpBh(variables: Dict[str, float]) -> float:
    """
    Predict x_peak_temp using flame_surrogate model.
    
    Target Variable: x_peak_temp
    Equation: (((0.121049255 - (u_avg * ((a_1 * -3.095461e-5) - 0.030999534))) / phi) + -0.20286718) / sqrt(eps_1 / (0.07169062 - abs((0.0684803 - lt_1) * ((ar * (0.27696922 / phi)) - ((eps_1 + phi) / u_avg)))))
    Complexity: 35
    Loss: 2.236714e-05
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted x_peak_temp
    """
    equation = "(((0.121049255 - (u_avg * ((a_1 * -3.095461e-5) - 0.030999534))) / phi) + -0.20286718) / sqrt(eps_1 / (0.07169062 - abs((0.0684803 - lt_1) * ((ar * (0.27696922 / phi)) - ((eps_1 + phi) / u_avg)))))"
    return evaluate_equation(equation, variables)

def predict_outlet_velocity_mQd6jn(variables: Dict[str, float]) -> float:
    """
    Predict outlet_velocity using flame_surrogate model.
    
    Target Variable: outlet_velocity
    Equation: (((((u_avg - 0.074130625) + (((0.6118055 - phi) * eps_0) * 1.218382)) / ar) - 0.08245268) * ((4.354683 ^ phi) - (((u_avg / (phi ^ 6.6918902)) / ((lt_1 / a_1) / 1.9612003e-6)) + -3.6320436))) + 0.36264047
    Complexity: 35
    Loss: 2.305315e-02
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted outlet_velocity
    """
    equation = "(((((u_avg - 0.074130625) + (((0.6118055 - phi) * eps_0) * 1.218382)) / ar) - 0.08245268) * ((4.354683 ^ phi) - (((u_avg / (phi ^ 6.6918902)) / ((lt_1 / a_1) / 1.9612003e-6)) + -3.6320436))) + 0.36264047"
    return evaluate_equation(equation, variables)

def predict_radiative_efficiency_z5mtkr(variables: Dict[str, float]) -> float:
    """
    Predict radiative_efficiency using flame_surrogate model.
    
    Target Variable: radiative_efficiency
    Equation: tanh(sqrt(ar * (0.033107847 / sqrt(u_avg))) - (abs((u_avg / (phi - ((tanh(lt_1 / -0.33472285) + 0.530116) / (eps_1 ^ 0.110800155)))) + ((phi - (lt_1 * 7.651003)) / -0.18806067)) * 0.014762474))
    Complexity: 34
    Loss: 1.347188e-04
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted radiative_efficiency
    """
    equation = "tanh(sqrt(ar * (0.033107847 / sqrt(u_avg))) - (abs((u_avg / (phi - ((tanh(lt_1 / -0.33472285) + 0.530116) / (eps_1 ^ 0.110800155)))) + ((phi - (lt_1 * 7.651003)) / -0.18806067)) * 0.014762474))"
    return evaluate_equation(equation, variables)

def predict_pressure_drop_HcETM0(variables: Dict[str, float]) -> float:
    """
    Predict pressure_drop using no_flame_surrogate model.
    
    Target Variable: pressure_drop
    Equation: ((((10.657699 / lt_0) / eps_1) + (((1907.9698 - ((u_avg * ((-1.929231 / lt_1) - ((242.21515 / ar) / ar))) / (eps_1 ^ (eps_0 - -3.15043)))) + (-9.665854 / lt_0)) / eps_0)) + -2365.6792) * u_avg
    Complexity: 35
    Loss: 2.626633e+05
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted pressure_drop
    """
    equation = "((((10.657699 / lt_0) / eps_1) + (((1907.9698 - ((u_avg * ((-1.929231 / lt_1) - ((242.21515 / ar) / ar))) / (eps_1 ^ (eps_0 - -3.15043)))) + (-9.665854 / lt_0)) / eps_0)) + -2365.6792) * u_avg"
    return evaluate_equation(equation, variables)

def predict_outlet_velocity_p5nFcU(variables: Dict[str, float]) -> float:
    """
    Predict outlet_velocity using no_flame_surrogate model.
    
    Target Variable: outlet_velocity
    Equation: tanh((0.52097124 / ((ar + ((((((lt_1 / eps_1) - 0.0248695) / (eps_0 + -0.23611252)) * (-0.55701584 / eps_1)) - k_0) * 0.022242285)) - 0.7066063)) - (-0.17581312 - (ar * -0.018146198))) * (u_avg + -0.0050839707)
    Complexity: 34
    Loss: 1.098521e-04
    
    Args:
        variables: Dict with keys like {'phi', 'ar', 'u_avg', 'eps_0', 'eps_1', ...}
    
    Returns:
        Predicted outlet_velocity
    """
    equation = "tanh((0.52097124 / ((ar + ((((((lt_1 / eps_1) - 0.0248695) / (eps_0 + -0.23611252)) * (-0.55701584 / eps_1)) - k_0) * 0.022242285)) - 0.7066063)) - (-0.17581312 - (ar * -0.018146198))) * (u_avg + -0.0050839707)"
    return evaluate_equation(equation, variables)
