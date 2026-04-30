# EVOLVE-BLOCK-START
"""
PolynomialReal Task:

Given a polynomial of degree n with all real roots, the task is to approximate the roots of the polynomial.
The goal is to compute the approximated roots so that each is within 0.001 of the true value—i.e., correct to at least three decimal places (any extra precision beyond three decimal places is not considered).
A given solution's distance is said to be the average absolute difference between the approximated roots and the true roots.

Input: A list of polynomial coefficients in descending order.

Example input:
[1.0, -3.54, 3.25, -1.17, 0.23],

(The polynomial is x^4 - 3.54x^3 + 3.25x^2 - 1.17x + 0.23)

Output: A list of approximated roots in descending order.

Example output:
[1.544, 1.022, -0.478, -1.273]

Category: numerical_methods

OPTIMIZATION OPPORTUNITIES:
Consider these algorithmic improvements for substantial performance gains:
- JAX JIT compilation: Try JAX for massive speedups on numerical operations - start with pure functions
- Companion matrix methods: Alternative eigenvalue-based approach to numpy.roots for better performance
- Iterative root-finding: Newton-Raphson, Durand-Kerner, or Aberth methods with better convergence
- Compilation techniques: JIT frameworks often require specific code structures (functions vs methods)
- Polynomial structure exploitation: Leverage properties like Chebyshev polynomials or specific forms
- Deflation techniques: Remove found roots to improve conditioning for remaining roots
- Initial guess optimization: Better starting points for iterative methods using root bounds
- Specialized algorithms: Research advanced root-finding algorithms
- Multiple precision: Use higher precision only when needed for better accuracy
- Vectorized operations: Process multiple polynomials or roots simultaneously
- Memory-efficient methods: Avoid unnecessary array copies and intermediate calculations
- Hybrid approaches: Combine different methods based on polynomial degree or properties
- Real-root isolation: Use Sturm sequences or Descartes' rule for real-only root finding

This is the initial implementation that will be evolved by OpenEvolve.
The solve method will be improved through evolution.
"""
import logging
import random
import numpy as np
from typing import Any, Dict, List, Optional
import jax
import jax.numpy as jnp

@jax.jit
def _solve_roots_jax(coefficients):
    """
    JIT-compiled function to find polynomial roots using JAX.
    This function leverages jax.numpy.roots which is a JIT-compatible
    implementation for finding roots based on the companion matrix eigenvalues.
    Assumes coefficients have no leading zeros.
    """
    # The problem guarantees all roots are real, so we can safely take the real part
    # to discard small imaginary parts arising from numerical precision errors.
    # strip_zeros=False is crucial for JIT compatibility as it ensures a static output shape.
    real_roots = jnp.real(jnp.roots(coefficients, strip_zeros=False))
    # Sort the roots in descending order.
    return jnp.sort(real_roots)[::-1]

class PolynomialReal:
    """
    Initial implementation of polynomial_real task.
    This will be evolved by OpenEvolve to improve performance and correctness.
    """
    
    def __init__(self):
        """Initialize the PolynomialReal."""
        pass
    
    def solve(self, problem):
        """
        Solve the polynomial_real problem using a JIT-compiled JAX function.
        
        Args:
            problem: A list of polynomial coefficients in descending order.
                   
        Returns:
            A list of real roots of the polynomial, sorted in decreasing order.
        """
        try:
            coeffs_np = np.array(problem, dtype=np.float64) # Use float64 for better precision

            # Handle edge cases for zero or constant polynomials
            # Find the first non-zero coefficient to determine the true degree
            first_nonzero_idx = 0
            for i, coeff in enumerate(coeffs_np):
                if abs(coeff) > 1e-9: # Check for non-zero with a small tolerance
                    first_nonzero_idx = i
                    break
            else: # All coefficients are effectively zero
                return []
            
            trimmed_coeffs = coeffs_np[first_nonzero_idx:]

            if len(trimmed_coeffs) <= 1:
                # If after trimming, only one coefficient remains (e.g., [5.0]),
                # it's a constant non-zero polynomial, which has no roots.
                return []

            # Use a JIT-compiled function for root finding.
            # Convert trimmed_coeffs to jax.numpy array with float64.
            computed_roots = _solve_roots_jax(jnp.array(trimmed_coeffs, dtype=jnp.float64))
            
            # block_until_ready() ensures the asynchronous JAX computation is finished.
            return computed_roots.block_until_ready().tolist()
            
        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise e
    
    def is_solution(self, problem, solution):
        """
        Check if the provided solution is valid.
        
        Args:
            problem: The original problem
            solution: The proposed solution
                   
        Returns:
            True if the solution is valid, False otherwise
        """
        try:
            """
            Check if the polynomial root solution is valid and optimal.

            A valid solution must:
            1. Match the reference solution (computed using np.roots) within a small tolerance
            2. Be sorted in descending order

            :param problem: A list of polynomial coefficients (real numbers) in descending order.
            :param solution: A list of computed real roots.
            :return: True if the solution is valid and optimal, False otherwise.
            """
            coefficients = problem
            reference_roots = np.roots(coefficients)
            reference_roots = np.real(reference_roots) # Problem states all roots are real, so np.real is sufficient
            reference_roots = np.sort(reference_roots)[::-1]
            candidate = np.array(solution)
            reference = np.array(reference_roots)
            tol = 1e-6
            error = np.linalg.norm(candidate - reference) / (np.linalg.norm(reference) + 1e-12)
            if error > tol:
                logging.error(f"Polynomial real solution error {error} exceeds tolerance {tol}.")
                return False
            return True
            
        except Exception as e:
            logging.error(f"Error in is_solution method: {e}")
            return False

def run_solver(problem):
    """
    Main function to run the solver.
    This function is used by the evaluator to test the evolved solution.
    
    Args:
        problem: The problem to solve
        
    Returns:
        The solution
    """
    solver = PolynomialReal()
    return solver.solve(problem)

# EVOLVE-BLOCK-END

# Test function for evaluation
if __name__ == "__main__":
    # Example usage
    print("Initial polynomial_real implementation ready for evolution")
