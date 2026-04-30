# EVOLVE-BLOCK-START
"""
FFT Complex

This task requires computing the N-dimensional Fast Fourier Transform (FFT) of a complex-valued matrix.  
The FFT is a mathematical technique that converts data from the spatial (or time) domain into the frequency domain, revealing both the magnitude and phase of the frequency components.  
The input is a square matrix of size n×n, where each element is a complex number containing both real and imaginary parts.  
The output is a square matrix of the same size, where each entry is a complex number representing a specific frequency component of the input data, including its amplitude and phase.  
This transformation is crucial in analyzing signals and data with inherent complex properties.

Input:
A complex-valued n×n matrix represented as an array of complex numbers.

Example input:
[[0.5+0.5j, 0.7+0.7j],
 [0.2+0.2j, 0.9+0.9j]]

Output:
An n×n matrix of complex numbers, where each element provides the frequency-domain information of the corresponding element in the input matrix.

Example output:
[[(1.8+0.3j), (-0.2+0.1j)],
 [(0.3-0.1j), (0.6+0.2j)]]

Category: signal_processing

OPTIMIZATION OPPORTUNITIES:
Consider these algorithmic improvements for substantial performance gains:
- Batch FFT processing: Process multiple matrices simultaneously to amortize setup costs
- Prime-factor algorithms: Use specialized algorithms for non-power-of-2 sizes (mixed-radix FFTs)
- In-place transforms: Reduce memory allocation overhead by computing FFT in-place
- Wisdom/planning optimization: Pre-compute optimal FFT strategies (pyfftw.interfaces)
- JIT compilation: Use JAX's jit-compiled FFT for significant speedups on repeated operations
- Real-optimized variants: If input has special structure, use rfft variants where applicable
- Memory layout optimization: Ensure optimal data layout for cache-friendly access patterns
- Specialized libraries: Consider mkl_fft, pyfftw, or JAX implementations vs scipy.fftpack
- Hardware acceleration: Leverage SIMD instructions and optimized BLAS implementations
- Algorithm selection: Choose optimal FFT variant based on input size and characteristics
- Multi-dimensional optimization: Optimize axis ordering for multi-dimensional transforms

This is the initial implementation that will be evolved by OpenEvolve.
The solve method will be improved through evolution.
"""
import logging
import numpy as np
import jax
import jax.numpy as jnp

_jit_fftn = jax.jit(jnp.fft.fftn)
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional

class FFTComplexScipyFFTpack:
    """
    Initial implementation of fft_cmplx_scipy_fftpack task.
    This will be evolved by OpenEvolve to improve performance and correctness.
    """
    
    @staticmethod
    def solve(problem):
        """
        Solve the fft_cmplx_scipy_fftpack problem.
        
        Args:
            problem: Dictionary containing problem data specific to fft_cmplx_scipy_fftpack
                   
        Returns:
            The solution in the format expected by the task
        """
        try:
            """
            Compute the N-dimensional FFT using a JIT-compiled JAX function.
            """
            return _jit_fftn(problem)
            
        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise e
    
    @staticmethod
    def is_solution(problem, solution):
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
            Check if the FFT solution is valid and optimal.

            A valid solution must match the reference implementation (numpy's FFT)
            within a small tolerance.

            :param problem: Input complex array.
            :param solution: Computed FFT result.
            :return: True if the solution is valid and optimal, False otherwise.
            """
            tol = 1e-6
            reference = np.fft.fftn(problem)
            error = np.linalg.norm(solution - reference) / (np.linalg.norm(reference) + 1e-12)
            if error > tol:
                logging.error(f"FFT solution error {error} exceeds tolerance {tol}.")
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
    return FFTComplexScipyFFTpack.solve(problem)

# EVOLVE-BLOCK-END

# Test function for evaluation
if __name__ == "__main__":
    # Example usage
    print("Initial fft_cmplx_scipy_fftpack implementation ready for evolution")
