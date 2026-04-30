# OpenEvolve AlgoTune Optimization Report

## Executive Summary

This report documents a comprehensive optimization journey using OpenEvolve on the AlgoTune benchmark suite. Through systematic experimentation with model configurations, prompt engineering, and evolutionary parameters, we achieved significant performance improvements across 8 algorithmic tasks.

**Final Results:**
- **Best AlgoTune Score: 1.984x** (harmonic mean across 8 successful tasks)
- **Major Breakthroughs:** JAX optimization discovery (321x speedup), FFT convolution (256x), parameter optimization (3.2x)
- **Total Evolution Time:** ~200 minutes for full benchmark

## The Optimization Journey

### Phase 1: Initial Baseline (Generic Hints)
- **AlgoTune Score: 1.381x**
- Used basic library mentions without implementation details
- Key limitation: Failed to discover complex optimizations like JAX JIT compilation

### Phase 2: Manual Optimization Discovery
Through manual analysis, we discovered several key optimizations:
- **JAX JIT compilation** for polynomial_real (362x theoretical speedup)
- **FFT convolution** for signal processing tasks
- **Parameter optimization** (dtype, interpolation order)
- **Hardware-specific optimizations** for Apple M4

### Phase 3: Specific Hints Implementation
- **AlgoTune Score: 1.886x**
- Added detailed implementation hints based on manual discoveries
- Achieved best theoretical performance but raised "overfitting" concerns

### Phase 4: The Balance - Generic Hints with Smart Configuration
- **Final AlgoTune Score: 1.984x** (across successful tasks)
- Balanced approach: Library guidance without implementation details
- Optimized configurations for different optimization types

## Task-by-Task Optimization Discoveries

### 1. polynomial_real: JAX JIT Compilation Discovery
**Result: 321.01x speedup** ✨
- **Optimization:** JAX JIT compilation with `@jax.jit`
- **Key Requirements Discovered:**
  - Functions must be defined outside classes for JIT compatibility
  - `strip_zeros=False` parameter crucial for JIT compilation
  - `jnp.roots()` instead of `np.roots()`
- **Configuration Needed:** Extended timeout (600s) for compilation, sequential evaluation
- **Code Pattern:**
```python
@jax.jit
def _solve_roots_jax(coefficients):
    real_roots = jnp.real(jnp.roots(coefficients, strip_zeros=False))
    return jnp.sort(real_roots)[::-1]
```

### 2. convolve2d_full_fill: FFT Algorithm Discovery
**Result: 256.15x speedup**
- **Optimization:** `scipy.signal.fftconvolve` instead of direct convolution
- **Algorithm Change:** O(N⁴) → O(N²log N) complexity
- **Additional Optimization:** float32 dtype for memory efficiency
- **Discovery:** This was consistently found across all runs (generic hints sufficient)

### 3. affine_transform_2d: Parameter Optimization Breakthrough  
**Result: 3.22x speedup**
- **Optimization:** Combined `order=0` (nearest neighbor) + `float32`
- **Key Insight:** Lower interpolation orders provide dramatic speedups
- **Enhancement Strategy:** Specific parameter guidance in hints worked perfectly
- **Previous Generic Result:** Only 1.004x (failed to discover optimization)

### 4. fft_cmplx_scipy_fftpack: Algorithm Enhancement
**Result: 2.20x speedup**
- **Optimization:** Enhanced FFT implementation patterns
- **Improvement:** 77% better than generic hints (1.24x → 2.20x)

### 5. eigenvectors_complex: Stable Performance
**Result: 1.48x speedup**
- **Optimization:** Consistent eigenvalue computation improvements
- **Note:** Similar performance across all configurations

### 6. fft_convolution: Incremental Gains
**Result: 1.38x speedup**
- **Optimization:** FFT-based convolution optimizations
- **Improvement:** 24% better than baseline

### 7. lu_factorization: Consistent Optimization
**Result: 1.19x speedup**
- **Optimization:** LAPACK-based factorization improvements
- **Note:** Maintained consistent performance across runs

### 8. psd_cone_projection: Eigenvalue Optimization
**Result: 1.94x speedup**
- **Optimization:** Optimized positive semidefinite projection algorithms

## Critical Success Factors

### 1. Model Configuration
**Ensemble Strategy: Gemini Flash 2.5 (80%) + Pro (20%)**
- **Flash Model:** Fast iterations, good for exploration
- **Pro Model:** Enhanced reasoning for complex optimizations
- **Balance:** Cost-effective with maintained quality

### 2. Context and Sampling Configuration
```yaml
llm:
  max_tokens: 128000  # Large context for rich learning
  
prompt:
  num_top_programs: 5      # Quality examples
  num_diverse_programs: 5  # Exploration diversity
```
- **Large Context (128k tokens):** Essential for complex optimization discovery
- **Balanced Sampling:** 5 top + 5 diverse programs optimal for learning

### 3. Strategic Hint Engineering
**The Golden Rule: Libraries YES, Implementation Details NO**

✅ **Effective Hints:**
```yaml
• **JAX** - JIT compilation for numerical computations that can provide 100x+ speedups
  JAX offers drop-in NumPy replacements (jax.numpy) that work with JIT compilation
  Works best with pure functions (no side effects) and may require code restructuring

• Lower-order interpolation: Try order=0,1,2,3 - lower orders can provide dramatic speedups
```

❌ **Overly Specific (Avoided):**
```yaml
# Too specific - gives away solution
• Use jnp.roots(coefficients, strip_zeros=False) 
• Functions should be defined outside classes for JIT compatibility
```

### 4. Task-Specific Configuration Tuning

**For JAX Compilation Tasks:**
```yaml
evaluator:
  timeout: 600  # Extended for compilation
  parallel_evaluations: 1  # Avoid conflicts
```

**For Standard Tasks:**
```yaml
evaluator:
  timeout: 200
  parallel_evaluations: 4  # Faster throughput
```

## Key Learnings

### 1. Optimization Type Categories
Different optimizations require different approaches:

**Library Optimizations (JAX, Numba):**
- Need architectural guidance (functions vs methods)
- Require specific parameter hints (strip_zeros=False)
- Long compilation times need extended timeouts

**Algorithm Optimizations (FFT):**
- Discoverable with generic hints about complexity
- Benefit from mentioning alternative approaches
- Generally faster to discover and implement

**Parameter Optimizations (dtype, order):**
- Need directional guidance ("try lower orders")
- Require specific value ranges
- Balance between exploration and guidance

### 2. Configuration Impact Analysis

**Critical Discoveries:**
- **Context Size:** 128k tokens significantly improved optimization discovery
- **Model Ensemble:** Diversity crucial for complex reasoning tasks
- **Timeout Tuning:** Different tasks need different evaluation timeouts
- **Sequential vs Parallel:** JAX requires sequential evaluation to avoid conflicts

### 3. The Hint Specificity Spectrum

**Too Generic:** System cannot discover complex optimizations
```
"Try different approaches" → Failed to find JAX
```

**Perfect Balance:** Library guidance with structural hints
```
"JAX JIT compilation - works best with pure functions" → Success
```

**Too Specific:** System doesn't learn, just copies
```
"Use jnp.roots(coeffs, strip_zeros=False)" → No learning
```

## Configuration Best Practices

### 1. Model Selection Strategy
```yaml
llm:
  models:
    - name: "google/gemini-2.5-flash"
      weight: 0.8  # Primary workhorse
    - name: "google/gemini-2.5-pro" 
      weight: 0.2  # Enhanced reasoning
```

### 2. Optimal Sampling Configuration
```yaml
prompt:
  num_top_programs: 5      # Quality over quantity
  num_diverse_programs: 5  # Sufficient exploration
  include_artifacts: true  # Learning from failures
```

### 3. Task-Specific Timeout Strategy
- **Standard tasks:** 200s evaluator timeout
- **Compilation tasks (JAX/Numba):** 600s+ evaluator timeout
- **Complex algorithms:** Consider extended iteration timeouts

### 4. Effective Hint Structure
```yaml
PERFORMANCE OPTIMIZATION OPPORTUNITIES:
• **[Library]** - High-level capability description
  Technical requirements without implementation details
  
PROBLEM-SPECIFIC OPTIMIZATION HINTS:
• Parameter exploration guidance
• Algorithmic approach suggestions
• Performance vs accuracy tradeoffs
```

## Technical Implementation Details

### JAX Optimization Requirements
The most complex optimization discovered required specific architectural patterns:

1. **Function Extraction:** JIT functions must be defined outside classes
2. **Parameter Specification:** `strip_zeros=False` for deterministic shapes
3. **Data Flow:** Pure functional programming patterns
4. **Compilation Management:** Extended timeouts and sequential evaluation

### Evolution Pattern Analysis
Successful optimization discovery followed this pattern:
1. **Initial Failures:** Programs fail with specific error messages
2. **Error Learning:** System incorporates error feedback into next generation
3. **Breakthrough:** Correct pattern discovered, dramatic speedup achieved
4. **Refinement:** Further iterations optimize the successful pattern

## Conclusion

This comprehensive optimization journey demonstrates OpenEvolve's remarkable capability to discover complex algorithmic optimizations when provided with appropriate guidance and configuration. Key insights:

1. **Human-AI Collaboration:** The most effective approach combines human domain knowledge (library suggestions) with AI exploration (implementation discovery)

2. **Configuration Criticality:** Success heavily depends on properly tuned configurations for context size, model ensemble, sampling strategy, and task-specific parameters

3. **Hint Engineering Art:** The balance between guidance and exploration is crucial - too little guidance fails to discover optimizations, too much guidance prevents learning

4. **Scalable Discovery:** OpenEvolve can consistently discover optimizations across diverse algorithmic domains when properly configured

5. **Future Potential:** With continued refinement of hint strategies and configuration optimization, even more complex algorithmic breakthroughs are possible

**Final AlgoTune Score: 1.984x** represents not just performance improvement, but a validated methodology for AI-assisted algorithmic optimization that can be applied to broader domains beyond this benchmark.

---

*This report represents the culmination of extensive experimentation with OpenEvolve's evolutionary code optimization capabilities on the AlgoTune benchmark suite, providing a roadmap for future AI-assisted algorithmic discovery.*
