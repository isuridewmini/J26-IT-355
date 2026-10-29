import 'dart:math' as math;

/// erf via the Numerical Recipes Chebyshev fit (fractional error < 1.2e-7).
double erf(double x) {
  final z = x.abs();
  final t = 1 / (1 + 0.5 * z);
  final r = t *
      math.exp(-z * z -
          1.26551223 +
          t *
              (1.00002368 +
                  t *
                      (0.37409196 +
                          t *
                              (0.09678418 +
                                  t *
                                      (-0.18628806 +
                                          t *
                                              (0.27886807 +
                                                  t *
                                                      (-1.13520398 +
                                                          t *
                                                              (1.48851587 +
                                                                  t *
                                                                      (-0.82215223 +
                                                                          t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}

/// Exact (erf-based) GELU, as torch.nn.GELU() uses.
double gelu(double x) => 0.5 * x * (1 + erf(x / math.sqrt2));

double sigmoid(double x) => 1 / (1 + math.exp(-x));
