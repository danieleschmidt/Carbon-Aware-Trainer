# Model Performance Analysis Report

## Input Size Scaling

| Input Size | Processing Time (s) | Throughput (points/s) |
|------------|--------------------|-----------------------|
| 50 | 0.005 | 9169.3 |
| 100 | 0.011 | 9503.0 |
| 300 | 0.030 | 9855.1 |
| 500 | 0.050 | 9902.4 |

## Forecast Horizon Scaling

| Horizon (hours) | Processing Time (s) | Predictions/s |
|-----------------|--------------------|--------------|
| 12 | 0.017 | 688.9 |
| 24 | 0.018 | 1365.5 |
| 48 | 0.017 | 2764.2 |

## Computational Complexity Analysis

**Estimated Complexity**: O(1) - Constant time

**Linear Fit**: R² = 1.000
**Performance Equation**: time = 0.000100 × input_size + 0.000

## Recommendations

- ✅ Model scales well to large input sizes
- ✅ Excellent computational complexity for scaling
