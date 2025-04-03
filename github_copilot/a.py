

def fibonacci_series(n):
    """Generate a Fibonacci series up to n terms."""
    series = []
    a, b = 0, 1
    for _ in range(n):
        series.append(a)
        a, b = b, a + b
    return series

# Example usage:
print(fibonacci_series(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(fibonacci_series(0))   # Output: []