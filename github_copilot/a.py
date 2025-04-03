from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_number(n):
    """Return the nth Fibonacci number."""
    if n < 2:
        return n
    return fibonacci_number(n - 1) + fibonacci_number(n - 2)

def fibonacci_series(n):
    """Generate a Fibonacci series up to n terms using lru_cache."""
    return [fibonacci_number(i) for i in range(n)]

# Example usage:
print(fibonacci_series(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print(fibonacci_series(0))   # Output: []