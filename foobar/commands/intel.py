import random
from ..prometheus import build_errors, react_results

@build_errors
def build():
    if random.randint(0, 1) == 0:
        raise ValueError('bad luck')
    react([0] * random.randint(1, 5))
    return 'command'

@react_results
def react(product):
    if len(product) == 5:
        raise ValueError('5')
    half = len(product) / 2
    return half, len(product) - half
