import random, time
from ..prometheus import rule_timer

@rule_timer
def decide():
    time.sleep(random.randint(1, 4))
    return 'rule'
