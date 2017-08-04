import inspect, os

from prometheus_client  import Counter as PromCounter
from prometheus_client  import Summary as PromSummary

def get_name_by_file():
    filename = inspect.getframeinfo(inspect.currentframe().f_back.f_back).filename
    return os.path.basename(filename).split('.')[0]

def rule_timer(rule):
    name = 'rule_' + get_name_by_file()
    rule_summary = PromSummary(name, name + ' timer')
    @rule_summary.time()
    def decorated(*args, **kwds):
        return rule(*args, **kwds)
    return decorated

def build_errors(build):
    name = 'build_command_' + get_name_by_file()
    build_counter = PromCounter(name, name + ' errors')
    @build_counter.count_exceptions()
    def decorated(*args, **kwds):
        return build(*args, **kwds)
    return decorated

def react_results(react):
    name         = 'react_command_' + get_name_by_file()
    success_name = 'success_' + name
    fail_name    = 'fail_' + name
    success_counter = PromCounter(success_name, success_name)
    fail_counter    = PromCounter(fail_name, fail_name)
    react_errors    = PromCounter(name, name + 'errors')
    @react_errors.count_exceptions()
    def decorated(*args, **kwds):
        result = react(*args, **kwds)
        if result is None:
            return
        success, fail = result
        success_counter.inc(success)
        fail_counter.inc(fail)
        return success, fail
    return decorated

