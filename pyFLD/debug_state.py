# manages the debug stack depth and formatting for pyFLD
# also create warning class

# import warnings
import functools

_DEBUG_DEPTH = 0

def push():
    global _DEBUG_DEPTH
    _DEBUG_DEPTH += 1

def pop():
    global _DEBUG_DEPTH
    _DEBUG_DEPTH = max(0, _DEBUG_DEPTH - 1)

def depth():
    return _DEBUG_DEPTH

def keep_depth(func):
    """Restore the debug depth when func exits, also through an exception or an early
    return, so that a push without its pop cannot shift all later messages."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _DEBUG_DEPTH
        entry_depth = _DEBUG_DEPTH
        try: return func(*args, **kwargs)
        finally: _DEBUG_DEPTH = entry_depth
    return wrapper

def keep_depth_in(cls):
    """Apply keep_depth to every method of cls that opens a debug region, i.e. whose
    code calls _push_region. Used by fluids._GenericFluid for itself and all subclasses."""
    for name, attr in list(vars(cls).items()):
        if hasattr(attr, '__code__') and '_push_region' in attr.__code__.co_names:
            setattr(cls, name, keep_depth(attr))
    return cls

def format_debug(prefix='', msg='', indent=0, log_char="| "):
    """Formats a debug message with indentation."""
    return log_char*indent + prefix + ": " + msg

# class PyFLDWarning(UserWarning): pass
# def format_warning(prefix='', msg='', indent=0):
#     warnings.warn(format_debug(prefix, msg, indent, log_char="* "), PyFLDWarning)
def format_warning(prefix='', msg='', indent=0):
    print(format_debug(prefix, msg, indent, log_char="* "))

def format_info(prefix='', msg='', indent=0):
    print(format_debug(prefix, msg, indent, log_char="i "))