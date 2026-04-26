# manages the debug stack depth and formatting for pyFLD
# also create warning class

# import warnings

_DEBUG_DEPTH = 0

def push():
    global _DEBUG_DEPTH
    _DEBUG_DEPTH += 1

def pop():
    global _DEBUG_DEPTH
    _DEBUG_DEPTH = max(0, _DEBUG_DEPTH - 1)

def depth():
    return _DEBUG_DEPTH

def format_debug(prefix='', msg='', indent=0, log_char="| "):
    """Formats a debug message with indentation."""
    return log_char*indent + prefix + ": " + msg

# class PyFLDWarning(UserWarning): pass
# def format_warning(prefix='', msg='', indent=0):
#     warnings.warn(format_debug(prefix, msg, indent, log_char="* "), PyFLDWarning)
def format_warning(prefix='', msg='', indent=0):
    print(format_debug(prefix, msg, indent, log_char="* "))