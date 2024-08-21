import logging
import os
import sys


def split_tool_command(tool_key: str) -> (str, str):
    """utility that converts a tool command to the name
    of the function and the containing module"""

    fragments = tool_key.split("__")

    func_name = fragments[-1]

    module_path = os.path.sep.join(fragments[0:-1]) + ".py"
    return module_path, func_name



def _set_up_logger():
    """Set up logger for IDEAS


    this code moved from old-style IDEAS toolboxes.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(lambda r: r.levelno <= logging.ERROR)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
