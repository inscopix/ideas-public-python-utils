import importlib
import json
import logging
import os

from ideas.utils import split_tool_command

logger = logging.getLogger(__name__)


def run_tool(tool_command: str, inputs_json_file: str = "inputs.json"):
    inputs_json_file = "inputs.json"
    with open(inputs_json_file, "r") as f:
        tool_inputs = json.load(f)

    module_loc = "toolbox.tools"
    module_name, func_name = split_tool_command(tool_command)

    module_name = module_name.replace(".py", "")
    module_name = module_name.replace(os.path.sep, ".")

    # remove any leading or trailing dots
    clean_loc = module_loc.strip(".")
    # Import the module and function
    module = importlib.import_module(f"{clean_loc}.{module_name}")
    func = getattr(module, func_name)

    logger.info("Calling function with arguments:")
    logger.info(json.dumps(tool_inputs, indent=2))

    func(**tool_inputs)
