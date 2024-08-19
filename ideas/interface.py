import importlib
import json
import logging
import os

from beartype import beartype
from beartype.typing import Any, Dict, List
from ideas.data_model_v2 import make_output_manifest
from ideas.utils import (
    _set_up_logger,
    get_file_size,
    rename_file_using_rule,
    sh_2_json,
    split_tool_command,
)
from tabulate import tabulate

_set_up_logger()
logger = logging.getLogger()


@beartype
def rename_outputs(
    tool_command: str,
    inputs_json_file: str,
    manifest_file: str,
) -> None:
    """this function renames output files and modifies
    the output manifest to reflect renaming"""

    logger.info("Attempting to rename outputs...")

    rename_rules_file = f"/ideas/info/rename_rules_{tool_command}.json"
    results_file = f"/ideas/info/results_{tool_command}.json"

    if not os.path.exists(rename_rules_file) or not os.path.exists(
        results_file
    ):
        # no rename rules defined, abort
        logger.warning("No rename rules or results defined.")
        return

    # read JSON files
    with open(manifest_file, "r") as file:
        data = json.load(file)

    with open(rename_rules_file, "r") as file:
        rules = json.load(file)

    with open(results_file, "r") as file:
        results = json.load(file)

    if tool_command not in rules:
        # no rename rules for this tool, abort
        logger.error("No rename rules defined for this tool.")
        return

    # get a list of all file names
    # we have to traverse the output manifest
    # instead of doing something else
    # because this is the only authoritative source
    # of output files
    # so we go over the output manifest and extract
    # all non-source files
    file_paths = _get_files(data)

    # now go over each file and see if we have to rename it
    for file_path in file_paths:
        basename = os.path.basename(file_path)
        file_key, _ = os.path.splitext(basename)
        dirname = os.path.dirname(file_path)

        rule = None
        is_multiple = False
        index = None
        for key in rules[tool_command].keys():
            if key == file_key:
                # key matches file key exactly -> single file output
                rule = rules[tool_command][key]
                break
            elif file_key.startswith(f"{key}."):
                # file key starts with key. -> multi-file output
                try:
                    # verify the end of the file key is an int
                    index = int(file_key[len(key) + 1 :])
                    rule = rules[tool_command][key]
                    logger.info(
                        f"Got index for multi-file output: {file_key}, {index}"
                    )
                except ValueError:
                    logger.warn(
                        f"Failed to get index for multi-file output: {file_key}, {file_key[len(key)+1:]}"
                    )

                # validate this file is marked as a multi-file output in results info
                for result in results:
                    if key == result["key"]:
                        is_multiple = result["multiple"]
                        break

                if not is_multiple:
                    logger.warn(
                        "Output file with index is not marked as a multi-file output. Skipping rename"
                    )
                    index = None
                    rule = None

                break

        if rule is None:
            logger.info(f"No rename rules for {tool_command}/{file_key}")
            # likely there is no rule for this, for
            # whatever reason. skip
            continue

        pattern = rule["pattern"]

        if len(pattern) == 0:
            # this pattern makes no sense, and the
            # only logical interpretation is to do nothing
            logger.error(f"Zero-length rename pattern for {file_path}")
            continue

        if "custom_string" in pattern:
            pattern[pattern.index("custom_string")] = rule["custom_string"]

        with open(inputs_json_file, "r") as file:
            inputs = json.load(file)

        new_name = rename_file_using_rule(
            original_name=basename,
            options=pattern,
            inputs=inputs,
            index=index,
        )
        new_name_with_path = os.path.join(dirname, new_name)

        # rename the file
        logger.info(f"Renaming {basename} --> {new_name}")
        os.rename(file_path, new_name_with_path)

        # modify the entry in the output manifest
        for group in data["groups"]:
            for file in group["files"]:
                if file["file_name"] == basename:
                    file["file_name"] = new_name
                    file["file_path"] = file["file_path"].replace(
                        basename, new_name
                    )

                if "preview" in file:
                    for preview in file["preview"]:
                        if preview["name"] == basename:
                            preview["name"] = new_name
                            preview["file_path"] = preview[
                                "file_path"
                            ].replace(basename, new_name)

    # save the modified output_manifest
    with open(
        manifest_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


@beartype
def write_output_metadata(
    metadata_file: str = "output_metadata.json",
    manifest_file: str = "output_manifest.json",
):
    """Write output_metadata.json if the tool wrote output metadata

    If the tool created a file called output_metadata.json that
    contains key-value pairs inside a key that corresponds to a
    file key, then wrap the whole thing in containers so that
    the BE can parse this correctly
    ."""

    if not os.path.exists(metadata_file):
        # nothing to do
        return

    with open(manifest_file, "r") as f:
        output_manifest = json.load(f)

    with open(metadata_file, "r") as f:
        output_metadata = json.load(f)

    mapped_output_metadata = {
        "schema_version": "2.0.0",
        "group_metadata": [],
        "series_metadata": [],
        "file_metadata": [],
    }

    # track group ids to prevent duplicate groups from appearing in metadata manifest
    group_ids = []
    for file_key, file_metadata in output_metadata.items():
        for group in output_manifest["groups"]:
            if group["group_id"] not in group_ids:
                mapped_output_metadata["group_metadata"].append(
                    dict(
                        group_id=group["group_id"],
                        add={
                            "ideas": {"dataset": {"group_type": "tool_output"}}
                        },
                        inherit=[],
                    )
                )
                group_ids.append(group["group_id"])

            for file in group["files"]:
                # in the case of multiple files in one output the key will be
                # file["file_key"].x, where x is the index of the file in the output.
                if (
                    file["file_key"] == file_key
                    or os.path.splitext(file["file_name"])[0] == file_key
                ):
                    mapped_output_metadata["file_metadata"].append(
                        dict(
                            file_id=file["file_id"],
                            add=dict(ideas=file_metadata),
                            inherit=[],
                        )
                    )

    with open(metadata_file, "w") as f:
        json.dump(mapped_output_metadata, f, indent=4)


@beartype
def _get_files(manifest_data: Dict[str, Any]) -> List[str]:
    """This function returns a list of all files"""
    file_paths = []
    for group in manifest_data["groups"]:
        for file in group["files"]:
            if file["file_category"] == "source":
                continue

            file_paths.append(file["file_path"])

            if "preview" in file:
                for preview in file["preview"]:
                    file_paths.append(preview["file_path"])

    # file_paths may contain duplicates, so let's fix that
    return list(set(file_paths))


@beartype
def run_tool(
    *,
    tool_command: str,
    toolbox_info_file: str = "/ideas/info/toolbox_info.json",
    inputs_sh_file: str = "inputs.sh",
    inputs_json_file: str = "inputs.json",
    manifest_file: str = "output_manifest.json",
    metadata_file: str = "output_metadata.json",
    module_loc: str = "toolbox.tools",
) -> None:
    """ "
    Runs a specified tool from a toolbox.

    This function imports and runs a tool from a toolbox created using the
    ideas-toolbox-creator tool. It also handles the generation of output
    manifests and metadata, and can rename outputs based on rules.

    Parameters:
    tool_command (str): The tool command to run.
        Formatted as <module_name>__<function_name>, where <module_name>
        is the name of the module where the tool function exists,
        and <function_name> is the name of the tool function in the module.
    toolbox_info_file (str, optional): The path to the toolbox_info file.
        Defaults to "/ideas/info/toolbox_info.json".
    inputs_sh_file (str, optional): The path to the inputs shell file.
        Defaults to "inputs.sh". Contains the same information as inputs.json,
    inputs_json_file (str, optional): The path to the inputs JSON file.
        Defaults to "inputs.json". Contains the same information as inputs.sh.
    manifest_file (str, optional): The path to the output manifest file.
        Defaults to "output_manifest.json".
    metadata_file (str, optional): The path to the output metadata file.
        Defaults to "output_metadata.json".
    module_loc (str, optional): The location of the module. in the repository
        Defaults to "toolbox.tools".

    Returns:
    None
    """

    module_name, func_name = split_tool_command(tool_command)

    module_name = module_name.replace(".py", "")
    module_name = module_name.replace(os.path.sep, ".")

    # remove any leading or trailing dots
    clean_loc = module_loc.strip(".")
    # Import the module and function
    module = importlib.import_module(f"{clean_loc}.{module_name}")
    func = getattr(module, func_name)

    logging.info(f"Successfully imported {func_name} from {module}")
    if not os.path.exists(inputs_sh_file) and not os.path.exists(
        inputs_json_file
    ):
        raise FileNotFoundError(
            f"""
    The inputs.sh and inputs.json files were not 
    found in the current directory ({os.getcwd()}).
    Cannot proceed. """
        )

    if not os.path.exists(toolbox_info_file):
        raise FileNotFoundError(
            f"""
    The toolbox_info.json files was not found. Expected it 
    to exist at {toolbox_info_file} """
        )

    with open(toolbox_info_file, "r") as f:
        info = json.load(f)

    tool_found = False
    tool_key = None
    for tool in info["tools"]:
        if tool["command"] == f"{tool_command}.sh":
            tool_found = True
            tool_key = tool["key"]
            break

    if not tool_found:
        raise AttributeError(
            f"""Tool command '{tool_command}.sh' not found 
            in toolbox_info.json."""
        )

    if not os.path.exists(inputs_json_file):
        sh_2_json(inputs_sh_file, inputs_json_file)

    with open(inputs_json_file, "r") as f:
        tool_inputs = json.load(f)

    # call the function
    logger.info("Calling function with arguments:")
    logger.info(json.dumps(tool_inputs, indent=2))
    func(**tool_inputs)

    # now make the output manifest

    # check if we have annotations
    annotations_file = f"/ideas/info/results_{tool_command}.json"
    if os.path.exists(annotations_file):
        logger.info(
            "Annotations exist, so will attempt to make output manifest..."
        )

        make_output_manifest(
            inputs_json_file=inputs_json_file,
            tool_key=tool_key,
            toolbox_info_loc=toolbox_info_file,
            output_dir=os.getcwd(),
        )

        # metadata support
        write_output_metadata(
            metadata_file=metadata_file, manifest_file=manifest_file
        )

        # determine if we're being called by toolbox creator
        # if we're not being called by the toolbox creator
        # then we should rename outputs (if possible)
        # because we're likely being run on IDEAS

        rename = True
        if (
            "TC_NO_RENAME" in os.environ
            and os.environ["TC_NO_RENAME"] == "true"
        ):
            rename = False
        if rename:
            logger.info("Renaming outputs based on rules...")
            rename_outputs(tool_command, inputs_json_file, manifest_file)
        else:
            logger.info("Will not attempt to rename files.")

        # Now print out the outputs and file sizes
        with open(manifest_file, "r") as file:
            final_data = json.load(file)
        file_paths = _get_files(final_data)
        file_paths = sorted(file_paths)

        message = []
        for path in file_paths:
            message.append([os.path.basename(path), get_file_size(path)])

        table = tabulate(
            message, headers=["File", "Size"], tablefmt="fancy_grid"
        )
        logger.info("Output files and sizes:")
        logger.info("\n" + table)

    else:
        logger.info(
            "\n"
            + tabulate(
                [
                    [
                        """Cannot generate output manifest. If you're seeing this
            message when you first run a tool, that's OK. You 
            should now annotate outputs so that subsequent runs 
            of the tool can generate a output manifest (which is
            required for IDEAS)"""
                    ]
                ],
                headers=["No Annotations!"],
                tablefmt="fancy_grid",
                colalign=("center",),
            )
        )
