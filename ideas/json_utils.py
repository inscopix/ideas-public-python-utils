"""utilities to work with JSON files in ideas"""

import json

from beartype import beartype

# the reason for this import is because write
# may also use st.session_state to determine
# what value to write to the JSON file
try:
    import streamlit as st
except ImportError:
    pass


@beartype
def read(
    *,
    key: str,
    info_loc: str,
):
    """helper function to read out a specific value
    from the toolbox info.json file

    Example usage:

    read(key="tools.[tool_key].name",
        info_loc="path/to/toolbox_info.json")

    If your arrays are indexed using a key that is not
    called key, specify that using array_key

    returns the name of the tool with key `tool_key`

    """
    with open(info_loc, "r") as f:
        info = json.load(f)

    key = key.split(".")

    for k in key:
        if "[" in k:
            k = k.replace("[", "")
            k = k.replace("]", "")

            try:
                # first check if it's a numeric value
                int(k)
                info = info[int(k)]
            except ValueError:
                # not a number, so need to figure out
                # the index

                if "=" in k:
                    array_key, k = k.split("=")
                    array_key = array_key.replace("[", "")
                else:
                    array_key = "key"

                # find the correct array element
                # we assume at this point that info
                # is a list
                for idx, thing in enumerate(info):
                    if thing[array_key] == k:
                        # index into array
                        info = info[idx]

                        break

        else:
            # simple indexing, moving to child object
            info = info[k]

    return info


def write(
    *,
    key: str,
    info_loc: str,
    value=None,
) -> None:
    """helper function to write out a specific value
    to the toolbox info.json file

    Example usage (outside of streamlit):

    write(key="tools.[tool_key].name",
        value="wow new name",
        info_loc="path/to/toolbox_info.json"
        )

    If you're using this inside streamlit, you
    can omit `value`:

    write(key="tools.[tool_key].name",
        info_loc="path/to/toolbox_info.json"
        )

    and the value will be inferred from a widget
    with the same key

    """

    if value is None:
        # read from session state
        value = st.session_state[key]

    with open(info_loc, "r") as f:
        source_info = json.load(f)

    info = source_info
    key = key.split(".")

    # move into the dictionary that contains our key
    # of interest. because the last entry in key **should**
    # be a string, going to the penultimate value
    # should be a dictionary
    for k in key[:-1]:
        if "[" in k:
            k = k.replace("[", "")
            k = k.replace("]", "")

            try:
                # first check if it's a numeric value
                int(k)
                source_info = source_info[int(k)]
            except ValueError:
                # not a number, so need to figure out
                # the index

                if "=" in k:
                    array_key, k = k.split("=")
                    array_key = array_key.replace("[", "")
                else:
                    array_key = "key"

                # find the correct array element
                # we assume at this point that info
                # is a list
                for idx, thing in enumerate(source_info):
                    if thing[array_key] == k:
                        # index into array
                        source_info = source_info[idx]
                        break

        else:
            # simple indexing, moving to child object
            source_info = source_info[k]

    source_info[key[-1]] = value

    # save data
    with open(info_loc, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def delete(
    *,
    key: str,
    info_loc: str,
) -> None:
    """Deletes a specific key and value from the toolbox info.json file.

    Args:
        key (str): The key to be deleted. It should be in dot notation format.
        info_loc (str): The file path of the info.json file.

    Returns:
        None

    Example usage:
        delete(key="tools.[{tool_key}].params.[{arg_name}].type.min",
            info_loc="path/to/toolbox_info.json"
        )
    """

    with open(info_loc, "r") as f:
        source_info = json.load(f)

    info = source_info
    key = key.split(".")

    # move into the dictionary that contains our key
    # of interest. because the last entry in key **should**
    # be a string, going to the penultimate value
    # should be a dictionary
    for k in key[:-1]:
        if "[" in k:
            k = k.replace("[", "")
            k = k.replace("]", "")

            try:
                # first check if it's a numeric value
                int(k)
                source_info = source_info[int(k)]
            except ValueError:
                # not a number, so need to figure out
                # the index

                if "=" in k:
                    array_key, k = k.split("=")
                    array_key = array_key.replace("[", "")
                else:
                    array_key = "key"

                # find the correct array element
                # we assume at this point that info
                # is a list
                for idx, thing in enumerate(source_info):
                    if thing[array_key] == k:
                        # index into array
                        source_info = source_info[idx]
                        break

        else:
            # simple indexing, moving to child object
            source_info = source_info[k]

    del source_info[key[-1]]

    # save data
    with open(info_loc, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def check(*, info_loc: str, key: str) -> None:
    """
    Checks if a key is present in the .json file. If the key is not present, it creates it.

    Args:
        info_loc (str): The location of the .json file.
        key (str): The key to check in the .json file.

    Returns:
        None
    """
    try:
        read(
            info_loc=info_loc,
            key=key,
        )
        return True
    except KeyError:
        return False
