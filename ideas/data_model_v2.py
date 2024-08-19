import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from glob import glob
from pathlib import Path

from beartype import beartype
from beartype.typing import List, Optional
from ideas import json_utils as ju
from ideas import schemas
from ideas.utils import _check_in, isxd_type, unique_id
from ideas_commons.constants import data
from jsonschema import validate

# from the spec in ideas-commons
allowed_file_formats = [thing.value[1] for thing in data.FileFormat]
allowed_file_types = [thing.value[1] for thing in data.FileType]
allowed_file_structures = [thing.value[1] for thing in data.FileStructure]
allowed_file_categories = [thing.value[1] for thing in data.FileCategory]
allowed_object_types = [thing.value[1] for thing in data.ObjectType]
allowed_group_types = [thing.value[1] for thing in data.GroupType]


@beartype
def make_output_manifest(
    *,
    inputs_json_file: str,
    toolbox_info_loc: str,
    tool_key: str,
    output_manifest_file_loc: str = "output_manifest.json",
    output_dir: str,
):
    """Helper function that makes an output manifest.

    This uses information from the results and param
    section of the toolbox_info.
    """

    if not os.path.exists(toolbox_info_loc):
        raise FileNotFoundError(
            f"""
toolbox_info.json not found, cannot proceed. 
Expected to find it at {toolbox_info_loc} 
(which was supplied to this function)"""
        )

    if not os.path.exists(inputs_json_file):
        raise FileNotFoundError(
            f"""
inputs.json not found, cannot proceed. 
Expected to find it at {inputs_json_file} 
(which was supplied to this function)"""
        )

    # extract the tool command as this is what's used
    # to name the relevant info files for the tool
    tool_command = ju.read(
        key=f"tools.[{tool_key}].command",
        info_loc=toolbox_info_loc,
    )[
        :-3
    ]  # strip the .sh from the command string to just get <module_name>__<function_name>

    # read the JSON in directly
    # we trust that the JSON is correctly formatted
    with open(inputs_json_file, "r") as file:
        args = json.load(file)

    # figure out which input parameters are source
    # files and create IdeasFiles out of them
    source_files = []
    for key, file_paths in args.items():
        is_source = ju.read(
            key=f"tools.[{tool_key}].params.[{key}].type.is_source",
            info_loc=toolbox_info_loc,
        )
        if not is_source:
            continue

        # this param is a source file
        param_info = ju.read(
            key=f"tools.[{tool_key}].params.[{key}].type",
            info_loc=toolbox_info_loc,
        )

        # Note: it doesn't really matter if input files
        # are part of a series or not. Tools does not care.
        # we do not mark whether input files are part of a
        # series or not because a) we have no way of knowing
        # and b) the BE doesn't need us to say anything about
        # this.
        # therefore, we treat all files identically and lump
        # them all together

        for file_path in file_paths:
            # there can be many file filters, so how do we know
            # what the input file is? we assume
            # that file formats can vary, but file types and
            # structures stay the same. so we read the latter
            # from the first file filter, and determine
            # the file format from the file

            _, file_ext = os.path.splitext(file_path)

            this_file = IdeasFile(
                file_key=key,
                file_path=file_path,
                file_type=param_info["file_filters"][0]["file_type"],
                file_format=file_ext[1:],  # lose the .
                file_structure=param_info["file_filters"][0]["file_structure"],
                file_category="source",
            )
            source_files.append(this_file)

    if len(source_files) == 0:
        raise Exception(
            """It is impossible for a tool to have 0 input
             files. Something has gone wrong."""
        )

    # now do output files
    result_files = []
    # read in the annotations
    json_file = os.path.join(
        Path(toolbox_info_loc).parent,
        f"results_{tool_command}.json",
    )

    # figure out all the files that can possibly be
    # generated
    with open(json_file, "r") as file:
        annotations = json.load(file)

    # make ideas files out of these files
    all_preview_info = dict()
    for annotation in annotations:
        # an output can have multiple file formats
        # search for any files in the output dir that match the
        # output name + one of the possible file formats
        file_exists = False

        # previously file_format was str, so both list and str are accepted for backwards compatibility
        if not isinstance(annotation["file_format"], (list, str)):
            raise ValueError(
                f"File format in results annotation must be either list or str, but is {type(annotation['file_format'])}"
            )

        file_formats = (
            annotation["file_format"]
            if isinstance(annotation["file_format"], list)
            else [annotation["file_format"]]
        )
        for file_format in file_formats:
            basename = annotation["key"] + "." + file_format

            file_path = os.path.join(
                output_dir,
                basename,
            )

            # has this been created?
            if not os.path.exists(file_path):
                # in the case of multiple files for one output
                # files must be named annotation["key"].x.file_format
                # where x is the index of the file in the output
                if (
                    annotation["is_output"] or annotation["is_preview"]
                ) and annotation["multiple"]:
                    # search for files that match this pattern and sort the order
                    file_paths = sorted(
                        glob(
                            f"{output_dir}/{annotation['key']}.*.{file_format}"
                        )
                    )
                    if file_paths:
                        file_exists = True
                        break
                continue
            else:
                file_exists = True
                break

        if not file_exists:
            # the file does not exist, don't include in the output manifest
            continue

        if not annotation["multiple"]:
            # this is a single output file
            file_paths = [file_path]

        if annotation["is_output"]:
            # this is a result output

            parent_keys = annotation["parent_keys"]

            parent_ids = [
                file.file_id
                for file in source_files
                if file.file_key in parent_keys
            ]

            for file_path in file_paths:
                this_result = IdeasFile(
                    file_path=file_path,
                    file_structure=annotation["file_structure"],
                    file_key=annotation["key"],
                    file_type=annotation["file_type"],
                    file_format=file_format,
                    file_category="result",
                    help=annotation["help"],
                    parent_ids=parent_ids,
                )

                result_files.append(this_result)

        if annotation["is_preview"]:
            # this file is a preview

            for file_path in file_paths:
                preview_info = dict(
                    name=os.path.basename(file_path),
                    help=annotation["help"],
                    file_path=file_path,
                    file_format=annotation["file_format"],
                )
                for preview_of in annotation["preview_of"]:
                    if preview_of not in all_preview_info:
                        all_preview_info[preview_of] = []
                    all_preview_info[preview_of].append(preview_info)

    def _get_output_num(path):
        """small helper function to parse the index number of an output file,
        if one exists in the output file path.

        in the case of multiple file outputs, this will return a number,
        otherwise it will return none."""
        basename, _ = os.path.splitext(path)
        _, num_ext = os.path.splitext(basename)
        if len(num_ext) > 1 and num_ext[0] == ".":
            try:
                return int(num_ext[1:])
            except Exception:
                pass

    # attach previews to result files
    for file in result_files:
        if file.file_key in all_preview_info.keys():
            output_num = _get_output_num(file.file_path)
            if output_num is None:
                file.preview = all_preview_info[file.file_key]
            else:
                # in the case of multiple files in one output
                # map the preview for the file using the index number in both file paths
                file.preview = []
                for preview_info in all_preview_info[file.file_key]:
                    preview_num = _get_output_num(preview_info["file_path"])
                    if preview_num == output_num:
                        file.preview.append(preview_info)

    # make a IdeasGroup and put all files in here
    group = IdeasGroup(
        files=source_files + result_files,
        group_key=tool_command + "_output",
    )

    group.save_output_manifest(
        output_manifest_file_loc=os.path.join(
            output_dir,
            output_manifest_file_loc,
        ),
    )

    # validate schema
    with open(output_manifest_file_loc, "r") as file:
        data = json.load(file)

    validate(
        instance=data,
        schema=schemas.output_manifest_v2,
    )


class MetadataInstruction(Enum):
    """Metadata instructions"""

    ADD = 1, "add"
    INHERIT = 2, "inherit"


@dataclass
class IdeasFile:
    """wrapper around a single file that exists on disk"""

    file_key: str = None  # called key in toolbox info
    file_name: str = None
    file_id: str = field(default_factory=unique_id)
    file_path: str = None
    file_type: str = None
    file_format: str = None
    file_structure: str = None
    file_category: str = None

    parent_ids: Optional[List[str]] = None

    series: Optional[dict] = None

    preview: Optional[List[dict]] = None

    # needed in toolbox info.json
    help: str = None
    required: bool = False
    hidden: bool = False

    def __post_init__(self):
        if self.file_path is None:
            raise Exception("file path must be specified")

        if not os.path.exists(self.file_path):
            raise Exception(f"file_path: {self.file_path} does not exist")

        # automatic inference
        if self.file_name is None:
            self.file_name = os.path.basename(self.file_path)

        if self.file_format is None:
            remainder, file_format = os.path.splitext(self.file_path)
            self.file_format = file_format[1:]
            # tar.gz is a unique case that we need to handle
            if self.file_format == "gz":
                if remainder.split(".")[-1] == "tar":
                    self.file_format = "tar.gz"

        if self.file_format == "isxd" and self.file_type is None:
            # attempt to figure out file type from ISXD file
            self.file_type = isxd_type(self.file_path)

        # attempt to automatically infer file structure
        # file structure is mostly redundant with file_type
        # so let's save the user some time
        if self.file_structure is None:
            mapper = dict(
                csv="table",
                movie="binary",
                miniscope_movie="movie",
                miniscope_image="image",
                cell_set="binary",
                neural_events="binary",
            )
            if self.file_type in mapper.keys():
                self.file_structure = mapper[self.file_type]
            else:
                raise Exception("file_structure not set.")

        # validate inputs

        _check_in(self.file_format, allowed_file_formats)
        _check_in(self.file_structure, allowed_file_structures)
        _check_in(self.file_type, allowed_file_types)

    def dict(self):
        """convert object to a dictionary"""
        return {k: str(v) for k, v in asdict(self).items()}

    def to_manifest(self) -> dict:
        """creates a dict with info that needs to be
        in the output manifest"""

        data = dict(
            file_key=self.file_key,
            file_name=self.file_name,
            file_id=self.file_id,
            file_path=self.file_path,
            file_type=self.file_type,
            file_format=self.file_format,
            file_structure=self.file_structure,
            file_category=self.file_category,
        )

        if self.parent_ids is not None:
            data["parent_ids"] = self.parent_ids

        if self.preview is not None:
            data["preview"] = self.preview

        return data


@beartype
@dataclass
class IdeasGroup:
    """Collection of objects and files"""

    files: List[IdeasFile]
    group_key: str
    group_type: str = "tool_output"
    group_id: str = field(default_factory=unique_id)
    series = []

    def __post_init__(self):
        if len(self.files) == 0:
            raise Exception(
                """You cannot create a group
             with 0 files."""
            )

    # @beartype
    # def __init__(
    #     self,
    #     *,
    #     group_key: str,  # some user defined name
    #     group_type: str = "tool_output",  # TODO validate
    #     required: bool = False,
    #     multiple: bool = False,
    #     series=None,
    #     members: List[IdeasFile],
    #     add_metadata=None,
    #     inherit_metadata=None,
    # ):
    #     _check_in(group_type, allowed_group_types)

    #     """Construct group on IDEAS"""
    #     self.group_key = group_key
    #     self.group_name = group_key.replace(
    #         "_", " "
    #     ).title()  # Only in toolbox_info.json

    #     self.group_type = group_type
    #     self.required = required
    #     self.multiple = multiple

    #     self.group_id = unique_id()
    #     self.series = series

    #     # create a dictionary with keys of UUIDs <- what is a UUIDs?
    #     self.files = {o.file_id: o for o in members}

    #     self.group_metadata = {
    #         "group_id": self.group_id,
    #         MetadataInstruction.ADD.value[1]: {}
    #         if add_metadata is None
    #         else add_metadata,
    #         MetadataInstruction.INHERIT.value[1]: []
    #         if inherit_metadata is None
    #         else inherit_metadata,
    #     }

    #     if (
    #         "ideas"
    #         not in self.group_metadata[MetadataInstruction.ADD.value[1]]
    #     ):
    #         self.group_metadata[MetadataInstruction.ADD.value[1]]["ideas"] = {}
    #     if (
    #         "dataset"
    #         not in self.group_metadata[MetadataInstruction.ADD.value[1]][
    #             "ideas"
    #         ]
    #     ):
    #         self.group_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
    #             "dataset"
    #         ] = {}
    #     self.group_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
    #         "dataset"
    #     ]["group_type"] = self.group_type

    def make_output_manifest(
        self,
        manifest_loc: str = "output_manifest.json",
    ):
        """makes an output manifest from a IdeasGroup"""
        data = dict(
            group_key=self.group_key,
            group_type=self.group_type,
            group_id=self.group_id,
            series=self.series,
        )

        data["files"] = [file.to_manifest() for file in self.files]

        return data

    def add_group_metadata(self, key, value):
        """Add group-level metadata.

        :param key: metadata key
        :param value: metadata value associated with the key
        """
        if key in self.group_metadata[MetadataInstruction.ADD.value[1]]:
            raise KeyError("Group metadata key is already defined")
        self.group_metadata[MetadataInstruction.ADD.value[1]][key] = value

    @beartype
    def save_output_manifest(
        self,
        *,
        output_manifest_file_loc: str = "output_manifest.json",
    ) -> None:
        """Save output manifest to disk in a JSON format
        that the IDEAS backend can read
        """

        # convert to a dict, and put it inside a list
        # this is the way the JSON is to be formatted
        groups = [self.make_output_manifest()]
        data = dict(
            schema_version="2.0.0",
            groups=groups,
        )

        with open(output_manifest_file_loc, "w") as f:
            json.dump(data, f, indent=2)

    @beartype
    def save_metadata_manifest(
        self,
        *,
        output_manifest_file_loc: str = "output_metadata.json",
        include_sources: bool = False,
    ):
        """Save metadata manifest to disk.


        :param output_dir: path to the output directory
        :param include_sources: if True sources metadata will be included
        """

        # gather metadata
        group_metadata = [self.group_metadata]
        file_metadata = []

        # loop over each object in the group
        for obj in self.files.values():
            if include_sources:
                file_metadata.append(obj.file_metadata)
            else:
                # check if a file is a source
                if not obj.file_category == data.FileCategory.SOURCE.value[1]:
                    file_metadata.append(obj.file_metadata)

            # construct metadata manifest
            metadata_manifest = {
                "group_metadata": group_metadata,
                "file_metadata": file_metadata,
            }

            # save metadata manifest to disk
            with open(output_manifest_file_loc, "w") as f:
                json.dump(metadata_manifest, f, indent=4)
