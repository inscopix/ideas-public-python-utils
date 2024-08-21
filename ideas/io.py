import warnings

import numpy as np
import scipy
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from scipy.ndimage import convolve
from skimage.measure import find_contours, perimeter

import ideas.measures as measures
from ideas.types import NumpyFloatVector, NumpyVector
from ideas.validation import cell_set_series

try:
    import isx
except ImportError as err:
    print(
        """
+-----------------------------------------+
| Could not import isx. You need to have  |
| the IDPS API installed or the python-   |
| based experimental API                  |
+-----------------------------------------+
"""
    )
    raise err


def movie_to_correlation_image(
    movie_array: np.ndarray,
    *,
    eight_neighbours: bool = True,
    swap_dim: bool = True,
    rolling_window: int = None,
) -> np.ndarray:
    """Compute the correlation image for the input dataset
    Y using a faster FFT based method.
    Taken from caiman implementation:
    https://github.com/flatironinstitute/CaImAn/blob/
    b5932abe85c478fae4004e8df14535f3f5f7274a/caiman/summary_images.py#L73
    Parameters:
        movie_array <np.array>: Input movie data in 3D or 4D format
        eight_neighbours <Bool>: Use 8 neighbors if true,
        and 4 if false for 3D data (default =
        True) Use 6 neighbors for 4D data, irrespectively.
        swap_dim <Bool>: True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front


        rolling_window <int>: Rolling window
    Returns:
        corr_img <np.array>: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels


    this was originally in the IDPS toolbox, moved here and
    cleaned up a bit.
    """

    if swap_dim:
        movie_array = np.transpose(
            movie_array,
            tuple(
                np.hstack(
                    (movie_array.ndim - 1, list(range(movie_array.ndim))[:-1])
                )
            ),
        )

    movie_array = movie_array.astype("float32")
    if rolling_window is None:
        movie_array -= np.mean(movie_array, axis=0)
        movie_array_std = np.std(movie_array, axis=0)
        movie_array_std[movie_array_std == 0] = np.inf
        movie_array /= movie_array_std
    else:
        movie_array_sum = np.cumsum(movie_array, axis=0)
        movie_array_rm = (
            movie_array_sum[rolling_window:]
            - movie_array_sum[:-rolling_window]
        ) / rolling_window
        movie_array[:rolling_window] -= movie_array_rm[0]
        movie_array[rolling_window:] -= movie_array_rm
        del movie_array_rm, movie_array_sum
        movie_array_std = np.cumsum(movie_array**2, axis=0)
        movie_array_rst = np.sqrt(
            (
                movie_array_std[rolling_window:]
                - movie_array_std[:-rolling_window]
            )
            / rolling_window
        )
        movie_array_rst[movie_array_rst == 0] = np.inf
        movie_array[:rolling_window] /= movie_array_rst[0]
        movie_array[rolling_window:] /= movie_array_rst
        del movie_array_std, movie_array_rst

    if movie_array.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype="float32")
            sz[1, 1, 1] = 0
        else:
            # yapf: disable
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                          dtype='float32')
            # yapf: enable
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype="float32")
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype="float32")

    movie_array_conv = convolve(
        movie_array, sz[np.newaxis, :], mode="constant"
    )
    MASK = convolve(
        np.ones(movie_array.shape[1:], dtype="float32"),
        sz,
        mode="constant",
    )

    # YYconv is the product of the actual movie array
    # with its convolved version
    YYconv = movie_array_conv * movie_array
    del movie_array, movie_array_conv
    if rolling_window is None:
        corr_img = np.mean(YYconv, axis=0) / MASK
    else:
        YYconv_cs = np.cumsum(YYconv, axis=0)
        del YYconv
        YYconv_rm = (
            YYconv_cs[rolling_window:] - YYconv_cs[:-rolling_window]
        ) / rolling_window
        del YYconv_cs
        corr_img = YYconv_rm / MASK

    return corr_img


@beartype
def cell_set_to_traces(files: Union[str, List[str]]):
    # -> (NumpyFloat2DArray, NumpyFloatVector):
    """read cell set file and get all traces

    Returns numpy array (time x num_cells) with traces
    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(files)

    all_traces = []
    for file in files:
        cellset = isx.CellSet.read(file)

        num_cells = cellset.num_cells
        num_samples = cellset.timing.num_samples

        traces = np.full((num_samples, num_cells), np.nan)

        for i in range(num_cells):
            traces[:, i] = cellset.get_cell_trace_data(i)

        all_traces.append(traces)

    return np.concatenate(all_traces)


@beartype
def cell_set_to_time(files: Union[str, List[str]]) -> NumpyFloatVector:
    """returns time vector for a cellset.isxd file

    time is a vector in seconds from the start of the recording

    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(files)

    all_time = []
    last_time = 0
    for file in files:
        cellset = isx.CellSet.read(file)

        dt = cellset.timing.period.secs_float
        time = np.linspace(
            0,
            (cellset.timing.num_samples - 1) * dt,
            cellset.timing.num_samples,
        )

        all_time.append(time + last_time)
        last_time += time[-1]

    return np.concatenate(all_time)


@beartype
def cell_set_to_positions(
    files: Union[str, List[str]], *, threshold: float = 4.0
):
    """returns cell positions given a cell set file path"""

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(files)

    # just read out from first file
    # assume that this is a valid series

    cellset = isx.CellSet.read(files[0])

    num_cells = cellset.num_cells

    positions_x = np.full((num_cells), np.nan)
    positions_y = np.full((num_cells), np.nan)

    for i in np.arange(num_cells).tolist():
        footprint = cellset.get_cell_image_data(i)
        footprint -= footprint.min()
        footprint /= footprint.max()
        footprint[footprint < 1 / threshold] = 0
        pos = scipy.ndimage.center_of_mass(footprint)
        positions_y[i] = pos[0]
        positions_x[i] = pos[1]

    return positions_x, positions_y


@beartype
def cell_set_to_status(files: Union[str, List[str]]) -> NumpyVector:
    """cellset file to array of status for each neuron

    ### Arguments:

    -cell_set_file: location of cellset .isxd file

    ### Returns:

    - np.array: status array (Nx1)
    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(files)

    # simply return the status of the first file
    # cell set series validation guarantees that
    # if status are different, then it will not be a
    # valid series

    cellset = isx.CellSet.read(files[0])
    num_cells = cellset.num_cells
    status = [cellset.get_cell_status(i) for i in range(num_cells)]

    return np.array(status)


@beartype
def cell_set_to_footprint_metrics(
    files: Union[str, List[str]],
    *,
    threshold: float = 4.0,
) -> dict:
    """cellset file to array of areas of footprints

    counts the number of pixels above some threshold, defined
    by the max of that footprint.

    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(files)

    # just operate on the first one, assume that
    # this is a valid series (validation done
    # in cell_set_series)

    cellset = isx.CellSet.read(files[0])

    areas = np.zeros(cellset.num_cells)
    perimeters = np.zeros(cellset.num_cells)

    for i in range(cellset.num_cells):
        footprint = cellset.get_cell_image_data(i)

        # discretize
        footprint = footprint > footprint.max() / threshold

        areas[i] = np.sum(footprint)
        perimeters[i] = perimeter(footprint, neighborhood=8)

        if perimeters[i] < 16:
            # we can't estimate this well, so we will default
            # to a value so that the circularity is
            # constrained to 1
            perimeters[i] = np.sqrt(areas[i] * np.pi * 4)

    areas = areas.astype(int)

    # TODO replace this with a nicer roundness measure
    # that depends on the ratios of inscribed to
    # superscribed circles, see:
    # https://en.wikipedia.org/wiki/Roundness
    circularity = (4 * np.pi * areas) / (perimeters**2)

    return dict(
        areas=areas,
        perimeters=perimeters,
        circularity=circularity,
    )


def cell_set_to_contours(
    files: Union[str, List[str]], *, threshold: float = 4.0
) -> Tuple[List, List]:
    """generate list of contours from a cell set file

    ### Arguments:

    - cell_set_file: name of isxd CellSet file
    - threshold: to consider footprints within cell

    threshold is a parameter that divides the maximum of each cell's
    footprint, and defines the threshold to draw a contour around


    ### Returns

    - contours_x: list of list of arrays of contours
    - contours_y: list of list of arrays of contours

    This convoluted format is so that this can be
    fed directly into bokeh.multi_polygons
    """

    if isinstance(files, str):
        files = [files]

    files = cell_set_series(files)

    # just operate on the first one, assume that
    # this is a valid series (validation done
    # in cell_set_series)

    # generate contours of all cells
    contours_x = []
    contours_y = []

    cellset = isx.CellSet.read(files[0])

    for i in range(cellset.num_cells):
        footprint = cellset.get_cell_image_data(i)

        if footprint.max() == 0:
            warnings.warn(
                f"""cell {i} in {files[0]} has all-zero 
    footprints, which is a bug in the cell extraction. 
    We cannot draw contours for this cell.""",
                stacklevel=2,
            )

            contours_x.append([[0]])
            contours_y.append([[0]])
            continue

        xy = find_contours(footprint.transpose(), footprint.max() / threshold)

        contours_x.append([[xy[0][:, 0]]])
        contours_y.append([[xy[0][:, 1]]])

    return (contours_x, contours_y)


@beartype
def prototypical_cells(
    cell_set_file: str,
    *,
    only_from_these_cells: Optional[NumpyVector] = None,
) -> tuple:
    """return indicies of cells with max, median and min skew

    ### Arguments:

    - cell_set_file: name of cell set file
    - only_from_these_cells: array of cell indicies that we are
    allowed to choose from. If None, all cells are allowed


    ### Returns

    - (max_cell, median_cell, min_cell) tuple of indices

    may contain duplicates

    """
    traces = cell_set_to_traces(cell_set_file)

    skew = measures.matrix_skew(traces)

    if only_from_these_cells is not None:
        all_cells = np.arange(len(skew))
        bad_cells = np.setdiff1d(all_cells, only_from_these_cells)
        skew[bad_cells] = np.nan

    return (
        np.nanargmax(skew),
        np.nanargmin(np.abs(skew - np.nanmedian(skew))),
        np.nanargmin(skew),
    )
