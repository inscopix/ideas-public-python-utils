from typing import Optional

from ideas.data_model_v2 import (
    Roi,
    get_rois_from_json,
    RectangleRoi,
    EllipseRoi,
    PolygonRoi,
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse


class Zone:
    """Zone object used for tracking zone occupancy in behavior movies."""

    def __init__(
        self,
        id: str = "",
        enabled: bool = True,
        description: str = "",
        roi: Optional[Roi] = None,
    ):
        """Create a zone object.

        :param id: The id of the zone.
        :param enabled: Flag indicating whether the zone is enabled for zone occupancy detection.
        :param description: Description of the zone provided by the user.
        :param roi: The roi of the zone.
        """
        self.id = id
        self.enabled = enabled
        self.description = description
        self.roi = roi

    def __eq__(self, other):
        """Determine if two zones are equal."""
        return (
            self.id == other.id
            and self.enabled == other.enabled
            and self.description == other.description
            and self.roi == other.roi
        )

    def __str__(self):
        """Create str representation of a zone."""
        return f"Zone:\n\tid = {self.id}\n\tenabled = {self.enabled}\n\tdescription = {self.description}\n\troi = {self.roi}"


def get_zones_from_json(j):
    """Get zones from json serialized roi inputs sent by the FE.

    :param j: The json object to convert.
    """

    rois = get_rois_from_json(j)
    zones = []
    for id, roi in enumerate(rois):
        zones.append(Zone(id=id, roi=roi))
    return zones


def convert_zones_to_dict(zones):
    """Convert zones to a list of dictionaries.

    :param zones: The zones to convert.
    """
    data = []
    for zone in zones:
        roi = zone.roi
        # Handle each ROI type
        if roi.shape_type == Roi.ShapeType.rectangle:
            # Create the DataFrame entry for the current ROI
            data.append(
                {
                    "ID": zone.id,
                    "Name": roi.name,
                    "Type": roi.shape_type.value,
                    "X 0": roi.points[0][0],  # top left
                    "Y 0": roi.points[0][1],
                    "X 1": roi.points[1][0],  # top right
                    "Y 1": roi.points[1][1],
                    "X 2": roi.points[2][0],  # bottom right
                    "Y 2": roi.points[2][1],
                    "X 3": roi.points[3][0],  # bottom left
                    "Y 3": roi.points[3][1],
                    "Color": roi.stroke,
                }
            )
        elif roi.shape_type == Roi.ShapeType.ellipse:
            # Create the DataFrame entry for the current ROI
            data.append(
                {
                    "ID": zone.id,
                    "Name": roi.name,
                    "Type": roi.shape_type.value,
                    "X 0": roi.center[0],
                    "Y 0": roi.center[1],
                    "Major Axis": roi.rx * 2,
                    " Minor Axis": roi.ry * 2,
                    " Angle": roi.rotation,
                    "Color": roi.stroke,
                }
            )
        elif roi.shape_type == Roi.ShapeType.polygon:
            # Initialize the DataFrame entry for the current ROI and the points list
            data.append(
                {
                    "ID": zone.id,
                    "Name": roi.name,
                    "Type": roi.shape_type.value,
                    "Color": roi.stroke,
                }
            )
            # Add the points to the list and the DataFrame entry
            for i, point in enumerate(roi.points):
                data[-1][f"X {i}"] = point[0]
                data[-1][f"Y {i}"] = point[1]
    return data


def read_zones_from_dict(df):
    """Read zones from a csv file.

    :param filename: The csv file to read from.
    """
    zones = []
    rows = df if isinstance(df, list) else [r for _, r in df.iterrows()]
    for row in rows:
        shape_type = Roi.ShapeType(row["Type"])
        if shape_type == Roi.ShapeType.rectangle:
            roi = RectangleRoi(
                group_key="",
                name=row["Name"],
                stroke=(255, 255, 255) if "Color" not in row else row["Color"],
                left=row["X 0"],
                top=row["Y 0"],
                points=[(row[f"X {i}"], row[f"Y {i}"]) for i in range(4)],
            )
        elif shape_type == Roi.ShapeType.ellipse:
            roi = EllipseRoi(
                group_key="",
                name=row["Name"],
                stroke=(255, 255, 255) if "Color" not in row else row["Color"],
                center=(row["X 0"], row["Y 0"]),
                rx=row["Major Axis"] / 2.0,
                ry=row[" Minor Axis"] / 2.0,
                rotation=row[" Angle"],
            )
        elif shape_type == Roi.ShapeType.polygon:
            num_points = int(
                len([k for k in row.keys() if "X " in k or "Y " in k]) / 2
            )
            roi = PolygonRoi(
                group_key="",
                name=row["Name"],
                stroke=(255, 255, 255) if "Color" not in row else row["Color"],
                points=[
                    (row[f"X {i}"], row[f"Y {i}"]) for i in range(num_points)
                ],
            )

        zones.append(
            Zone(
                id=row["ID"],
                enabled=True if "Enabled" not in row else row["Enabled"],
                description=""
                if "Description" not in row or pd.isna(row["Description"])
                else row["Description"],
                roi=roi,
            )
        )

    return zones


def read_zones_from_csv(filename):
    """Read zones from a csv file.

    :param filename: The csv file to read from.
    """
    df = pd.read_csv(filename)
    return read_zones_from_dict(df)


def write_zones_to_csv(zones, filename):
    """Write zones to a csv  file.

    :param zones: The zones to write.
    :param filename: The csv file to write to.
    """
    data = convert_zones_to_dict(zones)
    df = pd.DataFrame(columns=data[0].keys())

    for idx, zone in enumerate(data):
        for key in zone.keys():
            if key not in df.columns:
                df[key] = None
        df.loc[idx] = zone

    # Remove color column if it exists
    if "Color" in df.columns:
        df.drop("Color", axis=1, inplace=True)

    df.to_csv(filename, index=False)


def plot_zones_on_im(zones, im):
    """Plot zones on an image"""
    for zone in zones:
        roi = zone.roi
        # Get the color if it exists
        if roi.stroke:
            color = roi.stroke
        else:
            color = (255, 255, 255)

        if roi.shape_type == Roi.ShapeType.ellipse:
            # Draw the ellipse on the preview image
            im = cv2.ellipse(
                im,
                (int(roi.center[0]), int(roi.center[1])),
                (
                    int(roi.rx),
                    int(roi.ry),
                ),
                roi.rotation,
                0,
                360,
                color,
                2,
            )
        elif (
            roi.shape_type == Roi.ShapeType.polygon
            or roi.shape_type == Roi.ShapeType.rectangle
        ):
            points = np.array(roi.points, np.int32).reshape((-1, 1, 2))
            # Draw the polygon on the preview image
            im = cv2.polylines(
                im,
                [points],
                isClosed=True,
                color=color,
                thickness=2,
            )

    return im


def plot_zones_on_ax(
    zones,
    ax: plt.Axes = None,
    cmap: str = "viridis_r",
    title: str = "Trajectory with Zones",
    x_label: str = "X (pixels)",
    y_label: str = "Y (pixels)",
    use_zone_colors: bool = False,
):
    """Draw the zones on the plot."""

    col_pal = sns.color_palette(cmap, n_colors=len(zones))
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Sort zones by name so that the colors are consistent
    zone_names = [zone.roi.name for zone in zones]
    zone_names.sort()

    for idx, zone_name in enumerate(zone_names):
        # There should only be one zone with the same name
        # Find the zone with the same name
        zone = next(
            (zone for zone in zones if zone.roi.name == zone_name),
            None,
        )
        roi = zone.roi
        if use_zone_colors and zone.stroke:
            color = tuple(val / 255 for val in zone.stroke)
        else:
            print("No zone color detected, using default colormap")
            color = col_pal[idx]

        if roi.shape_type == Roi.ShapeType.ellipse:
            ellipse = Ellipse(
                xy=roi.center,
                width=roi.rx * 2,
                height=roi.ry * 2,
                angle=roi.rotation,
                fill=False,
                edgecolor=color,
                linewidth=3,
            )
            ax.add_patch(ellipse)
            ax.text(
                roi.center[0],
                roi.center[1],
                roi.name,
                ha="center",
                va="center",
            )
        elif (
            roi.shape_type == Roi.ShapeType.polygon
            or roi.shape_type == Roi.ShapeType.rectangle
        ):
            coords = roi.points
            ax.plot(*zip(*(coords + [coords[0]])), color=color)
            # Find the center of the polygon
            x_center = sum(x for x, _ in coords) / len(coords)
            y_center = sum(y for _, y in coords) / len(coords)
            ax.text(x_center, y_center, roi.name, ha="center", va="center")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    return ax
