from enum import Enum

import numpy as np
from beartype.typing import List, Optional, Union

from ideas.utils import _hex_to_rgb


class Roi:
    """Base class for roi inputs.

    Rois are drawn in the IDEAS FE and passed to tools as json objects.
    """

    class ShapeType(Enum):
        """The type of shape of an roi input."""

        bounding_box = "boundingBox"
        rectangle = "rectangle"
        ellipse = "ellipse"
        polygon = "polygon"
        contour = "contour"

    def __init__(
        self,
        shape_type: ShapeType,
        group_key: str,
        name: str,
        stroke: Union[str, tuple],
    ):
        """Create instance of roi.

        :param shape_type: The type of shape of the roi.
        :param group_key: The key of the group that the roi belongs to.
        :param name: The name of the roi given by the user.
        :param stroke: The color of the roi given by the user.
        """

        self.shape_type = shape_type
        self.group_key = group_key
        self.name = name
        if isinstance(stroke, str):
            self.stroke = _hex_to_rgb(stroke)
        elif isinstance(stroke, tuple):
            self.stroke = stroke

    def __eq__(self, other):
        """Determine if two rois are equal."""
        return (
            self.shape_type == other.shape_type
            and self.group_key == other.group_key
            and self.name == other.name
            and self.stroke == other.stroke
        )


class PolygonRoi(Roi):
    "An roi with a polygon shape."

    def __init__(
        self,
        group_key: str,
        name: str,
        stroke: Union[str, tuple],
        points: List[tuple],
    ):
        """Create instance of polygon roi.

        :param group_key: The key of the group that the roi belongs to.
        :param name: The name of the roi given by the user.
        :param stroke: The color of the roi given by the user.
        :param points: The points of the polygon.
        """

        Roi.__init__(
            self,
            shape_type=Roi.ShapeType.polygon,
            group_key=group_key,
            name=name,
            stroke=stroke,
        )
        self.points = points

    @classmethod
    def from_json(cls, j: dict):
        """Create a polygon roi from a json object.

        :param j: The roi input serialized as json.
        """
        if Roi.ShapeType(j["type"]) != Roi.ShapeType.polygon:
            raise ValueError("Unexpected shape type for roi input")

        return cls(
            j["groupKey"],
            j["name"],
            j["stroke"],
            [(p["x"], p["y"]) for p in j["points"]],
        )

    def __eq__(self, other):
        """Determine if two polygon rois are equal."""
        return Roi.__eq__(self, other) and np.allclose(
            self.points, other.points
        )

    def __str__(self):
        """Create str representation of polygon roi."""
        return f"Polygon(group_key={self.group_key},name={self.name},stroke={self.stroke},points={self.points})"


class RectangleRoi(Roi):
    "An roi with a rectangle shape."

    def __init__(
        self,
        group_key: str,
        name: str,
        stroke: Union[str, tuple],
        top: Union[int, float],
        left: Union[int, float],
        width: Optional[Union[int, float]] = None,
        height: Optional[Union[int, float]] = None,
        rotation: Optional[Union[int, float]] = None,
        points: Optional[List[tuple]] = None,
    ):
        """Create instance of rectangle roi.

        :param group_key: The key of the group that the roi belongs to.
        :param name: The name of the roi given by the user.
        :param stroke: The color of the roi given by the user.
        :param top: The top coordinate of the rectangle.
        :param left: The left coordinate of the rectangle.
        :param width: The width of the rectangle.
        :param height: The height of the rectangle.
        :param rotation: The angle of rotation, in degrees, of the rectangle.
        :param points: The points of the rectangle (after rotation). If not provided, computed from top, left, width, height.
        """
        Roi.__init__(
            self,
            shape_type=Roi.ShapeType.rectangle,
            group_key=group_key,
            name=name,
            stroke=stroke,
        )
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.rotation = rotation

        if points:
            self.points = points
        else:
            center_x = (
                self.left
                + ((self.width / 2) * np.cos(np.radians(self.rotation)))
                - ((self.height / 2) * np.sin(np.radians(self.rotation)))
            )
            center_y = (
                self.top
                + ((self.width / 2) * np.sin(np.radians(self.rotation)))
                + ((self.height / 2) * np.cos(np.radians(self.rotation)))
            )
            top_r_x = (
                center_x
                + ((self.width / 2) * np.cos(np.radians(self.rotation)))
                + ((self.height / 2) * np.sin(np.radians(self.rotation)))
            )
            top_r_y = (
                center_y
                + ((self.width / 2) * np.sin(np.radians(self.rotation)))
                - ((self.height / 2) * np.cos(np.radians(self.rotation)))
            )
            bot_r_x = (
                center_x
                + ((self.width / 2) * np.cos(np.radians(self.rotation)))
                - ((self.height / 2) * np.sin(np.radians(self.rotation)))
            )
            bot_r_y = (
                center_y
                + ((self.width / 2) * np.sin(np.radians(self.rotation)))
                + ((self.height / 2) * np.cos(np.radians(self.rotation)))
            )
            bot_l_x = (
                center_x
                - ((self.width / 2) * np.cos(np.radians(self.rotation)))
                - ((self.height / 2) * np.sin(np.radians(self.rotation)))
            )
            bot_l_y = (
                center_y
                - ((self.width / 2) * np.sin(np.radians(self.rotation)))
                + ((self.height / 2) * np.cos(np.radians(self.rotation)))
            )

            self.points = [
                (self.left, self.top),
                (top_r_x, top_r_y),
                (bot_r_x, bot_r_y),
                (bot_l_x, bot_l_y),
            ]

    @classmethod
    def from_json(cls, j: dict):
        """Create a rectangle roi from a json object.

        :param j: The roi input serialized as json.
        """

        if Roi.ShapeType(j["type"]) != Roi.ShapeType.rectangle:
            raise ValueError("Unexpected shape type for roi input")

        return cls(
            j["groupKey"],
            j["name"],
            j["stroke"],
            j["top"],
            j["left"],
            j["width"],
            j["height"],
            j["rotation"],
        )

    def __eq__(self, other):
        """Determine if two rectangle rois are equal."""
        return Roi.__eq__(self, other) and (
            self.top == other.top
            and self.left == other.left
            and self.width == other.width
            and self.height == other.height
            and self.rotation == other.rotation
            and np.allclose(self.points, other.points)
        )

    def __str__(self):
        """Create str representation of rectangle roi."""
        return f"Rectangle(group_key={self.group_key},name={self.name},stroke={self.stroke},top={self.top},left={self.left},width={self.width},height={self.height},rotation={self.rotation},points={self.points})"


class EllipseRoi(Roi):
    """An roi with an ellipse shape."""

    def __init__(
        self,
        group_key: str,
        name: str,
        stroke: Union[str, tuple],
        center: tuple,
        rx: Union[int, float],
        ry: Union[int, float],
        rotation: Union[int, float],
    ):
        """Create instance of an ellipse roi.

        :param group_key: The key of the group that the roi belongs to.
        :param name: The name of the roi given by the user.
        :param stroke: The color of the roi given by the user.
        :param center: The center point of the ellipse.
        :param rx: Half of the major axis of the ellipse.
        :param ry: Half of the minor axis of the ellipse.
        :param rotation: The angle of rotation of the ellipse.
        """
        Roi.__init__(
            self,
            shape_type=Roi.ShapeType.ellipse,
            group_key=group_key,
            name=name,
            stroke=stroke,
        )
        self.center = center
        self.rx = float(rx)
        self.ry = float(ry)
        self.rotation = float(rotation)

    @classmethod
    def from_json(cls, j: dict):
        """Create a rectangle roi from a json object.

        :param j: The roi input serialized as json.
        """
        if Roi.ShapeType(j["type"]) != Roi.ShapeType.ellipse:
            raise ValueError("Unexpected shape type for roi input")

        return cls(
            j["groupKey"],
            j["name"],
            j["stroke"],
            (j["center"]["x"], j["center"]["y"]),
            j["rx"],
            j["ry"],
            j["rotation"],
        )

    def __eq__(self, other):
        """Determine if two ellipse rois are equal."""
        return Roi.__eq__(self, other) and (
            self.center == other.center
            and self.rx == other.rx
            and self.ry == other.ry
            and self.rotation == other.rotation
        )

    def __str__(self):
        """Create str representation of ellipse roi."""
        return f"Ellipse(group_key={self.group_key},name={self.name},stroke={self.stroke},center={self.center},rx={self.rx},ry={self.ry},rotation={self.rotation})"


class ContourRoi(Roi):
    """An roi with a contour shape."""

    def __init__(
        self,
        group_key: str,
        name: str,
        stroke: Union[str, tuple],
        points: List[tuple],
        closed: bool,
    ):
        """Create instance of a contour roi.

        :param group_key: The key of the group that the roi belongs to.
        :param name: The name of the roi given by the user.
        :param stroke: The color of the roi given by the user.
        :param points: The points of the contour.
        :param closed: Flag indicating whether the contour is closed or not.
        """
        Roi.__init__(
            self,
            shape_type=Roi.ShapeType.contour,
            group_key=group_key,
            name=name,
            stroke=stroke,
        )
        self.points = points
        self.closed = closed

    @classmethod
    def from_json(cls, j: dict):
        """Create a contour roi from a json object.

        :param j: The roi input serialized as json.
        """
        if Roi.ShapeType(j["type"]) != Roi.ShapeType.contour:
            raise ValueError("Unexpected shape type for roi input")

        return cls(
            j["groupKey"],
            j["name"],
            j["stroke"],
            [(p["x"], p["y"]) for p in j["points"]],
            j["closed"],
        )

    def __eq__(self, other):
        """Determine if two contour rois are equal."""
        return Roi.__eq__(self, other) and (
            np.allclose(self.points, other.points)
            and self.closed == other.closed
        )

    def __str__(self):
        """Create str representation of contour roi."""
        return f"Contour(group_key={self.group_key},name={self.name},stroke={self.stroke},points={self.points},closed={self.closed})"


class BoundingBoxRoi(Roi):
    """An roi for a bounding box.

    Bounding box acts like a cropping rectangle of the FOV, and can contain other ROI groups drawn.
    """

    def __init__(
        self,
        group_key: str,
        name: str,
        stroke: Union[str, tuple],
        top: Union[int, float],
        left: Union[int, float],
        width: Union[int, float],
        height: Union[int, float],
    ):
        """Create instance of a bounding box roi.

        :param group_key: The key of the group that the roi belongs to.
        :param name: The name of the roi given by the user.
        :param stroke: The color of the roi given by the user.
        :param top: The top coordinate of the bounding box.
        :param left: The left coordinate of the bounding box.
        :param width: The width of the bounding box.
        :param height: The height of the bounding box.
        """
        Roi.__init__(
            self,
            shape_type=Roi.ShapeType.bounding_box,
            group_key=group_key,
            name=name,
            stroke=stroke,
        )
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    @classmethod
    def from_json(cls, j):
        """Create a bounding box roi from a json object.

        :param j: The roi input serialized as json.
        """
        if Roi.ShapeType(j["type"]) != Roi.ShapeType.bounding_box:
            raise ValueError("Unexpected shape type for roi input")

        return cls(
            j["groupKey"],
            j["name"],
            j["stroke"],
            j["top"],
            j["left"],
            j["width"],
            j["height"],
        )

    def __eq__(self, other):
        """Determine if two bounding box rois are equal."""
        return Roi.__eq__(self, other) and (
            self.top == other.top
            and self.left == other.left
            and self.width == other.width
            and self.height == other.height
        )

    def __str__(self):
        """Create str representation of bounding box roi."""
        return f"BoundingBox(group_key={self.group_key},name={self.name},stroke={self.stroke},top={self.top},left={self.left},width={self.width},height={self.height})"


def get_rois_from_json(j: dict):
    """Get rois from json serialized roi inputs sent by the FE.

    :param j: The json object to convert.
    """

    rois = []
    for el in j:
        shape_type = Roi.ShapeType(el["type"])
        if shape_type == Roi.ShapeType.polygon:
            rois.append(PolygonRoi.from_json(el))
        elif shape_type == Roi.ShapeType.rectangle:
            rois.append(RectangleRoi.from_json(el))
        elif shape_type == Roi.ShapeType.ellipse:
            rois.append(EllipseRoi.from_json(el))
        elif shape_type == Roi.ShapeType.contour:
            rois.append(ContourRoi.from_json(el))
        elif shape_type == Roi.ShapeType.bounding_box:
            rois.append(BoundingBoxRoi.from_json(el))
    return rois
