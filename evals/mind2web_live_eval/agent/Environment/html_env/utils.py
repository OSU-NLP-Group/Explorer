from typing import TypedDict, List
from enum import IntEnum
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union

import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image
import nltk
import random
import os
from multiprocessing import Pool
import re


@dataclass
class DetachedPage:
    url: str
    content: str  # html


class ElementNode(TypedDict):
    nodeId: int  # Element ID
    childIds: List[int]  # List of child element IDs
    siblingId: int  # Sibling element ranking
    twinId: int  # Same tag element ranking
    tagName: str  # Element
    attributes: dict  # Element attributes
    text: str  # Text attribute
    parentId: int  # Parent element
    htmlContents: str  # All information of the element
    depth: int  # Depth


TagNameList = [
    "button",
    "a",
    "input",
    "select",
    "textarea",
    "option",
    "datalist",
    "label",
    "div",
    "span",
    "p",
    "th",
    "tr",
    "td",
    "ul",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "filter-chip",
    "sup",
    "select-label",
    "optgroup",
]

MapTagNameList = ["span", "h1", "h2", "h3", "h4", "h5", "h6", "div", "li", "ul", "p"]

DelTagNameList = [
    "script",  # del
    "noscript",  # del
    "style",  # del
    "link",  # del
    "meta",  # del
]


ConditionTagNameList = ["span", "td", "th", "tr", "li", "div", "label", "filter-chip"]


TypeList = ["submit"]


def stringfy_selector(string: str):
    special_chars = "#.>+~[]():*^$|=@'"
    string = string.replace("\t", " ").replace("\n", " ").lstrip().rstrip()
    string = " ".join(string.split())
    for char in special_chars:
        string = string.replace(char, "\\" + char)
    string = ".".join(string.split(" "))
    if string[0].isdigit():
        string = f"\\{'{:X}'.format(ord(string[0]))}" + " " + string[1:]
    return string


def stringfy_value(string):
    special_chars = "#.>+~[]():*^$|=@'"
    for char in special_chars:
        string = string.replace(char, "\\" + char)
    return rf"{string}"


__all__ = [
    "ElementNode",
    "TagNameList",
    "DelTagNameList",
    "ConditionTagNameList",
    "TypeList",
    "stringfy_selector",
    "stringfy_value",
]


@dataclass
class DetachedPage:
    url: str
    content: str  # html


@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]


Observation = str | npt.NDArray[np.uint8]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]


from typing import TypedDict, List
from enum import IntEnum


class ElementNode(TypedDict):
    nodeId: int  # Element ID
    childIds: List[int]  # List of child element IDs
    siblingId: int  # Sibling element ranking
    twinId: int  # Same tag element ranking
    tagName: str  # Element
    attributes: dict  # Element attributes
    text: str  # Text attribute
    parentId: int  # Parent element
    htmlContents: str  # All information of the element
    depth: int  # Depth


TagNameList = [
    "button",
    "a",
    "input",
    "select",
    "textarea",
    "option",
    "datalist",
    "label",
    "div",
    "span",
    "p",
    "th",
    "tr",
    "td",
    "ul",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "filter-chip",
    "sup",
    "select-label",
    "optgroup",
]

MapTagNameList = ["span", "h1", "h2", "h3", "h4", "h5", "h6", "div", "li", "ul", "p"]

DelTagNameList = [
    "script",  # del
    "noscript",  # del
    "style",  # del
    "link",  # del
    "meta",  # del
]


ConditionTagNameList = ["span", "td", "th", "tr", "li", "div", "label", "filter-chip"]


TypeList = ["submit"]


def stringfy_selector(string: str):
    special_chars = "#.>+~[]():*^$|=@'"
    string = string.replace("\t", " ").replace("\n", " ").lstrip().rstrip()
    string = " ".join(string.split())
    for char in special_chars:
        string = string.replace(char, "\\" + char)
    string = ".".join(string.split(" "))
    if string[0].isdigit():
        string = f"\\{'{:X}'.format(ord(string[0]))}" + " " + string[1:]
    return string


def stringfy_value(string):
    special_chars = "#.>+~[]():*^$|=@'"
    for char in special_chars:
        string = string.replace(char, "\\" + char)
    return rf"{string}"
