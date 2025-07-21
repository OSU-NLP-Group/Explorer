import json
import re
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any, Optional, TypedDict, Union
from urllib.parse import urljoin, urlparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from beartype import beartype
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import CDPSession, Page, ViewportSize
from playwright.async_api import CDPSession as CDPSessionAsync, Page as PageAsync, ViewportSize as ViewportSizeAsync
from typing import Any, Dict, TypedDict, Union # dict, list, tuple
import logging
from collections import Counter

from evals.mind2web_orig_eval.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    IGNORED_ACTREE_PROPERTIES,
    UTTERANCE_MAX_LENGTH,
)

from .utils import (
    AccessibilityTree,
    BrowserConfig,
    BrowserInfo,
    Observation,
    png_bytes_to_numpy,
)

Observation = str | npt.NDArray[np.uint8]

class ObservationProcessor:
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
    }

@beartype
def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))

def convert_to_corners(box):
    x_center, y_center, width, height = box
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    return [x_min, y_min, width, height]

def calculate_iou(box1, box2):
    # Calculate the (x, y) coordinates of the intersection rectangle
    # [x_left, y_left, w, h]
    # box2[2] -= box2[0]
    # box2[3] -= box2[1]

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    # If the boxes don't overlap, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # Calculate the union area by using the formula: union(A, B) = A + B - intersection(A, B)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def find_closest_center_coordinate(box_coordinate, interactive_rects):
    
    thresh = 0.9
    closest_coordinate_idx = None
    
    for i, rect in enumerate(interactive_rects):
        coord = rect[3]
        
        iou = calculate_iou(box_coordinate, coord)
        
        # print(f"box1:{box_coordinate}, box2:{coord}, IOU: {iou}")

        if iou > thresh:
            closest_coordinate_idx = i
            break
    
    return closest_coordinate_idx

def remove_extra_eol(text):
    # Replace EOL symbols
    text = text.replace('\n', ' ')
    return re.sub(r'\s{2,}', ' ', text)


def get_first_line(s):
    first_line = s.split('\n')[0]
    tokens = first_line.split()
    if len(tokens) > 8:
        return ' '.join(tokens[:8]) + '...'
    else:
        return first_line
    
def get_element_description(element, tag_name, role_value, type_value):
    '''
         Asynchronously generates a descriptive text for a web element based on its tag type.
         Handles various HTML elements like 'select', 'input', and 'textarea', extracting attributes and content relevant to accessibility and interaction.
    '''

    salient_attributes = [
        "alt",
        "aria-describedby",
        "aria-label",
        "aria-role",
        "input-checked",
        # "input-value",
        "label",
        "name",
        "option_selected",
        "placeholder",
        "readonly",
        "text-value",
        "title",
        "value",
    ]

    parent_value = "parent_node: "
    parent_locator = element.locator('xpath=..')
    num_parents = parent_locator.count()
    if num_parents > 0:
        # only will be zero or one parent node
        parent_text = (parent_locator.inner_text(timeout=0) or "").strip()
        if parent_text:
            parent_value += parent_text
    parent_value = remove_extra_eol(get_first_line(parent_value)).strip()
    if parent_value == "parent_node:":
        parent_value = ""
    else:
        parent_value += " "

    if tag_name == "select":
        text1 = "Selected Options: "
        text2 = ""
        text3 = " - Options: "
        text4 = ""

        text2 = element.evaluate(
            "select => select.options[select.selectedIndex].textContent", timeout=0
        )

        if text2:
            options = element.evaluate("select => Array.from(select.options).map(option => option.text)",
                                             timeout=0)
            text4 = " | ".join(options)

            if not text4:
                text4 = element.text_content(timeout=0)
                if not text4:
                    text4 = element.inner_text(timeout=0)

            return parent_value+text1 + remove_extra_eol(text2.strip()) + text3 + text4

    input_value = ""

    none_input_type = ["submit", "reset", "checkbox", "radio", "button", "file"]

    if tag_name == "input" or tag_name == "textarea":
        if role_value not in none_input_type and type_value not in none_input_type:
            text1 = "input value="
            text2 = element.input_value(timeout=0)
            if text2:
                input_value = text1 + "\"" + text2 + "\"" + " "

    text_content = element.text_content(timeout=0)
    text = (text_content or '').strip()
    if text:
        text = remove_extra_eol(text)
        if len(text) > 80:
            text_content_in = element.inner_text(timeout=0)
            text_in = (text_content_in or '').strip()
            if text_in:
                return input_value + remove_extra_eol(text_in)
        else:
            return input_value + text

    # get salient_attributes
    text1 = ""
    for attr in salient_attributes:
        attribute_value = element.get_attribute(attr, timeout=0)
        if attribute_value:
            text1 += f"{attr}=" + "\"" + attribute_value.strip() + "\"" + " "

    text = (parent_value + text1).strip()
    if text:
        return input_value + remove_extra_eol(text.strip())


    # try to get from the first child node
    first_child_locator = element.locator('xpath=./child::*[1]')

    num_childs = first_child_locator.count()
    if num_childs>0:
        for attr in salient_attributes:
            attribute_value = first_child_locator.get_attribute(attr, timeout=0)
            if attribute_value:
                text1 += f"{attr}=" + "\"" + attribute_value.strip() + "\"" + " "

        text = (parent_value + text1).strip()
        if text:
            return input_value + remove_extra_eol(text.strip())

    return None


def get_element_data(element, tag_name):
    tag_name_list = ['a', 'button',
                     'input',
                     'select', 'textarea', 'adc-tab']

    # await aprint(element,tag_name)
    if element.is_hidden(timeout=0) or element.is_disabled(timeout=0):
        return None

    tag_head = ""
    real_tag_name = ""
    if tag_name in tag_name_list:
        tag_head = tag_name
        real_tag_name = tag_name
    else:
        real_tag_name = element.evaluate("element => element.tagName.toLowerCase()", timeout=0)
        if real_tag_name in tag_name_list:
            # already detected
            return None
        else:
            tag_head = real_tag_name

    role_value = element.get_attribute('role', timeout=0)
    type_value = element.get_attribute('type', timeout=0)
    # await aprint("start to get element description",element,tag_name )
    description = get_element_description(element, real_tag_name, role_value, type_value)
    if not description:
        return None

    rect = element.bounding_box() or {'x': 0, 'y': 0, 'width': 0, 'height': 0}

    if role_value:
        tag_head += " role=" + "\"" + role_value + "\""
    if type_value:
        tag_head += " type=" + "\"" + type_value + "\""

    box_model = [rect['x'], rect['y'], rect['width'], rect['height']]
    center_point = ((box_model[0] + box_model[2]) / 2, (box_model[1] + box_model[3]) / 2)
    selector = element


    return [center_point, description, tag_head, box_model, selector, real_tag_name]

def get_interactive_elements_with_playwright(page):
    interactive_elements_selectors = [
        'select', '[role="radio"]', '[role="option"]', '[role="combobox"]',
        '[role="listbox"]', '[role="menu"]',
        '[type="radio"]', '[type="combobox"]', '[type="listbox"]',
        '[type="menu"]',
    ]

    tasks = []

    seen_elements = set()
    for selector in interactive_elements_selectors:
        locator = page.locator(selector)
        element_count = locator.count()
        for index in range(element_count):
            element = locator.nth(index)
            tag_name = selector.replace(":not([tabindex=\"-1\"])", "")
            tag_name = tag_name.replace(":not([contenteditable=\"false\"])", "")
            task = get_element_data(element, tag_name)

            tasks.append(task)

    interactive_elements = []
    for i in tasks:
        if i:
            if i[0] in seen_elements:
                continue
            else:
                seen_elements.add(i[0])
                interactive_elements.append(i)
    return interactive_elements

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



class ImageObservationProcessor(ObservationProcessor):
    def __init__(
        self,
        args,
        observation_type: str,
        viewport_size,
    ):  
        self.args = args
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.viewport_size = viewport_size
        self.meta_data = create_empty_metadata()

    def get_page_bboxes(self, page: Page) -> list[list[float]]:
        """JavaScript code to return bounding boxes and other metadata from HTML elements."""
        js_script = """
        (() => {
            const interactableSelectors = [
                'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
                '.btn'
            ];

            const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
            const modifiedTextSelectors = textSelectors.map(selector =>
                `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
            );

            const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
            const elements = document.querySelectorAll(combinedSelectors.join(', '));

            const pixelRatio = window.devicePixelRatio;
            let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable\\n";
            let counter = 1;

            elements.forEach(element => {
                const rect = element.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                let altText = element.getAttribute('alt') || '';
                altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
                const classList = element.className || '';
                const id = element.id || '';
                let textContent = element.textContent || '';
                textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent

                // Determine if the element is interactable
                const isInteractable = interactableSelectors.some(selector => element.matches(selector));

                const dataString = [
                    counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                    (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                    (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                    altText, classList, id, textContent, isInteractable
                ].map(value => `"${value}"`).join(",");

                csvContent += dataString + "\\n";
                counter++;
            });

            return csvContent;
        })();
        """
        # Save the bbox as a CSV
        csv_content = page.evaluate(js_script)
        return csv_content

    def draw_bounding_boxes(
        self,
        data_string,
        screenshot_img,
        viewport_size=None,
        add_ids=True,
        bbox_color=None,
        min_width=8, # 8
        min_height=8, # 8
        bbox_padding=0,
        bbox_border=2,
        plot_ids=None,
        intent=None,
    ):
        """
        min_width and min_height: Minimum dimensions of the bounding box to be plotted.
        """
        # Read CSV data
        df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')
        df["Area"] = df["Width"] * df["Height"]
        # Remove bounding boxes that are clipped.
        b_x, b_y = (
            self.browser_config["win_left_bound"],
            self.browser_config["win_upper_bound"],
        )
        # Filter out bounding boxes that are not in the viewport, and are too large
        # import pdb; pdb.set_trace()
        if viewport_size is not None:
            init_num_bboxes = len(df)
            df = df[
                (df["Bottom"] - b_y >= 0)
                & (df["Top"] - b_y <= viewport_size["height"])
                & (df["Right"] - b_x >= 0)
                & (df["Left"] - b_x <= viewport_size["width"])
            ]
            # df = df[
            #     (df["Top"] - b_y >= 0)
            #     & (df["Bottom"] - b_y <= viewport_size["height"])
            #     & (df["Left"] - b_x >= 0)
            #     & (df["Right"] - b_x <= viewport_size["width"])
            # ]
            viewport_area = viewport_size["width"] * viewport_size["height"]
            # Filter out bounding boxes that too large (more than 80% of the viewport)
            df = df[df["Area"] <= 0.8 * viewport_area]
            # print(f"Filtered out {init_num_bboxes - len(df)} bounding boxes that are not in the viewport or too large.")

        # Open the screenshot image
        img = screenshot_img.copy()
        draw = ImageDraw.Draw(img)

        # Load a TTF font with a larger size
        
        font_path = "media/SourceCodePro-SemiBold.ttf"
        font_size, padding = 16, 2
        # font = ImageFont.truetype(font_path, font_size)
        font = ImageFont.load_default()

        # Create a color cycle using one of the categorical color palettes in matplotlib
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bbox_id2visid = {}
        bbox_id2desc = {}
        index = 0
        id2center = {}
        existing_text_rectangles = []
        text_to_draw = []
        # Provide [id] textContent inputs to the model as text.
        text_content_elements = []
        text_content_text = set()  # Store text of interactable elements

        # Iterate through each row in the CSV and draw bounding boxes
        for _, row in df.iterrows():
            if not row["Interactable"]:
                content = ""
                # Add image alt-text to the text representation.
                if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                    content += row["Alt"]
                # Add HTML textContent (if any) to the text representation.
                if pd.notna(row["TextContent"]):
                    content += (
                        row["TextContent"]
                        .strip()
                        .replace("\n", "")
                        .replace("\t", "")
                    )[
                        :200
                    ]  # Limit to 200 characters to avoid having too much text

                # Check if the text is a CSS selector
                if content and not (
                    content.startswith(".") and "{" in content
                ):
                    # Add elements which are not interactable as StaticText
                    if content not in text_content_text:
                        text_content_elements.append(
                            f"[] [StaticText] [{content}]"
                        )
                        text_content_text.add(content)
                continue

            if (plot_ids is not None) and (row["ID"] not in plot_ids):
                continue
            
            if intent is not None:
                intent_match = any([inte in str(row["TextContent"]) for inte in intent.split(" ")])
                if not intent_match:
                    continue
            unique_id = str(index + 1)
            bbox_id2visid[
                row["ID"]
            ] = unique_id  # map the bounding box ID to the unique character ID
            top, right, bottom, left, width, height = (
                row["Top"],
                row["Right"],
                row["Bottom"],
                row["Left"],
                row["Width"],
                row["Height"],
            )
            left, right, top, bottom = left - b_x, right - b_x, top - b_y, bottom - b_y
            id2center[unique_id] = ((left + right) / 2, (bottom + top) / 2, width, height)

            if width >= min_width and height >= min_height:
                # Get the next color in the cycle
                color = bbox_color or color_cycle[index % len(color_cycle)]
                draw.rectangle(
                    [
                        left - bbox_padding,
                        top - bbox_padding,
                        right + bbox_padding,
                        bottom + bbox_padding,
                    ],
                    outline=color,
                    width=bbox_border,
                )
                bbox_id2desc[row["ID"]] = color

                # Draw the text on top of the rectangle
                if add_ids:
                    # Calculate list of possible text positions
                    text_positions = [
                        (left - font_size, top - font_size),  # Top-left corner
                        (
                            left,
                            top - font_size,
                        ),  # A little to the right of the top-left corner
                        (right, top - font_size),  # Top-right corner
                        (
                            right - font_size - 2 * padding,
                            top - font_size,
                        ),  # A little to the left of the top-right corner
                        (left - font_size, bottom),  # Bottom-left corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                    ]
                    text_width = draw.textlength(unique_id, font=font)
                    text_height = font_size  # Assume the text is one line

                    if viewport_size is not None:
                        for text_position in text_positions:
                            new_text_rectangle = [
                                text_position[0] - padding,
                                text_position[1] - padding,
                                text_position[0] + text_width + padding,
                                text_position[1] + text_height + padding,
                            ]

                            # Check if the new text rectangle is within the viewport
                            if (
                                new_text_rectangle[0] >= 0
                                and new_text_rectangle[1] >= 0
                                and new_text_rectangle[2]
                                <= viewport_size["width"]
                                and new_text_rectangle[3]
                                <= viewport_size["height"]
                            ):
                                # If the rectangle is within the viewport, check for overlaps
                                overlaps = False
                                for (
                                    existing_rectangle
                                ) in existing_text_rectangles:
                                    if self.rectangles_overlap(
                                        new_text_rectangle,
                                        existing_rectangle,
                                        padding * 2,
                                    ):
                                        overlaps = True
                                        break

                                if not overlaps:
                                    break
                            else:
                                # If the rectangle is outside the viewport, try the next position
                                continue
                    else:
                        # If none of the corners work, move the text rectangle by a fixed amount
                        text_position = (
                            text_positions[0][0] + padding,
                            text_positions[0][1],
                        )
                        new_text_rectangle = [
                            text_position[0] - padding,
                            text_position[1] - padding,
                            text_position[0] + text_width + padding,
                            text_position[1] + text_height + padding,
                        ]

                    existing_text_rectangles.append(new_text_rectangle)
                    text_to_draw.append(
                        (new_text_rectangle, text_position, unique_id, color)
                    )

                    content = ""
                    if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                        content += row["Alt"]
                    if pd.notna(row["TextContent"]):
                        content += (
                            row["TextContent"]
                            .strip()
                            .replace("\n", "")
                            .replace("\t", "")
                        )[
                            :200
                        ]  # Limit to 200 characters
                    text_content_elements.append(
                        f"[{unique_id}] [{row['Element']}] [{content}]"
                    )
                    if content in text_content_text:
                        # Remove text_content_elements with content
                        text_content_elements = [
                            element
                            for element in text_content_elements
                            if element.strip() != content
                        ]
                    text_content_text.add(content)

            index += 1

        for text_rectangle, text_position, unique_id, color in text_to_draw:
            # Draw a background rectangle for the text
            draw.rectangle(text_rectangle, fill=color)
            draw.text(text_position, unique_id, font=font, fill="white")

        content_str = "\n".join(text_content_elements)
        return img, id2center, content_str

    def rectangles_overlap(self, rect1, rect2, padding):
        """
        Check if two rectangles overlap.
        Each rectangle is represented as a list [x1, y1, x2, y2].
        """
        return not (
            rect1[2] < rect2[0] + padding
            or rect1[0] > rect2[2] - padding
            or rect1[1] > rect2[3] - padding
            or rect1[3] < rect2[1] + padding
        )

    def process(self, page: Page, client: CDPSession, intent) -> npt.NDArray[np.uint8]:
        page.wait_for_load_state("load", timeout=5000)
        # page.wait_for_load_state("load", timeout=2000)
        # page.wait_for_load_state("load", timeout=500)
        try:
            # breakpoint()
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=2000)
            browser_info = self.fetch_browser_info(page, client)

        self.browser_config = browser_info["config"]

        if self.observation_type == "image_som":
            # Produce the SoM image, with bounding boxes
            try:
                # import pdb; pdb.set_trace()
                screenshot_bytes = page.screenshot() # full_page=True
                # screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page) #  js code, output som_bboxes is csv string
                screenshot_img = Image.open(BytesIO(screenshot_bytes))

                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                    intent=intent,
                )
                
                # bbox_img.save(f"test_{page.url.split('/')[-1]}.png")
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
            except:
                page.wait_for_event("load")
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
        else:
            try:
                screenshot = png_bytes_to_numpy(page.screenshot())
            except:
                page.wait_for_event("load")
                screenshot = png_bytes_to_numpy(page.screenshot())
            return screenshot, ""

    # @profile
    def process_new(self, page: Page, client: CDPSession, intent, add_class=False, use_id_selector=False, goal=None, omit_task_desc_ele=False, opt=None) -> npt.NDArray[np.uint8]:
        import os
        from .set_of_mark import add_set_of_mark
        page.wait_for_load_state("load", timeout=5000)

        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page, client)

        self.browser_config = browser_info["config"]
        
        # import pdb; pdb.set_trace();

        # Listen for console events
        page.on("console", lambda msg: print(f"Console log: {msg.text}"))

        if add_class:
            # print('add_class_id')
            with open(os.path.join("evals", "miniwob", "page_script_3.js"), "rt") as fh:
                page.evaluate(fh.read())
        else:
            with open(os.path.join("evals", "miniwob", "page_script.js"), "rt") as fh:
                page.evaluate(fh.read())

        rects = page.evaluate("MultimodalWebSurfer.getInteractiveRects();")

        logging.info('len(rects) = {}'.format(len(rects)))
        logging.info('rects = {}'.format(rects))

        id2center = {}

        for box_id in rects:
            box = rects[box_id]
            # print(len(box["rects"]))
            if len(box["rects"])>0:
                id2center[box_id] = (box["rects"][0]["x"] + box["rects"][0]["width"]/2, box["rects"][0]["y"] + box["rects"][0]["height"]/2, box["rects"][0]["width"], box["rects"][0]["height"])
            
        self.som_id_info = id2center
        self.rects = rects

        self.meta_data["obs_nodes_info"] = id2center

        # logging.info("id2center = {}".format(id2center))
        
        
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(page.screenshot(), rects, draw=False)
        
        # logging.info(f"Visible Rects: {visible_rects}")

        '''
        self.som_id_info_visible = {ele_id : self.som_id_info[ele_id] for ele_id in visible_rects}

        self.som_id_info_visible_rev = defaultdict(list)
        for ele_id in self.som_id_info_visible:
            self.som_id_info_visible_rev[self.som_id_info_visible[ele_id]].append(ele_id)

        bbox_counter = Counter(self.som_id_info_visible.values())
        elements_to_remove = set()

        # print(bbox_counter.most_common(10))

        for coord, count in bbox_counter.items():
            if count==1:
                continue
            
            # all_tag_names = []
            # all_tags = []
            # all_ele_ids = []

            all_elements = self.som_id_info_visible_rev[coord]
            for ele_id in all_elements:
                box = rects[ele_id]
                # tag_name = box["tag_name"].upper()
                alt_text = box["aria-name"].replace("\n", " ").replace("\t", " ")
                # all_tag_names.append(alt_text)
                # all_tags.append(tag_name)
                # all_ele_ids.append(ele_id)

                if len(alt_text)>0:
                    elements_to_remove.update(set(all_elements)-set([ele_id]))
                    break

            # print(f'coord = {coord}, all_tags = {all_tags}, all_tag_names = {all_tag_names}, all_ele_ids = {all_ele_ids}')

        # print('elements_to_remove = {}'.format(elements_to_remove))
        visible_rects = [x for x in visible_rects if x not in elements_to_remove]
        
        for x in elements_to_remove:
            del rects[x]
        
        # logging.info('new visible_rects coord = {}'.format({x: self.som_id_info_visible[x] for x in visible_rects}))
        '''
        # create a new accessibility tree
        acc_tree = []

        # for box_id in visible_rects:
        for box_id in rects:
            box = rects[box_id]
            alt_text = box["aria-name"].replace("\n", " ").replace("\t", " ")
            tag_name = box["tag_name"].upper()

            if tag_name == 'LI':
                continue

            if opt.omit_empty_div and tag_name == 'DIV' and alt_text=="":
                if box_id in visible_rects:
                    visible_rects.remove(box_id)
                continue

            if omit_task_desc_ele and alt_text == goal:
                if box_id in visible_rects:
                    visible_rects.remove(box_id)
                continue
            
            acc_tree.append("[{}] [{}] [{}]".format(box_id, tag_name, alt_text))

        acc_tree = "\n".join(acc_tree)
        # print(acc_tree)
        '''
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(page.screenshot(), rects)
        '''
        # recompute som_id_info_visible
        self.som_id_info_visible = {ele_id : self.som_id_info[ele_id] for ele_id in visible_rects}
        
        som_screenshot = np.array(som_screenshot)
        return som_screenshot, acc_tree, rects

    async def process_new_async(self, page: PageAsync, client: CDPSessionAsync, intent) -> npt.NDArray[np.uint8]:
        import os
        from .set_of_mark import add_set_of_mark
        
        # logging.info('inside process_new_async')
        await page.wait_for_load_state("load", timeout=5000)

        try:
            browser_info = await self.fetch_browser_info_async(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = await self.fetch_browser_info_async(page, client)

        self.browser_config = browser_info["config"]

        # with open(os.path.join("web_traj_gen", "page_script.js"), "rt") as fh:
        with open(os.path.join("mind2web_live_eval", "agent", "Utils", "page_script_2.js"), "rt") as fh:
            await page.evaluate(fh.read())
        rects = await page.evaluate("MultimodalWebSurfer.getInteractiveRects();")

        # print(rects)
        
        

        id2center = {}

        for box_id in rects:
            box = rects[box_id]
            id2center[box_id] = (box["rects"][0]["x"] + box["rects"][0]["width"]/2, box["rects"][0]["y"] + box["rects"][0]["height"]/2, box["rects"][0]["width"], box["rects"][0]["height"])
        self.som_id_info = id2center
        self.meta_data["obs_nodes_info"] = id2center

        # print("id2center: ", id2center)
        page_screenshot = await page.screenshot()
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(page_screenshot, rects)
        w, h = som_screenshot.size

        # logging.info(f"Visible Rects: {visible_rects}")

        # create a new accessibility tree
        acc_tree = []

        for box_id in visible_rects:
            box = rects[box_id]
            alt_text = box["aria-name"].replace("\n", " ").replace("\t", " ")
            tag_name = box["tag_name"].upper()
            acc_tree.append("[{}] [{}] [{}]".format(box_id, tag_name, alt_text))

        acc_tree = "\n".join(acc_tree)
        # print(acc_tree)

        '''
        bboxes_visible = [rects[v] for v in visible_rects]
        bboxes_visible_ratio = [[b['rects'][0]['x']/w, b['rects'][0]['y']/h, b['rects'][0]['width']/w, b['rects'][0]['height']/h] for b in bboxes_visible]
        # convert from xywh to xcycwh
        bboxes_visible_ratio = [[b[0] + b[2]/2, b[1] + b[3]/2, b[2], b[3]] for b in bboxes_visible_ratio]
        '''
        '''
        screenshot_bytes = page.screenshot()
        # som_screenshot.save('/home/yadonglu/sandbox/data/orca/parsed_html_demo_img_result_mmwebsurfer.png')
        som_screenshot_bytes = BytesIO()
        som_screenshot.save(som_screenshot_bytes, format="PNG")
        som_screenshot_bytes = som_screenshot_bytes.getvalue()
        '''
        som_screenshot = np.array(som_screenshot)

        # print('som_screenshot = ', som_screenshot)
        # print('acc_tree = ', acc_tree)

        # print('type(som_screenshot) = ', type(som_screenshot))
        # print('type(acc_tree) = ', type(acc_tree))

        return som_screenshot, acc_tree
    
    @beartype
    def fetch_browser_info(
        self,
        page,
        client,
        # page: Page,
        # client: CDPSession,
    ) -> BrowserInfo:
        # print('flag 1')

        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # print('flag 2')

        # logging.info('tree = {}'.format(tree))
        # await tree()
        
        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # print('flag 3')

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        # print('flag 4')

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    @beartype
    async def fetch_browser_info_async(
        self,
        page,
        client,
        # page: Page,
        # client: CDPSession,
    ) -> BrowserInfo:
        # print('flag 1')

        # extract domtree

        tree = await client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )
        # tree = await client.send(
        #     "DOMSnapshot.captureSnapshot",
        # )
        # await the previous line
        # await tree

        # await client.detach()

        # Evaluate JavaScript to get the DOM tree
        # tree = await page.evaluate("""() => {
        #     function getDomTree(element) {
        #         return {
        #             tagName: element.tagName,
        #             children: Array.from(element.children).map(child => getDomTree(child))
        #         };
        #     }
        #     return getDomTree(document.documentElement);
        # }""")

        # print('flag 2')

        # logging.info('tree = {}'.format(tree))
        # await tree()
        
        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # print('flag 3')

        # extract browser info
        win_upper_bound = await page.evaluate("window.pageYOffset")
        win_left_bound = await page.evaluate("window.pageXOffset")
        win_width = await page.evaluate("window.screen.width")
        win_height = await page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = await page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        # print('flag 4')

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info
    
    @beartype
    def get_element_center(self, element_id: str) -> tuple[float, float]:
        if not self.observation_type == "image_som":
            raise ValueError(
                "get_element_center() is only supported for 'image_som' observation type."
            )

        browser_config = self.browser_config

        # print(self.som_id_info[element_id])
        center_x, center_y, width, height = self.som_id_info[element_id]
        # import pdb; pdb.set_trace()

        # print(f"center_x: {center_x}, center_y: {center_y}")
        return (
            center_x,
            center_y,
        )
        # return (
        #     center_x / self.viewport_size["width"],
        #     center_y / self.viewport_size["height"],
        # )

class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = (
            create_empty_metadata()
        )  # use the store meta data of this observation type

        if self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            self.captioning_fn = captioning_fn
            # Cache captions.
            self.url2caption = {}

    @beartype
    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ) -> BrowserInfo:
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    @beartype
    @staticmethod
    def partially_in_viewport(
        bound: list[float], config: BrowserConfig
    ) -> bool:
        [x, y, width, height] = bound
        elem_left_bound = x
        elem_top_bound = y
        elem_right_bound = x + width
        elem_lower_bound = y + height

        not_in_viewport = (
            elem_left_bound < config["win_right_bound"]
            and elem_right_bound >= config["win_left_bound"]
            and elem_top_bound < config["win_lower_bound"]
            and elem_lower_bound >= config["win_upper_bound"]
        )
        return not_in_viewport

    @beartype
    def retrieve_viewport_info(self, info: BrowserInfo) -> None:
        """Add viewport related information to the DOMTree
        1. add union bound, which is a union of all the bounds of the nodes in the subtree
        This is only used when current_viewport_only is enabled since it is quite slow
        """
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]

        graph = defaultdict(lambda: [])
        assert len(node_names) == len(parent)
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        union_bounds: list[list[float] | None] = [None for _ in bounds]

        def valid_bbox(bound: list[float] | None) -> bool:
            if bound is None:
                return False
            # no width or height
            if np.isclose(bound[2], 0):
                return False
            if np.isclose(bound[3], 0):
                return False
            return True

        def add_union_bound(idx: int) -> list[float] | None:
            if idx in layout_node_cursor:
                cursor = layout_node_cursor.index(idx)
                node_bound = bounds[cursor].copy()
                tree_bounds: list[Any] = [node_bound]
                for child_idx in graph[idx]:
                    child_bound = add_union_bound(child_idx)
                    tree_bounds.append(
                        child_bound.copy() if child_bound else None
                    )

                tree_bounds = [b for b in tree_bounds if valid_bbox(b)]
                # convert to absolute coordinates
                for i in range(len(tree_bounds)):
                    tree_bounds[i][2] = tree_bounds[i][0] + tree_bounds[i][2]
                    tree_bounds[i][3] = tree_bounds[i][1] + tree_bounds[i][3]

                if len(tree_bounds) == 0:
                    assert not valid_bbox(node_bound)
                    node_union_bound = [0.0, 0.0, 0.0, 0.0]
                else:
                    left_bound = min([b[0] for b in tree_bounds])
                    top_bound = min([b[1] for b in tree_bounds])
                    right_bound = max([b[2] for b in tree_bounds])
                    bottom_bound = max([b[3] for b in tree_bounds])
                    node_union_bound = [
                        left_bound,
                        top_bound,
                        right_bound - left_bound,
                        bottom_bound - top_bound,
                    ]

                # update the list
                union_bounds[cursor] = node_union_bound
            else:
                node_union_bound = None

            return node_union_bound

        add_union_bound(0)
        info["DOMTree"]["documents"][0]["layout"]["unionBounds"] = union_bounds

    @beartype
    def current_viewport_html(self, info: BrowserInfo) -> str:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        attributes = nodes["attributes"]
        node_value = nodes["nodeValue"]
        parent = nodes["parentIndex"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        union_bounds = layout["unionBounds"]

        graph = defaultdict(lambda: [])
        for node_idx in range(len(node_names)):
            parent_idx = parent[node_idx]
            if parent_idx != -1:
                graph[parent_idx].append(node_idx)

        def dfs(idx: int) -> str:
            node_name = strings[node_names[idx]].lower().strip()
            can_skip = "#" in node_name or "::" in node_name

            inner_text = ""
            node_value_idx = node_value[idx]
            if node_value_idx >= 0 and node_value_idx < len(strings):
                inner_text = " ".join(strings[node_value_idx].split())
            node_attributes = [strings[i] for i in attributes[idx]]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            html = ""
            if not can_skip:
                html += f"<{node_name}"
                if {node_attributes_str}:
                    html += f" {node_attributes_str}"
                html += f">{inner_text}"
            else:
                html += f"{inner_text}"

            for child_idx in graph[idx]:
                if child_idx in layout_node_cursor:
                    cursor = layout_node_cursor.index(child_idx)
                    union_bound = union_bounds[cursor]
                    if not self.partially_in_viewport(
                        union_bound, info["config"]
                    ):
                        continue
                    html += dfs(child_idx)

            if not can_skip:
                html += f"</{node_name}>"

            return html

        html = dfs(0)
        return html

    @beartype
    def fetch_page_accessibility_tree(
        self, info: BrowserInfo, client: CDPSession
    ) -> AccessibilityTree:
        accessibility_tree: AccessibilityTree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        # add the bounding box of each node
        tree = info["DOMTree"]
        document = tree["documents"][0]
        nodes = document["nodes"]
        backend_node_id = nodes["backendNodeId"]
        node_names = nodes["nodeName"]

        layout = document["layout"]
        layout_node_cursor = layout["nodeIndex"]
        bounds = layout["bounds"]
        union_bounds = layout["unionBounds"]
        offsetrect_bounds = layout["offsetRects"]
        backend_id_to_bound = {}

        # get the mapping between backend node id and bounding box
        for idx in range(len(node_names)):
            if idx not in layout_node_cursor:
                continue
            cursor = layout_node_cursor.index(idx)
            node_bound = bounds[cursor]
            node_union_bound = union_bounds[cursor]
            node_offsetrect_bound = offsetrect_bounds[cursor]
            node_backend_id = backend_node_id[idx]
            backend_id_to_bound[node_backend_id] = [
                node_bound,
                node_union_bound,
                node_offsetrect_bound,
            ]

        parent_graph: dict[str, str] = {}
        refine_node_ids: list[str] = []
        for node in accessibility_tree:
            if "parentId" in node:
                parent_graph[node["nodeId"]] = node["parentId"]
            if "backendDOMNodeId" not in node:
                node["bound"] = None
                node["union_bound"] = None
                node["offsetrect_bound"] = None
            elif node["backendDOMNodeId"] not in backend_id_to_bound:
                refine_node_ids.append(node["nodeId"])
            else:
                node["bound"] = backend_id_to_bound[node["backendDOMNodeId"]][
                    0
                ]
                node["union_bound"] = backend_id_to_bound[
                    node["backendDOMNodeId"]
                ][1]
                node["offsetrect_bound"] = backend_id_to_bound[
                    node["backendDOMNodeId"]
                ][2]

        # refine the bounding box for nodes which only appear in the accessibility tree
        node_ids = [node["nodeId"] for node in accessibility_tree]
        for refine_node_id in refine_node_ids:
            child_id = refine_node_id
            parent_idx: None | int = None
            while child_id in parent_graph:
                parent_id = parent_graph[child_id]
                parent_idx = node_ids.index(parent_id)
                child_id = parent_id
                if accessibility_tree[parent_idx]["union_bound"] is not None:
                    break

            refine_node_idx = node_ids.index(refine_node_id)

            if parent_idx is not None:
                accessibility_tree[refine_node_idx][
                    "bound"
                ] = accessibility_tree[parent_idx]["bound"]
                accessibility_tree[refine_node_idx][
                    "union_bound"
                ] = accessibility_tree[parent_idx]["union_bound"]
                accessibility_tree[refine_node_idx][
                    "offsetrect_bound"
                ] = accessibility_tree[parent_idx]["offsetrect_bound"]
            else:
                accessibility_tree[refine_node_idx]["bound"] = None
                accessibility_tree[refine_node_idx]["union_bound"] = None
                accessibility_tree[refine_node_idx]["offsetrect_bound"] = None

        return accessibility_tree

    @beartype
    def current_viewport_accessibility_tree(
        self,
        info: BrowserInfo,
        accessibility_tree: AccessibilityTree,
    ) -> AccessibilityTree:
        config = info["config"]
        subtree = []
        for node in accessibility_tree:
            if not node["union_bound"]:
                continue

            [x, y, width, height] = node["union_bound"]
            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            ok = (
                elem_left_bound < config["win_right_bound"]
                and elem_right_bound >= config["win_left_bound"]
                and elem_top_bound < config["win_lower_bound"]
                and elem_lower_bound >= config["win_upper_bound"]
            )

            if ok:
                subtree.append(node)

        return subtree

    @beartype
    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree,
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "bound": node["bound"],
                        "union_bound": node["union_bound"],
                        "offsetrect_bound": node["offsetrect_bound"],
                        "text": node_str,
                    }

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    @beartype
    @staticmethod
    def clean_accesibility_tree(tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText '([^']+)'"

                match = re.search(pattern, line)
                if match:
                    static_text = match.group(1)
                    if all(
                        static_text not in prev_line
                        for prev_line in prev_lines
                    ):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)

    @beartype
    def process(self, page: Page, client: CDPSession) -> str:
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[
                        idx
                    ] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join(
                ["Tab {idx}" for idx in range(len(open_tabs))]
            )

        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page, client)

        if self.current_viewport_only:
            self.retrieve_viewport_info(browser_info)

        if self.observation_type == "html":
            if self.current_viewport_only:
                html = self.current_viewport_html(browser_info)
                content = html
            else:
                content = page.content()
        elif self.observation_type == "":
            content = ""
        elif self.observation_type == "accessibility_tree":
            accessibility_tree = self.fetch_page_accessibility_tree(
                browser_info, client
            )
            if self.current_viewport_only:
                accessibility_tree = self.current_viewport_accessibility_tree(
                    browser_info, accessibility_tree
                )
            content, obs_nodes_info = self.parse_accessibility_tree(
                accessibility_tree
            )
            content = self.clean_accesibility_tree(content)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info
        elif self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            # Check if the current page is an image url
            if page.url.endswith((".jpg", ".jpeg", ".png")):
                print("NOTE: We are on an image page!!!")
                # Load image from current url and run captioning on it.
                if page.url not in self.url2caption and self.captioning_fn is not None:
                    try:
                        image = Image.open(
                            requests.get(page.url, stream=True).raw
                        )
                        caption = self.captioning_fn([image])[0].strip()
                        self.url2caption[page.url] = remove_unicode(caption)
                    except Exception as e:
                        print("L579 WARNING: ", e)

                content = self.url2caption.get(page.url, "Image")
            else:
                if self.captioning_fn is not None:
                    images = page.query_selector_all("img")
                    image_urls = []
                    for image in images:
                        try:
                            image_url = image.get_attribute("src")
                            if not image_url.startswith(
                                ("http://", "https://", "www.")
                            ):
                                image_url = urljoin(page.url, image_url)
                            if image_url not in self.url2caption:
                                image_urls.append(image_url)
                        except Exception as e:
                            print("L604 WARNING: ", e)

                    # Run image captioning on image_url pixels. This is for models which use captioning as a baseline.
                    if len(image_urls) > 0:
                        image_pixels = []
                        valid_urls = []
                        for url in image_urls:
                            if "data:image/svg" in url:
                                continue
                            else:
                                try:
                                    image = Image.open(
                                        requests.get(url, stream=True).raw
                                    )
                                    image_pixels.append(image)
                                    valid_urls.append(url)
                                except Exception as e:
                                    print("L616 WARNING: ", e)

                        # Caption images.
                        if image_pixels:
                            # Run in batches of 4.
                            bs = 4
                            captions = []
                            for i in range(0, len(image_pixels), bs):
                                try:
                                    captions.extend(
                                        self.captioning_fn(
                                            image_pixels[i : i + bs]
                                        )
                                    )
                                except Exception as e:
                                    print("L628 WARNING: ", e)
                                    captions.extend(
                                        [""] * len(image_pixels[i : i + bs])
                                    )
                            assert len(valid_urls) == len(
                                captions
                            ), f"len(images)={len(valid_urls)}, len(captions)={len(captions)}"
                            for image_url, caption in zip(valid_urls, captions):
                                self.url2caption[image_url] = remove_unicode(
                                    caption.strip()
                                )

                    image_idx = 0
                    for image in images:
                        try:
                            original_alt = image.get_attribute("alt") or ""
                            image_url = image.get_attribute("src")
                            if not image_url.startswith(
                                ("http://", "https://", "www.")
                            ):
                                image_url = urljoin(page.url, image_url)

                            updated_alt = original_alt

                            if image_url in self.url2caption:
                                if self.url2caption[image_url] not in updated_alt:
                                    updated_alt = f"{updated_alt}, description: {self.url2caption[image_url]}"
                            elif "data:image/svg" not in image_url:
                                print(
                                    f"WARNING: {image_url} not in self.url2caption"
                                )

                            if "url:" not in updated_alt:
                                updated_alt = f"{updated_alt}, url: {image_url}"

                            safe_updated_alt = json.dumps(updated_alt)
                            image.evaluate(
                                f"node => node.alt = {safe_updated_alt}"
                            )
                        except Exception as e:
                            print("L653 WARNING:", e)

                if (
                    self.observation_type
                    == "accessibility_tree_with_captioner"
                ):
                    accessibility_tree = self.fetch_page_accessibility_tree(
                        browser_info, client
                    )
                    if self.current_viewport_only:
                        accessibility_tree = (
                            self.current_viewport_accessibility_tree(
                                browser_info, accessibility_tree
                            )
                        )
                    content, obs_nodes_info = self.parse_accessibility_tree(
                        accessibility_tree
                    )
                    content = self.clean_accesibility_tree(content)
                    self.obs_nodes_info = obs_nodes_info
                    self.meta_data["obs_nodes_info"] = obs_nodes_info
                else:
                    content = ""  # Not used for SoM
        else:
            raise ValueError(
                f"Invalid observation type: {self.observation_type}"
            )

        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"
        return content

    @beartype
    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["bound"]
        x, y, width, height = node_bound
        browser_config = self.browser_config
        b_x, b_y = (
            browser_config["win_left_bound"],
            browser_config["win_upper_bound"],
        )
        center_x = (x - b_x) + width / 2
        center_y = (y - b_y) + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )

class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]

class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        args,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ) -> None:
        self.main_observation_type = main_observation_type
        self.text_processor = TextObervationProcessor(
            text_observation_type,
            current_viewport_only,
            viewport_size,
            captioning_fn,
        )
        self.image_processor = ImageObservationProcessor(
            args, image_observation_type, viewport_size
        )
        self.viewport_size = viewport_size

    @beartype
    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({"text": text_space, "image": image_space})

    @beartype
    def get_observation(
        self, page: Page, client: CDPSession
    ) -> dict[str, Observation]:
        text_obs = self.text_processor.process(page, client)
        image_obs, content_str = self.image_processor.process(page, client)
        if content_str != "":
            text_obs = content_str
        return {"text": text_obs, "image": image_obs}

    @beartype
    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        return {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")