import random
import io
from PIL import Image, ImageDraw, ImageFont

TOP_NO_LABEL_ZONE = 20  # Don't print any labels close the top of the page


def add_set_of_mark(screenshot, ROIs, draw=True, no_fill=False):
    if isinstance(screenshot, Image.Image):
        return _add_set_of_mark(screenshot, ROIs, draw, no_fill)

    if not isinstance(screenshot, io.BufferedIOBase):
        screenshot = io.BytesIO(screenshot)

    image = Image.open(screenshot)
    result = _add_set_of_mark(image, ROIs, draw, no_fill)
    image.close()
    return result


def _add_set_of_mark(screenshot, ROIs, draw, no_fill):
    visible_rects = list()
    rects_above = list()  # Scroll up to see
    rects_below = list()  # Scroll down to see

    fnt = ImageFont.load_default()  # 14
    base = screenshot.convert("L").convert("RGBA")
    overlay = Image.new("RGBA", base.size)

    draw = ImageDraw.Draw(overlay)
    for r in ROIs:
        for rect in ROIs[r]["rects"]:
            # Empty rectangles
            if not rect:
                continue
            if rect["width"] * rect["height"] == 0:
                continue

            mid = (
                (rect["right"] + rect["left"]) / 2.0,
                (rect["top"] + rect["bottom"]) / 2.0,
            )

            if 0 <= mid[0] and mid[0] < base.size[0]:
                if mid[1] < 0:
                    rects_above.append(r)
                elif mid[1] >= base.size[1]:
                    rects_below.append(r)
                else:
                    visible_rects.append(r)
                    if draw:
                        _draw_roi(draw, int(r), fnt, rect, no_fill)

    comp = Image.alpha_composite(base, overlay)
    overlay.close()
    return comp, visible_rects, rects_above, rects_below


def _trim_drawn_text(draw, text, font, max_width):
    buff = ""
    for c in text:
        tmp = buff + c
        bbox = draw.textbbox((0, 0), tmp, font=font, anchor="lt", align="left")
        width = bbox[2] - bbox[0]
        if width > max_width:
            return buff
        buff = tmp
    return buff


def _draw_roi(draw, idx, font, rect, no_fill):
    color = _color(idx)
    luminance = color[0] * 0.3 + color[1] * 0.59 + color[2] * 0.11
    text_color = (0, 0, 0, 255) if luminance > 90 else (255, 255, 255, 255)

    roi = [(rect["left"], rect["top"]), (rect["right"], rect["bottom"])]

    label_location = (rect["right"], rect["top"])
    label_anchor = "rb"

    if label_location[1] <= TOP_NO_LABEL_ZONE:
        label_location = (rect["right"], rect["bottom"])
        label_anchor = "rt"

    # draw.rectangle(roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2)
    if no_fill:
        draw.rectangle(roi, outline=color, fill=None, width=2)
    else:
        draw.rectangle(
            roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2
        )

    bbox = draw.textbbox(
        label_location, str(idx), font=font, anchor=label_anchor, align="center"
    )
    bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
    draw.rectangle(bbox, fill=color)
    # draw.rectangle(bbox)

    draw.text(
        label_location,
        str(idx),
        fill=text_color,
        font=font,
        anchor=label_anchor,
        align="center",
    )


def _color(identifier):
    rnd = random.Random(int(identifier))
    color = [rnd.randint(0, 255), rnd.randint(125, 255), rnd.randint(0, 50)]
    rnd.shuffle(color)
    color.append(255)
    return tuple(color)
