import random
import io
from PIL import Image, ImageDraw, ImageFont
import json
import logging
from tqdm import tqdm

TOP_NO_LABEL_ZONE = 20  # Don't print any labels close the top of the page


def add_set_of_mark(screenshot, example, scores_all_data, use_top50=True, omit_top50_pos=True, keep_positive_overlaps=False):

    pos_boxes = {json.loads(json.loads(candidate)['attributes'])['backend_node_id']:json.loads(json.loads(candidate)['attributes'])['bounding_box_rect'].split(',') for candidate in example['pos_candidates']}
    neg_boxes = {json.loads(json.loads(candidate)['attributes'])['backend_node_id']:json.loads(json.loads(candidate)['attributes'])['bounding_box_rect'].split(',') for candidate in example['neg_candidates']}
    
    annotation_id = example['annotation_id']
    action_uid = example['action_uid']
    overall_id = f'{annotation_id}_{action_uid}'

    # logging.info('neg_boxes before = {}'.format(len(neg_boxes)))

    if use_top50:
        if not omit_top50_pos:
            pos_boxes = {k: pos_boxes[k] for k in list(scores_all_data['ranks'][overall_id].keys())[:50] if k in pos_boxes}
        neg_boxes = {k: neg_boxes[k] for k in list(scores_all_data['ranks'][overall_id].keys())[:50] if k in neg_boxes}

    # logging.info('neg_boxes after = {}'.format(len(neg_boxes)))

    logging.info('len(pos_boxes) = {}'.format(len(pos_boxes)))
    logging.info('len(neg_boxes) = {}'.format(len(neg_boxes)))

    if keep_positive_overlaps:
        pos_boxes_dedup, neg_boxes_dedup = remove_overlap_pos_neg(pos_boxes, neg_boxes)
        logging.info('len(pos_boxes_dedup) = {}'.format(len(pos_boxes_dedup)))
        logging.info('len(neg_boxes_dedup) = {}'.format(len(neg_boxes_dedup)))
    else:
        neg_boxes_dedup = remove_overlap(neg_boxes)
        logging.info('len(neg_boxes_dedup) = {}'.format(len(neg_boxes_dedup)))

        logging.info('pos_boxes = {}'.format(pos_boxes))

        try:
            keep_idx = list(pos_boxes.keys())[0]
            max_area = -1
            for box_id in pos_boxes:
                if float(pos_boxes[box_id][2]) * float(pos_boxes[box_id][3]) > max_area:
                    max_area = float(pos_boxes[box_id][2]) * float(pos_boxes[box_id][3])
                    keep_idx = box_id
            
            pos_boxes_dedup = {keep_idx: pos_boxes[keep_idx]}
        except:
            pos_boxes_dedup = {}

    # print('pos_boxes_dedup = {}'.format(pos_boxes_dedup))

    all_boxes = {**pos_boxes_dedup, **neg_boxes_dedup}

    all_boxes = {k: [float(v) for v in box] for k, box in all_boxes.items()}
    # logging.info('all_boxes = {}'.format(all_boxes))
    
    # Create a new dictionary with shuffled keys
    all_box_ids = list(all_boxes.keys())
    random.shuffle(all_box_ids)
    
    all_boxes = {key: all_boxes[key] for key in all_box_ids}

    # assume bbox format of left, top, width, height
    all_boxes = {k: {'rects': [{'left': box[0], 'top': box[1], 'width': box[2], 'height': box[3], 'right': box[0] + box[2], 'bottom': box[1] + box[3]}]} for k, box in all_boxes.items()}

    # map the ids of all_boxes
    backend_node_id_to_idx = {k: str(i) for i, k in enumerate(all_boxes.keys())}

    all_boxes = {backend_node_id_to_idx[k]:v for k, v in all_boxes.items()}
    pos_boxes_dedup = {backend_node_id_to_idx[k]:v for k, v in pos_boxes_dedup.items()}

    # all_boxes = [{'top': box[0], 'left': box[1], 'right': box[1] + box[2], 'bottom': box[0] + box[3]} for box in all_boxes]

    if isinstance(screenshot, Image.Image):
        return _add_set_of_mark(screenshot, all_boxes, pos_boxes_dedup, backend_node_id_to_idx)

    if not isinstance(screenshot, io.BufferedIOBase):
        screenshot = io.BytesIO(screenshot)

    image = Image.open(screenshot)
    result = _add_set_of_mark(image, all_boxes, pos_boxes_dedup, backend_node_id_to_idx)
    image.close()
    return result


def _add_set_of_mark(screenshot, ROIs, pos_boxes_dedup, backend_node_id_to_idx):
    visible_rects = list()
    rects_above = list() # Scroll up to see
    rects_below = list() # Scroll down to see

    fnt = ImageFont.load_default() # 14
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

            mid = ((rect["right"] + rect["left"]) / 2.0, (rect["top"] + rect["bottom"]) / 2.0)

            if 0 <= mid[0] and mid[0] < base.size[0]:
                if mid[1] < 0:
                    rects_above.append(r)
                elif mid[1] >= base.size[1]:
                    rects_below.append(r)
                else:
                    visible_rects.append(r)
                    _draw_roi(draw, int(r), fnt, rect)

    comp = Image.alpha_composite(base, overlay)
    overlay.close()
    return comp, visible_rects, rects_above, rects_below, pos_boxes_dedup, ROIs, backend_node_id_to_idx



def _draw_roi(draw, idx, font, rect):
    color = _color(idx)
    luminance = color[0] * 0.3 + color[1] * 0.59 + color[2] * 0.11
    text_color = (0, 0, 0, 255) if luminance > 90 else (255, 255, 255, 255)

    roi = [(rect["left"], rect["top"]), (rect["right"], rect["bottom"])]

    label_location = (rect["right"], rect["top"])
    label_anchor = "rb"

    if label_location[1] <= TOP_NO_LABEL_ZONE:
        label_location = (rect["right"], rect["bottom"])
        label_anchor = "rt"

    draw.rectangle(roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2)

    bbox = draw.textbbox(label_location, str(idx), font=font, anchor=label_anchor, align="center")
    bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
    draw.rectangle(bbox, fill=color)

    draw.text(label_location, str(idx), fill=text_color, font=font, anchor=label_anchor, align="center")


def _color(identifier):
    rnd = random.Random(int(identifier))
    color = [rnd.randint(0, 255), rnd.randint(125, 255), rnd.randint(0, 50)]
    rnd.shuffle(color)
    color.append(255)
    return tuple(color)


def remove_overlap(boxes, iou_threshold=0.3):

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection

        if union==0:
            return 0
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    # boxes = boxes.tolist()
    filtered_boxes = {}

    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        box1_coord = [float(boxes[box1][0]), float(boxes[box1][1]), float(boxes[box1][0]) + float(boxes[box1][2]), float(boxes[box1][1]) + float(boxes[box1][3])]

        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            box2_coord = [float(boxes[box2][0]), float(boxes[box2][1]), float(boxes[box2][0]) + float(boxes[box2][2]), float(boxes[box2][1]) + float(boxes[box2][3])]
            if i != j and IoU(box1_coord, box2_coord) > iou_threshold and box_area(box1_coord) > box_area(box2_coord):
                is_valid_box = False
                break

        if is_valid_box:
            # print('box1 = {}'.format(box1))
            # print('boxes[box1] = {}'.format(boxes[box1]))
            filtered_boxes.update({box1: boxes[box1]})

    return filtered_boxes

def remove_overlap_pos_neg(pos_boxes, neg_boxes, iou_threshold=0.3):
    # remove overlapping boxes but give priority to positive boxes

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection

        if union==0:
            return 0
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    # boxes = boxes.tolist()
    filtered_pos_boxes = {}
    filtered_neg_boxes = {}

    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(pos_boxes):
        box1_coord = [float(pos_boxes[box1][0]), float(pos_boxes[box1][1]), float(pos_boxes[box1][0]) + float(pos_boxes[box1][2]), float(pos_boxes[box1][1]) + float(pos_boxes[box1][3])]

        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(pos_boxes):
            box2_coord = [float(pos_boxes[box2][0]), float(pos_boxes[box2][1]), float(pos_boxes[box2][0]) + float(pos_boxes[box2][2]), float(pos_boxes[box2][1]) + float(pos_boxes[box2][3])]
            if i != j and IoU(box1_coord, box2_coord) > iou_threshold and box_area(box2_coord) > box_area(box1_coord):
                is_valid_box = False
                break

        if is_valid_box:
            # print('box1 = {}'.format(box1))
            # print('boxes[box1] = {}'.format(boxes[box1]))
            filtered_pos_boxes.update({box1: pos_boxes[box1]})

    all_boxes = {**filtered_pos_boxes, **neg_boxes}

    # filter negative boxes that overlap with positive boxes
    for i, box1 in enumerate(neg_boxes):
        box1_coord = [float(neg_boxes[box1][0]), float(neg_boxes[box1][1]), float(neg_boxes[box1][0]) + float(neg_boxes[box1][2]), float(neg_boxes[box1][1]) + float(neg_boxes[box1][3])]

        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(filtered_pos_boxes):
            box2_coord = [float(filtered_pos_boxes[box2][0]), float(filtered_pos_boxes[box2][1]), float(filtered_pos_boxes[box2][0]) + float(filtered_pos_boxes[box2][2]), float(filtered_pos_boxes[box2][1]) + float(filtered_pos_boxes[box2][3])]
            if IoU(box1_coord, box2_coord) > iou_threshold:
                is_valid_box = False
                break

        if is_valid_box:
            # print('box1 = {}'.format(box1))
            # print('boxes[box1] = {}'.format(boxes[box1]))
            filtered_neg_boxes.update({box1: neg_boxes[box1]})

    filtered_neg_boxes_v2 = {}

    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(filtered_neg_boxes):
        box1_coord = [float(filtered_neg_boxes[box1][0]), float(filtered_neg_boxes[box1][1]), float(filtered_neg_boxes[box1][0]) + float(filtered_neg_boxes[box1][2]), float(filtered_neg_boxes[box1][1]) + float(filtered_neg_boxes[box1][3])]

        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(filtered_neg_boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(filtered_neg_boxes):
            box2_coord = [float(filtered_neg_boxes[box2][0]), float(filtered_neg_boxes[box2][1]), float(filtered_neg_boxes[box2][0]) + float(filtered_neg_boxes[box2][2]), float(filtered_neg_boxes[box2][1]) + float(filtered_neg_boxes[box2][3])]
            if i != j and IoU(box1_coord, box2_coord) > iou_threshold and box_area(box1_coord) > box_area(box2_coord):
                is_valid_box = False
                break

        if is_valid_box:
            # print('box1 = {}'.format(box1))
            # print('filtered_neg_boxes[box1] = {}'.format(boxes[box1]))
            filtered_neg_boxes_v2.update({box1: filtered_neg_boxes[box1]})

    return filtered_pos_boxes, filtered_neg_boxes_v2

'''
def add_set_of_mark_pos_neg(screenshot, example):

    pos_boxes = {json.loads(json.loads(candidate)['attributes'])['backend_node_id']:json.loads(json.loads(candidate)['attributes'])['bounding_box_rect'].split(',') for candidate in example['pos_candidates']}
    neg_boxes = {json.loads(json.loads(candidate)['attributes'])['backend_node_id']:json.loads(json.loads(candidate)['attributes'])['bounding_box_rect'].split(',') for candidate in example['neg_candidates']}
    
    logging.info('pos_boxes = {}'.format(pos_boxes))

    all_boxes = {k: [float(v) for v in box] for k, box in all_boxes.items()}
    
    # assume bbox format of left, top, width, height
    pos_boxes = {k: {'rects': [{'left': box[0], 'top': box[1], 'width': box[2], 'height': box[3], 'right': box[0] + box[2], 'bottom': box[1] + box[3]}]} for k, box in pos_boxes.items()}
    neg_boxes = {k: {'rects': [{'left': box[0], 'top': box[1], 'width': box[2], 'height': box[3], 'right': box[0] + box[2], 'bottom': box[1] + box[3]}]} for k, box in neg_boxes.items()}

    # all_boxes = [{'top': box[0], 'left': box[1], 'right': box[1] + box[2], 'bottom': box[0] + box[3]} for box in all_boxes]

    screenshot = _add_set_of_mark_pos_neg(screenshot, pos_boxes, color='green')
    screenshot = _add_set_of_mark_pos_neg(screenshot, neg_boxes, color='red')

    return screenshot
'''
