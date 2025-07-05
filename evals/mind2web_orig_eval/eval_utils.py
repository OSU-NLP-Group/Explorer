import os
import logging
from PIL import Image
import ast
import json
import re
from bs4 import BeautifulSoup
import logging




# calculate action f1 following mind2web
def calculate_f1(pred, label):
    
    logging.info('pred = {}'.format(pred))
    logging.info('label = {}'.format(label))

    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1
