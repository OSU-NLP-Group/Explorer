import re
from typing import Literal

ROLES = (
    "alert",
    "alertdialog",
    "application",
    "article",
    "banner",
    "blockquote",
    "button",
    "caption",
    "cell",
    "checkbox",
    "code",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "definition",
    "deletion",
    "dialog",
    "directory",
    "document",
    "emphasis",
    "feed",
    "figure",
    "form",
    "generic",
    "grid",
    "gridcell",
    "group",
    "heading",
    "img",
    "insertion",
    "link",
    "list",
    "listbox",
    "listitem",
    "log",
    "main",
    "marquee",
    "math",
    "meter",
    "menu",
    "menubar",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "navigation",
    "none",
    "note",
    "option",
    "paragraph",
    "presentation",
    "progressbar",
    "radio",
    "radiogroup",
    "region",
    "row",
    "rowgroup",
    "rowheader",
    "scrollbar",
    "search",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "status",
    "strong",
    "subscript",
    "superscript",
    "switch",
    "tab",
    "table",
    "tablist",
    "tabpanel",
    "term",
    "textbox",
    "time",
    "timer",
    "toolbar",
    "tooltip",
    "tree",
    "treegrid",
    "treeitem",
)

SPECIAL_LOCATORS = (
    "alt_text",
    "label",
    "placeholder",
)

ASCII_CHARSET = "".join(chr(x) for x in range(32, 128))
FREQ_UNICODE_CHARSET = "".join(chr(x) for x in range(129, 130000))
UTTERANCE_MAX_LENGTH = 8192
ATTRIBUTE_MAX_LENGTH = 256
TEXT_MAX_LENGTH = 256
TYPING_MAX_LENGTH = 64
URL_MAX_LENGTH = 256
MAX_ELEMENT_INDEX_IN_VIEWPORT = 10
MAX_ELEMENT_ID = 1000
MAX_ANSWER_LENGTH = 512

MIN_REF = -1000000
MAX_REF = 1000000

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 240
TASK_WIDTH = 160
TASK_HEIGHT = 210

FLIGHT_WINDOW_WIDTH = 600
FLIGHT_WINDOW_HEIGHT = 700
FLIGHT_TASK_WIDTH = 375
FLIGHT_TASK_HEIGHT = 667
MAX_PAGE_NUMBER = 10

SPECIAL_KEYS = (
    "Enter",
    "Tab",
    "Control",
    "Shift",
    "Meta",
    "Backspace",
    "Delete",
    "Escape",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "PageDown",
    "PageUp",
    "Meta+a",
)

SPECIAL_KEY_MAPPINGS = {
    "backquote": "Backquote",
    "minus": "Minus",
    "equal": "Equal",
    "backslash": "Backslash",
    "backspace": "Backspace",
    "meta": "Meta",
    "tab": "Tab",
    "delete": "Delete",
    "escape": "Escape",
    "arrowdown": "ArrowDown",
    "end": "End",
    "enter": "Enter",
    "home": "Home",
    "insert": "Insert",
    "pagedown": "PageDown",
    "pageup": "PageUp",
    "arrowright": "ArrowRight",
    "arrowup": "ArrowUp",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
}

RolesType = Literal[
    "alert",
    "alertdialog",
    "application",
    "article",
    "banner",
    "blockquote",
    "button",
    "caption",
    "cell",
    "checkbox",
    "code",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "definition",
    "deletion",
    "dialog",
    "directory",
    "document",
    "emphasis",
    "feed",
    "figure",
    "form",
    "generic",
    "grid",
    "gridcell",
    "group",
    "heading",
    "img",
    "insertion",
    "link",
    "list",
    "listbox",
    "listitem",
    "log",
    "main",
    "marquee",
    "math",
    "meter",
    "menu",
    "menubar",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "navigation",
    "none",
    "note",
    "option",
    "paragraph",
    "presentation",
    "progressbar",
    "radio",
    "radiogroup",
    "region",
    "row",
    "rowgroup",
    "rowheader",
    "scrollbar",
    "search",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "status",
    "strong",
    "subscript",
    "superscript",
    "switch",
    "tab",
    "table",
    "tablist",
    "tabpanel",
    "term",
    "textbox",
    "time",
    "timer",
    "toolbar",
    "tooltip",
    "tree",
    "treegrid",
    "treeitem",
    "alt_text",
    "label",
    "placeholder",
]

MAX_VANILLA_STR_LENGTH = 1000

PLAYWRIGHT_LOCATORS = (
    "get_by_role",
    "get_by_text",
    "get_by_label",
    "get_by_placeholder",
    "get_by_alt_text",
    "get_by_title",
    "get_by_test_id",
    "filter",
    "frame_locator",
    "locator",
)

PLAYWRIGHT_ACTIONS = (
    "fill",
    "check",
    "select_option",
    "click",
    "hover",
    "dclick",
    "type",
    "focus",
    "goto",
    "press",
    "scroll",
)

IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
)

INJECTED_ATTR_NAME = "aria-roledescription"
BID_ATTR = "bid"  # the attribute name for extra meta data
BID_EXPR = r"([-0-9]+)"
FLOAT_EXPR = r"([+-]?(?:[0-9]*[.])?[0-9]+)"
BOOL_EXPR = r"([01])"

DATA_REGEXP = re.compile(
    BID_EXPR
    + r"_"
    + FLOAT_EXPR
    + r"_"
    + FLOAT_EXPR
    + r"_"
    + FLOAT_EXPR
    + r"_"
    + FLOAT_EXPR
    + r"_"
    + FLOAT_EXPR
    + r"_"
    + FLOAT_EXPR
    + r"_"
    + BOOL_EXPR
    + r"_"
    + r"(.*)"
)

IN_VIEWPORT_RATIO_THRESHOLD = 0.6
