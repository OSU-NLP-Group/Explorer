from beartype import beartype
from typing import TypedDict, Optional
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class Action(TypedDict):
    action_type: int
    element_id: int
    element_name: str
    url: str
    fill_text: str


class ActionTypes(IntEnum):
    NONE = 0
    CLICK = 1
    GOTO = 2
    GOOGLE_SEARCH = 3
    FILL_FORM = 4
    SWITCH_TAB = 5
    GO_BACK = 6
    FILL_SEARCH = 7
    SELECT_OPTION = 8
    HOVER = 9
    SCROLL_DOWN = 10
    SCROLL_UP = 11


@beartype
def create_click_action(elementid: int) -> Action:
    return {
        "action_type": ActionTypes.CLICK,
        "element_id": elementid,
        "url": "",
        "fill_text": "",
        "element_name": "",
    }


@beartype
def create_goto_action(elementid: int, url: str) -> Action:
    return {
        "action_type": ActionTypes.GOTO,
        "element_id": elementid,
        "url": url,
        "fill_text": "",
        "element_name": "",
    }


@beartype
def create_none_action(elementid: int) -> Action:
    return {
        "action_type": ActionTypes.NONE,
        "element_id": elementid,
        "url": "",
        "fill_text": "",
        "element_name": "",
    }


@beartype
def create_fill_action(elementid: int, fill_text: str) -> Action:
    return {
        "action_type": ActionTypes.FILL_FORM,
        "element_id": elementid,
        "url": "",
        "fill_text": fill_text,
        "element_name": "",
    }


@beartype
def create_fill_search_action(elementid: int, fill_text: str) -> Action:
    return {
        "action_type": ActionTypes.FILL_SEARCH,
        "element_id": elementid,
        "url": "",
        "fill_text": fill_text,
        "element_name": "",
    }


@beartype
def create_search_action(elementid: int, text: str) -> Action:
    return {
        "action_type": ActionTypes.GOOGLE_SEARCH,
        "element_id": elementid,
        "url": "https://www.google.com",
        "fill_text": text,
        "element_name": "",
    }


@beartype
def create_go_back_action(elementid: int) -> Action:
    return {
        "action_type": ActionTypes.GO_BACK,
        "element_id": elementid,
        "url": "",
        "fill_text": "",
        "element_name": "",
    }


@beartype
def create_select_option_action(elementid: int, target_value: str) -> Action:
    return {
        "action_type": ActionTypes.SELECT_OPTION,
        "element_id": elementid,
        "url": "",
        "fill_text": target_value,
        "element_name": "",
    }


@beartype
def create_hover_action(elementid: int) -> Action:
    return {
        "action_type": ActionTypes.HOVER,
        "element_id": elementid,
        "url": "",
        "fill_text": "",
        "element_name": "",
    }


@beartype
def create_scroll_down_action(elementid: int) -> Action:
    return {
        "action_type": ActionTypes.SCROLL_DOWN,
        "element_id": elementid,
        "url": "",
        "fill_text": "",
        "element_name": "",
    }


@beartype
def create_scroll_up_action(elementid: int) -> Action:
    return {
        "action_type": ActionTypes.SCROLL_UP,
        "element_id": elementid,
        "url": "",
        "fill_text": "",
        "element_name": "",
    }


@beartype
def create_action(
    elementid: int, action_type: str, action_input: Optional[str]
) -> Action:
    logger.info("inside create_action")
    logger.info(f"action_type: {action_type}, action_input: {action_input}")

    if action_type == "click":
        return create_click_action(elementid=elementid)
    elif action_type == "fill_form":
        return create_fill_action(elementid=elementid, fill_text=action_input)
    elif action_type == "fill_search":
        return create_fill_search_action(elementid=elementid, fill_text=action_input)
    elif action_type == "goto":
        return create_goto_action(elementid=elementid, url=action_input)
    elif action_type == "google_search":
        return create_search_action(elementid=elementid, text=action_input)
    elif action_type == "go_back":
        return create_go_back_action(elementid=elementid)
    elif action_type == "select_option":
        return create_select_option_action(
            elementid=elementid, target_value=action_input
        )
    elif action_type == "hover":
        return create_hover_action(elementid=elementid)
    elif action_type == "scroll_down":
        return create_scroll_down_action(elementid=elementid)
    elif action_type == "scroll_up":
        return create_scroll_up_action(elementid=elementid)
    else:
        return create_none_action(elementid=elementid)


__all__ = [
    "Action",
    "ActionTypes",
    "create_click_action",
    "create_fill_action",
    "create_none_action",
    "create_goto_action",
    "create_search_action",
    "create_go_back_action",
    "create_fill_search_action",
    "create_select_option_action",
    "create_hover_action",
    "create_scroll_down_action",
    "create_scroll_up_action",
    "create_action",
]
