from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    expect,
    sync_playwright,
)
import os
import time
import json
from beartype import beartype
from playwright.async_api import async_playwright
from .actions_sync import Action, execute_action, get_action_space
from .build_tree_sync import HTMLTree
from .utils import (
    DetachedPage,
    Observation,
)
from typing import Any
import threading
import traceback
# from .processors import ObservationHandler, ObservationMetadata
from .processors import ObservationHandler, ObservationMetadata, get_interactive_elements_with_playwright, find_closest_center_coordinate
from .actions_sync import ActionTypes
from PIL import Image, ImageDraw
import logging

logger = logging.getLogger(__name__)

class Tls(threading.local):
    def __init__(self) -> None:
        self.playwright = sync_playwright().start()
        # self.playwright = await async_playwright()
        # print("Create playwright instance in Thread", threading.current_thread().name)
    def close(self):
        self.playwright.stop()

class ScriptBrowserEnv:
    """
    The goal of this environment is to produce a prototype of a browser environment.
    In the end, we want to support a fully configurable browser environment with wide
    range of action spaces and observation spaces, both structured and unstructured.
    But in this prototype, we just support action space specified by Playwright script,
    and observation space is the html content of the page.
    """
    def __init__(self, args, browser_type: str, viewport_size: ViewportSize):
        self.args = args
        self.browser_type = browser_type
        self.viewport_size = viewport_size
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        self.tls = Tls()
            
        self.reset_finished = False
        self.current_viewport_only = False

        self.image_observation_type = "image_som"
        self.text_observation_type = "image_som"  # type: ignore[assignment]
        self.main_observation_type = "image"
        
        self.observation_handler = ObservationHandler(
            args,
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
            captioning_fn=None,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )

        self.tree = HTMLTree()

    @beartype
    def setup(self, url) -> None:

        print('inside setup')

        self.browser = self.tls.playwright.chromium.launch(headless=False, slow_mo=0)

        # Use custom viewport size if specified in the config, otherwise use the default.
        viewport_size = self.viewport_size.copy()

        self.context = self.browser.new_context(
            viewport=viewport_size,
            device_scale_factor=1,
        )
        
        # if not url.startswith("http"):
            # self.page = None
            # return
        page = self.context.new_page()
        client = page.context.new_cdp_session(
            page
        )  # talk to chrome devtools
        page.client = client  # type: ignore

        num_retries = 0

        for i in range(3):
            try:
                page.goto(url)
                break
            except Exception as e:
                logging.error(traceback.format_exc())
                num_retries += 1
                time.sleep(5)
        
        if num_retries == 3:
            raise Exception("Failed to load the page after 3 retries")

        # set the first page as the current page
        self.page = self.context.pages[0]
        self.page.bring_to_front()

        self.html_content = self.page.content()

        # print(self.html_content)
    
    async def async_setup(self, url) -> None:
        async with async_playwright() as p:
            self.browser = await p.chromium.launch(headless=False, slow_mo=0)
            viewport_size = self.viewport_size.copy()
            self.context = await self.browser.new_context(viewport=viewport_size, device_scale_factor=1)
            if not url.startswith("http"):
                self.page = None
                return
            self.page = await self.context.new_page()
            client = await self.page.context.new_cdp_session(self.page)  # talk to chrome devtools
            self.page.client = client  # type: ignore
            await self.page.goto(url)

            self.html_content = await self.page.content()
    
    def get_obs(self, image_path, som_model, caption_model_processor):
        som_image_obs, parsed_html_str, visible_rects = self.observation_handler.action_processor.process_new(self.page, self.page.client, intent=None, image_path=image_path, som_model=som_model, caption_model_processor=caption_model_processor)
        
        try:
            self.html_content = self.page.content()
            self.tree.fetch_html_content(self.html_content)
            logger.info("-- Successfully fetch html content")
            tab_name = self.page.title()
            parsed_html_str = f"current web tab name is \'{tab_name}\'\n" + parsed_html_str

        except Exception as e:
            logger.error(f"-- Failed to fetch html content,error occur {e}")
        
        n_visible_covered = 0

        for ele_id in visible_rects:
            if ele_id in self.tree.uniqueId2nodeId:
                n_visible_covered += 1

        logger.info("n_visible_covered = {}/{}".format(n_visible_covered, len(visible_rects)))

        return {'page': self.page, 'client': self.page.client, 'content_str': parsed_html_str, 'image_obs': som_image_obs, 'tree': self.tree}

    def close(self):
        if self.page is not None:
            self.page.close()
        if self.context is not None:
            self.context.close()
        if self.browser is not None:
            self.browser.close()

            self.tls.close()
    
    async def async_close(self):
        await self.page.close()
        # await self.context.close()
        await self.browser.close()

    def step(
        self, action
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:

        success = False
        fail_error = ""
        
        # som_image_obs, parsed_html_str = self.observation_handler.action_processor.process(self.page, self.page.client, None)
        # som_image_obs, parsed_html_str = self.observation_handler.action_processor.process_new(self.page, self.page.client, None)

        logging.info('action = {}'.format(action))
        logging.info('action = {}'.format(action["action_type"]))

        if action["action_type"] == ActionTypes.SELECT:
            # do the necessary processing to get id2selector
            interactive_rects = get_interactive_elements_with_playwright(self.page)

            # logging.info('interactive_rects = {}'.format(interactive_rects))
            # logging.info('rects = {}'.format(self.observation_handler.action_processor.rects))

            self.observation_handler.action_processor.id2selector = {}

            for box_id in self.observation_handler.action_processor.rects:
                box = self.observation_handler.action_processor.rects[box_id]

                box_coord = (box["rects"][0]["x"], box["rects"][0]["y"], box["rects"][0]["width"], box["rects"][0]["height"])
                idx = find_closest_center_coordinate(box_coord, interactive_rects)

                if idx is not None:
                    self.observation_handler.action_processor.id2selector[box_id] = interactive_rects[idx][4]

            logging.info('id2selector = {}'.format(self.observation_handler.action_processor.id2selector))

        try:
            self.page = execute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
                # self.tree,
            )

            success = True
        except Exception as e:
            fail_error = str(e)
            # traceback.print_exc()
            logging.info('Error in executing action: {}'.format(fail_error))
            logging.error(traceback.format_exc())

        logging.info("Action executed successfully: {}".format(success))
        
        return

    def get_selector(self, element) -> str:
        
        selector = self.page.evaluate('''(element) => {
                function elemToSelector(elem) {
                    const { tagName, id, className, parentElement } = elem;

                    if (id && /^[a-zA-Z].*/.test(id)) {
                        return `#${id}`;
                    }

                    let selector = tagName.toLowerCase();
                    if (className) {
                        selector += '.' + className.trim().replace(/ +/g, '.');
                    }

                    const needNthPart = (el) => {
                        if (!el.className) return true;
                        let sib = el.previousElementSibling;
                        while (sib) {
                            if (el.className !== sib.className) return false;
                            sib = sib.previousElementSibling;
                        }
                        return true;
                    };

                    const getNthPart = (el) => {
                        let childIndex = 1;
                        let sib = el.previousElementSibling;
                        while (sib) {
                            childIndex++;
                            sib = sib.previousElementSibling;
                        }
                        return `:nth-child(${childIndex})`;
                    };

                    if (needNthPart(elem)) {
                        selector += getNthPart(elem);
                    }

                    if (!parentElement) {
                        return selector;
                    }

                    return `${elemToSelector(parentElement)} > ${selector}`;
                }

                return elemToSelector(element);
            }''', element)

        return selector
    
    def get_element_value(self,element) -> str:
        text = element.text_content()
        if text and text.strip():
            return text.strip()

        title = element.get_attribute('title')
        if title:
            return title

        placeholder = element.get_attribute('placeholder')
        if placeholder:
            return placeholder

        aria_label = element.get_attribute('aria-label')
        if aria_label:
            return aria_label

        aria_checked = element.get_attribute('aria-checked')
        if aria_checked:
            return aria_checked

        element_type = element.get_attribute('type')
        if element_type:
            return element_type

        tag_name = element.evaluate('(element) => element.tagName.toLowerCase()')
        if tag_name == "select":
            return "Select an option value"

        return ""
    
    def get_element_by_coordinate(self, x=0, y=0):
        print("x = {}, y = {}".format(x, y))

        self.page.evaluate(f'window.scrollTo({x}, {y})')

        # element = self.page.evaluate_handle(f'document.elementFromPoint({x}, {y})')
        element = self.page.evaluate_handle('''([x, y]) => {
        const rect = document.documentElement.getBoundingClientRect();
        const adjustedX = x - rect.left;
        const adjustedY = y - rect.top;
        return document.elementFromPoint(adjustedX, adjustedY);
    }''', [x, y])

        print("element = ", element)

        try:
            selector = self.get_selector(element)
            value = self.get_element_value(element)
        except Exception as e:
            print(e)
            logger.info(traceback.format_exc())
            selector=''
            value=''

        print("ELEMENT VALUE IS DONE: ", value)
        return {'selector': selector, 'value': value}

@beartype
class ActionParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)