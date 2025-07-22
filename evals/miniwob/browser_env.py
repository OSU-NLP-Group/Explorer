from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    expect,
    sync_playwright,
)
import sys, os
import time
import json
from beartype import beartype
from playwright.async_api import async_playwright
from .actions import Action, execute_action, get_action_space
from .utils import (
    DetachedPage,
    Observation,
)
from typing import Any
import threading
import traceback

# from .processors import ObservationHandler, ObservationMetadata
from .processors import (
    ObservationHandler,
    ObservationMetadata,
    get_interactive_elements_with_playwright,
    find_closest_center_coordinate,
)
from .actions import ActionTypes
from PIL import Image, ImageDraw
import logging
import playwright


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

    def __init__(
        self, args, browser_type: str, viewport_size: ViewportSize, image_processor=None
    ):
        self.args = args
        self.browser_type = browser_type
        self.viewport_size = viewport_size
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        if not self.args.use_async_playwright:
            self.tls = Tls()
        else:
            self.tls = None

        self.reset_finished = False
        self.current_viewport_only = False

        self.image_processor = image_processor

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

        self.observation_space = self.observation_handler.get_observation_space()

    @beartype
    def setup(self, url, log_dir) -> None:
        print("inside setup")

        # self.context_manager = sync_playwright()
        # self.playwright = self.context_manager.__enter__()
        # self.browser = self.playwright.chromium.launch(
        #     headless=True, slow_mo=0
        # )
        if self.args.record_video:
            print("recording video...")
            self.browser = self.tls.playwright.chromium.launch(
                headless=False, slow_mo=0
            )
        else:
            self.browser = self.tls.playwright.chromium.launch(headless=True, slow_mo=0)

        # Use custom viewport size if specified in the config, otherwise use the default.
        viewport_size = self.viewport_size.copy()

        if self.args.record_video:
            self.context = self.browser.new_context(
                viewport=viewport_size, device_scale_factor=1, record_video_dir=log_dir
            )
        else:
            self.context = self.browser.new_context(
                viewport=viewport_size,
                device_scale_factor=1,
            )

        if not url.startswith("http"):
            self.page = None
            return
        page = self.context.new_page()
        client = page.context.new_cdp_session(page)  # talk to chrome devtools
        page.client = client  # type: ignore

        try:
            page.goto(url)
        except:
            print("error in goto[initial url]")
            logging.info("error in goto[initial url]")
            sys.stdout.flush()
            logging.info(traceback.format_exc())
            # raise e

        # set the first page as the current page
        self.page = self.context.pages[0]
        self.page.bring_to_front()

        self.html_content = self.page.content()

        # print(self.html_content)

    async def async_setup(self, url) -> None:
        async with async_playwright() as p:
            self.browser = await p.chromium.launch(headless=True, slow_mo=0)
            viewport_size = self.viewport_size.copy()
            self.context = await self.browser.new_context(
                viewport=viewport_size, device_scale_factor=1
            )
            if not url.startswith("http"):
                self.page = None
                return
            self.page = await self.context.new_page()
            client = await self.page.context.new_cdp_session(
                self.page
            )  # talk to chrome devtools
            self.page.client = client  # type: ignore
            await self.page.goto(url)

            self.html_content = await self.page.content()

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
        # TODO: revisit this
        # if not self.reset_finished:
        # raise RuntimeError("Call reset first before calling step.")

        # print('self.page before = {}'.format(self.page))

        success = False
        fail_error = ""

        # som_image_obs, parsed_html_str = self.observation_handler.action_processor.process(self.page, self.page.client, None)
        # som_image_obs, parsed_html_str = self.observation_handler.action_processor.process_new(self.page, self.page.client, None)

        logging.info("action = {}".format(action))
        logging.info("action = {}".format(action["action_type"]))

        """
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
        """
        if action["action_type"] == ActionTypes.SELECT:
            # do the necessary processing to get id2selector
            interactive_rects = get_interactive_elements_with_playwright(self.page)

            logging.info("interactive_rects = {}".format(interactive_rects))
            logging.info("rects = {}".format(self.image_processor.rects))

            self.image_processor.id2selector = {}

            for box_id in self.image_processor.rects:
                box = self.image_processor.rects[box_id]

                box_coord = (
                    box["rects"][0]["x"],
                    box["rects"][0]["y"],
                    box["rects"][0]["width"],
                    box["rects"][0]["height"],
                )
                idx = find_closest_center_coordinate(box_coord, interactive_rects)

                if idx is not None:
                    self.image_processor.id2selector[box_id] = interactive_rects[idx][4]

            logging.info("id2selector = {}".format(self.image_processor.id2selector))

        try:
            self.page = execute_action(
                action,
                self.page,
                self.context,
                # self.observation_handler.action_processor,
                self.image_processor,
            )

            self.page.wait_for_load_state("load", timeout=5000)

            success = True
        except Exception as e:
            fail_error = str(e)
            # traceback.print_exc()
            logging.error(traceback.format_exc())

        logging.info("Action executed successfully: {}".format(success))

        # hard sleep TODO[shuyanzh] suboptimal, may need to check network
        # TODO: revisit this
        # if self.sleep_after_execution > 0:
        # time.sleep(self.sleep_after_execution)

        return success


@beartype
class ActionParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
