from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    expect,
    sync_playwright,
)
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
from .processors import ObservationHandler, ObservationMetadata
from PIL import Image, ImageDraw
import logging

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

        if not self.args.use_async_playwright:
            self.tls = Tls()
        else:
            self.tls = None
            
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

    @beartype
    def setup(self, url) -> None:
        # self.context_manager = sync_playwright()
        # self.playwright = self.context_manager.__enter__()
        # self.browser = self.playwright.chromium.launch(
        #     headless=True, slow_mo=0
        # )
        self.browser = self.tls.playwright.chromium.launch(headless=True, slow_mo=0)
        # self.browser = self.tls.playwright.chromium.launch(headless=False, slow_mo=0)

        # Use custom viewport size if specified in the config, otherwise use the default.
        viewport_size = self.viewport_size.copy()

        self.context = self.browser.new_context(
            viewport=viewport_size,
            device_scale_factor=1,
        )
        if not url.startswith("http"):
            self.page = None
            return
        page = self.context.new_page()
        client = page.context.new_cdp_session(
            page
        )  # talk to chrome devtools
        page.client = client  # type: ignore

        try:
            page.goto(url)
        except Exception as e:
            logging.error(traceback.format_exc())
            raise e
        
        # set the first page as the current page
        self.page = self.context.pages[0]
        self.page.bring_to_front()
    
    async def async_setup(self, url) -> None:
        async with async_playwright() as p:
            self.browser = await p.chromium.launch(headless=True, slow_mo=0)
            viewport_size = self.viewport_size.copy()
            self.context = await self.browser.new_context(viewport=viewport_size, device_scale_factor=1)
            if not url.startswith("http"):
                self.page = None
                return
            self.page = await self.context.new_page()
            client = await self.page.context.new_cdp_session(self.page)  # talk to chrome devtools
            self.page.client = client  # type: ignore
            await self.page.goto(url)
    
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
        som_image_obs, parsed_html_str = self.observation_handler.action_processor.process_new(self.page, self.page.client, None)

        try:
            self.page = execute_action(
                action,
                self.page,
                self.context,
                self.observation_handler.action_processor,
            )

            success = True
        except Exception as e:
            fail_error = str(e)
            # traceback.print_exc()
            logging.error(traceback.format_exc())

        logging.info("Action executed successfully: {}".format(success))
        
        return

@beartype
class ActionParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)