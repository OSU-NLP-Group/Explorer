from playwright.async_api import async_playwright

import asyncio


async def main():
    async with async_playwright() as playwright:
        # configure browser with proxy

        browser = await playwright.chromium.launch(
            # proxy={
            #     "server": "2.59.181.125:19056",
            # },
            args=["--proxy-server=http://2.59.181.125:19056"]
        )

        context = await browser.new_context()

        page = await context.new_page()

        await page.goto("https://httpbin.io/ip")

        html_content = await page.content()

        print(html_content)

        await context.close()

        await browser.close()


asyncio.run(main())
