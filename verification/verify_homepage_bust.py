from playwright.sync_api import sync_playwright

def verify_homepage():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Wait for app to start
            page.goto("http://localhost:5000", timeout=10000)

            # Take screenshot of the homepage
            page.screenshot(path="verification/homepage_v2_cachebust.png")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_homepage()
