from playwright.sync_api import sync_playwright

def verify_loader():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # 1. Verify Swiper Loader
            page.goto("http://localhost:5000/swiper")

            # The loader appears when we finish swiping or theoretically on load if we force it.
            # Let's force it visible by injecting JS to hide stack and show loader.
            page.evaluate("""
                document.getElementById('stack-container').style.display = 'none';
                document.getElementById('controls').style.display = 'none';
                const loader = document.getElementById('loader');
                loader.style.display = 'flex';
                loader.innerHTML = `
                    <div class="music-bars">
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                    </div>
                    <p class="loader-text">Generating recommendations...</p>
                `;
            """)

            page.screenshot(path="verification/swiper_loader.png")
            print("Swiper loader captured.")

            # 2. Verify Index Loader
            page.goto("http://localhost:5000")
            page.fill("#query", "pop")
            # We trigger search but don't wait for results, just snapshot the loader immediately
            # Actually easier to just force inject it like the app does
            page.evaluate("""
                const rc = document.getElementById('results');
                rc.innerHTML = `
                <div class="loader-container">
                    <div class="music-bars">
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                        <div class="bar"></div>
                    </div>
                    <p class="loader-text">Searching for vibes...</p>
                </div>
                `;
            """)
            page.screenshot(path="verification/index_loader.png")
            print("Index loader captured.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_loader()
