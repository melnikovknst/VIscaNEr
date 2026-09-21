import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./frontend/tests",
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:8000",
    headless: true,
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: {
        ...devices["Desktop Edge"],
        channel: "msedge",
        viewport: { width: 1440, height: 1100 },
      },
    },
    {
      name: "mobile",
      use: {
        ...devices["iPhone 13"],
        defaultBrowserType: "chromium",
        channel: "msedge",
      },
    },
  ],
});
