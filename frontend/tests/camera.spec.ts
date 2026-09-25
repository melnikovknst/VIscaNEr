import { test, expect } from "@playwright/test";

// Chromium's built-in fake camera: a moving test pattern instead of a real
// device, with the permission granted up front.
test.use({
  permissions: ["camera"],
  launchOptions: {
    args: [
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
    ],
  },
});

test("the in-page camera shows a crosshair and hands the shot to the scanner", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  const open =
    testInfo.project.name === "mobile"
      ? page.getByRole("button", { name: "Сфотографировать", exact: true })
      : page.getByRole("button", { name: "Сделать фото с камеры" });
  await open.click();

  const camera = page.getByRole("dialog", { name: "Камера" });
  await expect(camera).toBeVisible();
  await expect(
    camera.getByText("Наведите перекрестие на этикетку"),
  ).toBeVisible();
  // The crosshair marks the frame centre - the point the recogniser uses.
  const box = await camera.locator(".crosshair").boundingBox();
  const size = page.viewportSize()!;
  expect(Math.abs(box!.x + box!.width / 2 - size.width / 2)).toBeLessThan(2);
  expect(Math.abs(box!.y + box!.height / 2 - size.height / 2)).toBeLessThan(2);
  await expect
    .poll(() =>
      camera.locator("video").evaluate((v: HTMLVideoElement) => v.videoWidth),
    )
    .toBeGreaterThan(100);
  await page.screenshot({ path: `tmp/ui-${testInfo.project.name}-camera.png` });

  await camera.getByRole("button", { name: "Сделать снимок" }).click();
  await expect(camera).toHaveCount(0);
  await expect(
    page.getByAltText("Выбранная фотография этикетки"),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Переснять" })).toBeVisible();
  // The stream is released once the viewfinder closes.
  expect(
    await page.evaluate(() => document.body.classList.contains("camera-open")),
  ).toBeFalsy();

  // Closing without a shot returns to the scanner untouched.
  await page.getByRole("button", { name: "Переснять" }).click();
  await expect(camera).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(camera).toHaveCount(0);
  expect(errors).toEqual([]);
});
