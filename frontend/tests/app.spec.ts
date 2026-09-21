import { test, expect, type Page } from "@playwright/test";

async function prepareScreenshot(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  // Full-page captures must also load images outside the current viewport.
  await page.locator("img").evaluateAll((images) => {
    images.forEach((image) => {
      image.loading = "eager";
    });
  });
  const cards = page.locator(".wine-card");
  if (await cards.count()) await cards.last().scrollIntoViewIfNeeded();
  await expect
    .poll(() =>
      page
        .locator("img")
        .evaluateAll((images) =>
          images
            .filter((image) => image.getClientRects().length > 0)
            .every((image) => image.complete && image.naturalWidth > 0),
        ),
    )
    .toBeTruthy();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
}

test("scanner, catalog card, collection and history form a working flow", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Ваше вино/ })).toBeVisible();
  await expect(page.getByText("2 103", { exact: true })).toHaveCount(1);
  await expect(
    page.getByText(
      /Модель готовится|Демонстрационный режим|Модель не подключена/,
    ),
  ).toHaveCount(0);
  await prepareScreenshot(page);
  await page.screenshot({
    path: `tmp/ui-${testInfo.project.name}-scanner.png`,
    fullPage: true,
  });
  await page.locator(".wine-image-button").first().click();
  await expect(
    page.getByRole("heading", { name: /LETO Каберне/ }),
  ).toBeVisible();
  await expect(page.locator(".result-notice")).toHaveCount(0);
  await prepareScreenshot(page);
  await page.screenshot({
    path: `tmp/ui-${testInfo.project.name}-result.png`,
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Сохранить в коллекцию", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "В вашей коллекции" }),
  ).toBeVisible();
  await page.reload();
  await page.goto("/#saved");
  await expect(
    page.getByRole("heading", { name: /LETO Каберне/ }),
  ).toBeVisible();
  await page.goto("/#history");
  await expect(
    page.getByRole("heading", { name: "История знакомства" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Здесь появятся ваши открытия" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});

test("catalog filtering and useful food pairing", async ({
  page,
}, testInfo) => {
  await page.goto("/#catalog");
  await page
    .getByRole("textbox", { name: "Поиск по каталогу" })
    .fill("Рислинг");
  await expect(page.locator(".wine-card")).not.toHaveCount(0);
  await expect(page.locator(".wine-card h3").first()).toContainText(/Рислинг/i);
  await prepareScreenshot(page);
  await page.screenshot({
    path: `tmp/ui-${testInfo.project.name}-catalog.png`,
    fullPage: true,
  });
  await page
    .getByRole("textbox", { name: "Поиск по каталогу" })
    .fill("совершеннонесуществующеевино");
  await expect(
    page.getByRole("heading", { name: "Пока ничего не нашли" }),
  ).toBeVisible();
  await page.goto("/#sommelier");
  await page.getByRole("button", { name: "Мясо и гриль" }).click();
  await page
    .getByRole("button", { name: "Подобрать вино", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "К мясу", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".pairing-results .wine-card")).toHaveCount(3);
  await prepareScreenshot(page);
  await page.screenshot({
    path: `tmp/ui-${testInfo.project.name}-pairing.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});

test("upload handles a missing model honestly and help is keyboard accessible", async ({
  page,
  request,
}) => {
  await page.goto("/");
  const meta = await (await request.get("/api/catalog/meta")).json();
  const image = await request.get(meta.featured[0].image_url);
  await page.getByLabel("Выбрать фото этикетки").setInputFiles({
    name: "label.png",
    mimeType: "image/png",
    buffer: await image.body(),
  });
  await expect(
    page.getByAltText("Выбранная фотография этикетки"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Распознать вино" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Распознавание временно недоступно",
  );
  await page.getByRole("button", { name: "Убрать фото" }).click();
  await page.getByRole("button", { name: "Как это работает" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});
