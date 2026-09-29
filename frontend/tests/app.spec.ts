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

test("upload is honest about the model and help is keyboard accessible", async ({
  page,
  request,
}) => {
  await page.goto("/");
  const health = await (await request.get("/api/health")).json();
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
  // The preview step always offers a way back out.
  await expect(
    page.getByRole("button", { name: "Выбрать другое" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Распознать вино" }).click();
  if (health.model_ready) {
    // A real model answers with a card or with a choice - never an error -
    // and the visitor's photo stays on screen for comparison.
    await expect(page.locator(".wine-detail, .result-chooser")).toBeVisible({
      timeout: 60_000,
    });
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect(page.getByAltText("Ваше фото этикетки")).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Сделать другое фото" }).first(),
    ).toBeVisible();
  } else {
    await expect(page.getByRole("alert")).toContainText(
      "Распознавание временно недоступно",
    );
    await page.getByRole("button", { name: "Убрать фото" }).click();
  }
  await page.getByRole("button", { name: "Как это работает" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("an uncertain answer offers the best guess first and the rest on request", async ({
  page,
  request,
}) => {
  const meta = await (await request.get("/api/catalog/meta")).json();
  const wines = meta.featured.slice(0, 4);
  await page.route("**/api/scan", (route) =>
    route.fulfill({
      json: {
        id: "test",
        status: "uncertain",
        wine: null,
        candidates: wines.map((wine: { slug: string }, i: number) => ({
          wine,
          confidence: 0.58 - i / 100,
        })),
        confidence: 0.58,
        margin: 0.01,
        elapsed_ms: 40,
        model_version: "test",
        provider: "five_stream",
        created_at: new Date().toISOString(),
        message: "",
      },
    }),
  );
  await page.goto("/");
  const image = await request.get(meta.featured[0].image_url);
  await page.getByLabel("Выбрать фото этикетки").setInputFiles({
    name: "label.png",
    mimeType: "image/png",
    buffer: await image.body(),
  });
  await page.getByRole("button", { name: "Распознать вино" }).click();
  await expect(
    page.getByRole("heading", { name: "Похоже, это оно" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: wines[0].name }),
  ).toBeVisible();
  // The other candidates are collapsed and unreachable until asked for.
  const others = page.locator("#other-guesses");
  await expect(others).toHaveAttribute("inert");
  expect((await others.boundingBox())?.height ?? 0).toBeLessThan(2);
  await page.getByRole("button", { name: "Другие варианты" }).click();
  await expect(others).not.toHaveAttribute("inert");
  await expect.poll(async () => (await others.boundingBox())?.height ?? 0).toBeGreaterThan(200);
  await expect(
    others.getByRole("button", { name: new RegExp(wines[1].name) }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Да, это оно" }).click();
  await expect(
    page.getByRole("heading", { name: wines[0].name }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});
