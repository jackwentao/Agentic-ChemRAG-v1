import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

test("uploads a PDF and asks a question", async ({ page }) => {
  const fixture = process.env.E2E_PDF_FIXTURE ?? path.resolve("../test_pdf/Synthesis, Characterization, Luminescent.pdf");
  test.skip(!fs.existsSync(fixture), `PDF fixture not found: ${fixture}`);
  await page.goto("/");
  await page.getByLabel(/Upload PDFs/i).setInputFiles(fixture);
  await expect(page.getByText(/Upload batch/i)).toBeVisible();
  await expect(page.locator(".document-status", { hasText: /chunks|failed|indexing/i }).first()).toBeVisible();
  await page.getByPlaceholder(/Ask about/i).fill("What is this paper mainly about?");
  await page.locator(".composer button").click();
  await expect(page.locator(".assistant-message").last()).toBeVisible({ timeout: 90_000 });
});
