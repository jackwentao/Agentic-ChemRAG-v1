import { expect, test } from "@playwright/test";

test("shows an error for non-PDF uploads", async ({ page }) => {
  await page.goto("/");
  const buffer = Buffer.from("not a pdf");
  await page.getByLabel(/Upload PDFs/i).setInputFiles({
    name: "notes.txt",
    mimeType: "text/plain",
    buffer
  });
  await expect(page.getByText(/Only PDF files are supported|Upload batch/i)).toBeVisible();
});
