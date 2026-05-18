# Fix Docs Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the static docs search dialog return results on the GitHub Pages deployment.

**Architecture:** Keep the docs site compatible with `output: "export"` and GitHub Pages by using Fumadocs UI static search end to end. Replace the current FlexSearch static export with the Orama static export expected by `fumadocs-ui` when `RootProvider` uses `search.options.type = "static"`.

**Tech Stack:** Next.js static export, Fumadocs UI 16.6.17, Fumadocs Core search server, pnpm, Playwright for manual/browser validation.

---

## Diagnosis

Live reproduction on `https://so101-nexus.com/docs`:

- Opening the search dialog works.
- Typing `mujoco` updates the input.
- The browser fetches `https://so101-nexus.com/api/search` and receives HTTP 200.
- The result list stays collapsed with `data-empty="true"` and no rendered results.

Root cause:

- [docs/app/layout.tsx](/home/johnsutor/Desktop/so101-nexus/docs/app/layout.tsx:15) configures Fumadocs search with `type: "static"`.
- In `fumadocs-ui@16.6.17`, `type: "static"` uses the Orama static client.
- [docs/app/api/search/route.ts](/home/johnsutor/Desktop/so101-nexus/docs/app/api/search/route.ts:2) exports a FlexSearch static index with `flexsearchFromSource(source)`.
- The deployed `/api/search` payload is FlexSearch-shaped, `{"type":"default","raw":{...}}`, not Orama-shaped.
- Because the client and exported database formats do not match, the search client never produces result items.

## Task 1: Add A Failing Search Shape Test

**Files:**

- Modify: `tests/test_docs_consistency.py`

**Step 1: Add a regression test that catches the current mismatch**

Add a test that reads the two docs files and asserts static UI search is not paired with FlexSearch static export.

```python
def test_docs_static_search_uses_orama_export() -> None:
    layout = (ROOT / "docs/app/layout.tsx").read_text()
    route = (ROOT / "docs/app/api/search/route.ts").read_text()

    assert 'type: "static"' in layout
    assert "createFromSource" in route
    assert "flexsearchFromSource" not in route
```

Use the existing `ROOT` pattern in `tests/test_docs_consistency.py`.

**Step 2: Run the narrow test and verify it fails before the fix**

Run:

```bash
uv run pytest tests/test_docs_consistency.py::test_docs_static_search_uses_orama_export -q
```

Expected before implementation:

```text
FAILED
```

## Task 2: Switch The Static Search Export To Orama

**Files:**

- Modify: `docs/app/api/search/route.ts`

**Step 1: Replace the FlexSearch route**

Change the route to:

```ts
import { source } from "@/lib/source";
import { createFromSource } from "fumadocs-core/search/server";

export const revalidate = false;
export const { staticGET: GET } = createFromSource(source);
```

This keeps `/api/search` static-exportable while matching the client selected by `type: "static"`.

**Step 2: Run the regression test**

Run:

```bash
uv run pytest tests/test_docs_consistency.py::test_docs_static_search_uses_orama_export -q
```

Expected:

```text
1 passed
```

## Task 3: Remove Unused FlexSearch Dependency

**Files:**

- Modify: `docs/package.json`
- Modify: `docs/pnpm-lock.yaml`

**Step 1: Remove dependencies no longer used by the docs app**

Remove these entries from `docs/package.json`:

```json
"flexsearch": "^0.8.212"
```

Remove this dev dependency:

```json
"@types/flexsearch": "^0.7.42"
```

**Step 2: Refresh the lockfile**

Run:

```bash
cd docs && pnpm install --lockfile-only
```

Expected:

```text
Done
```

## Task 4: Build And Inspect The Static Export

**Files:**

- Generated: `docs/out/api/search`

**Step 1: Build the docs**

Run:

```bash
cd docs && pnpm build
```

Expected:

```text
Export successful
```

**Step 2: Inspect the generated search payload**

Run:

```bash
node -e 'const fs=require("node:fs"); const data=JSON.parse(fs.readFileSync("docs/out/api/search","utf8")); console.log(data.type, Object.keys(data));'
```

Expected:

```text
advanced [...]
```

The important check is that the payload is not `type: "default"` with a `raw` key.

## Task 5: Browser-Validate Local Static Search

**Files:**

- No source changes.

**Step 1: Serve the static export**

Run:

```bash
cd docs && pnpm dlx serve out -l 4173
```

Keep this running while executing the browser check.

**Step 2: Use Playwright to verify the dialog returns results**

Run:

```bash
NODE_PATH=/tmp/so101-pw/node_modules node -e 'const { chromium } = require("playwright"); (async () => { const browser = await chromium.launch({ headless: true }); const page = await browser.newPage({ viewport: { width: 1280, height: 900 } }); await page.goto("http://127.0.0.1:4173/docs", { waitUntil: "networkidle" }); await page.getByRole("button", { name: /search/i }).first().click(); await page.keyboard.type("mujoco"); await page.waitForTimeout(2000); const text = await page.locator("[role=dialog]").innerText(); if (!/MuJoCo/i.test(text)) throw new Error(text); await browser.close(); })();'
```

Expected:

```text
no output, exit code 0
```

## Task 6: Run The Existing Docs Check

**Files:**

- No source changes.

**Step 1: Run docs checks**

Run:

```bash
make docs-check
```

Expected:

```text
tests pass and docs build succeeds
```

## Task 7: Commit

**Files:**

- `docs/app/api/search/route.ts`
- `docs/package.json`
- `docs/pnpm-lock.yaml`
- `tests/test_docs_consistency.py`

**Step 1: Commit the fix**

Run:

```bash
git add docs/app/api/search/route.ts docs/package.json docs/pnpm-lock.yaml tests/test_docs_consistency.py
git commit -m "fix: align docs static search index"
```

