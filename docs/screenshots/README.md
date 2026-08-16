# Screenshots wanted

The root [`README.md`](../../README.md) already has the `<img>` tags in place.
Save each capture here with the **exact filename** below and it will appear —
no README edit needed.

Sarathi is a native Windows desktop app, so these have to be captured on a
machine actually running it (`npm start`).

| # | Filename | Screen | Route | What should be visible |
| :-- | :--- | :--- | :--- | :--- |
| 1 | `01-system-info.png` | Hardware profiler | `/system` | Detected GPU name, dedicated VRAM vs shared memory, system RAM, backend |
| 2 | `02-browse-catalog.png` | Model catalog | `/browse` | The Recommended / Compatible / May Run grouping, with real model cards and sizes |
| 3 | `03-storage-download.png` | Download & installed models | `/models` | A download in progress (progress bar + speed) **or** the installed model shelf with classifications |
| 4 | `04-launch-agent.png` | Launch / agent connect | `/launch` | The provider list and the gateway address a coding agent connects to |
| 5 | `05-lora-adapters.png` | LoRA skill files | `/lora` | Discovered capability adapters (coding, reasoning, tool-calling, mathematics, research) and their validation state |
| 6 | `06-dharma-yatra.png` | Dharma Yatra startup screen | *launched terminal* | The terminal window a launched tool opens — Ratha / Yoddha / Astra / Sena / Chakra with real values |

## Capture notes

- **Window size**: resize Sarathi to roughly **1280×800** before capturing. The
  README renders each image at 420 px wide in a two-column table, so anything
  much wider loses legibility.
- **Format**: PNG. Windows `Win + Shift + S` (Snip) or `Win + PrtScn` is fine.
- **Compress before committing.** Keep each file **under ~300 KB** so the README
  stays fast for a casual visitor. Any of these work:
  - drag onto <https://squoosh.app> and re-export as PNG or WebP
  - `oxipng -o 4 --strip safe docs/screenshots/*.png`
  - `pngquant --quality=65-85 --ext .png --force docs/screenshots/*.png`
- **Check before you commit**: `ls -lh docs/screenshots/` — if the six PNGs total
  more than ~2 MB, compress again.
- **Redact anything personal**: local file paths under `C:\Users\<you>\`, Hugging
  Face tokens, or workspace names you would rather not publish. The repo is
  public.
