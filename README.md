# Shivam's AI Personal Assistant

A free, always-on **work PA** for Shivam Negi (Internal Auditor & Trainer, Moustache).
Reasons, reminds, reflects, reports — and researches the web on command.
Runs on GitHub Actions · a Google Sheet brain · Telegram interface · a crew of free AI providers.

## What it does
- **/today** – ranked Top-5 tasks, each with a ✅ Done button
- **/add**, **/done** – manage tasks from chat
- **/reflect** – nightly 5-question check-in → logged to the sheet
- **/report** – Director report DRAFT with 👍 Approve / ✏️ Edit / 🔄 Regenerate (you send it, review-then-send)
- **/ask**, **/research** – it thinks, uses tools, and searches the web for you
- Scheduled: morning Top-5 (7am), nightly reflection (9pm), daily/weekly/monthly summaries

## Free AI layers (auto-fallback)
Groq → Cerebras → OpenRouter → Gemini. Add a key + it lights up. Slots ready for Cloudflare & SambaNova.

## Setup (once)
1. **Create a PUBLIC GitHub repo** (public = unlimited free Actions minutes) and add these 3 files
   + the `.github/workflows/` folder.
2. Add **Repository Secrets** (Settings → Secrets and variables → Actions → New repository secret):

   | Secret | Value |
   |---|---|
   | `TELEGRAM_TOKEN` | your BotFather token |
   | `TELEGRAM_CHAT_ID` | your numeric chat id |
   | `SHEET_ID` | the id in your Sheet URL (`/d/<THIS>/edit`) |
   | `GOOGLE_SA_JSON` | the FULL contents of the service-account JSON key file |
   | `GROQ_API_KEY` | `gsk_...` |
   | `GEMINI_API_KEY` | your Gemini key |
   | `CEREBRAS_API_KEY` | *(optional, add later)* |
   | `OPENROUTER_API_KEY` | *(optional, add later)* |

3. **Test:** Actions tab → **PA Reminders** → Run workflow → job = `ping`. You should get a Telegram message with a button.
4. **Go live:** Actions tab → **PA Listener** → Run workflow. Buttons + commands are now real-time. The reminder schedules run on their own.

## Notes
- Secrets are encrypted and never appear in logs, even in a public repo. Only the (generic) code is public; your data lives in your private Google Sheet.
- Times are IST. Change the cron lines in `reminders.yml` to adjust.
- The service account (`shivam-automates@…`) must have **Editor** access to the sheet (already granted).
