#!/usr/bin/env python3
"""
Shivam's AI Personal Assistant  —  a multi-layer, thinking work PA.

One file, many jobs. Run with:  python pa_bot.py <job>
Jobs: ping | morning | reflection | collect | daily | weekly | monthly | report | listen

Reads/writes ONE Google Sheet (tabs: PA_Config, PA_Goals, PA_Tasks, PA_Reflections,
PA_DailyLog, and later PA_Calendar, PA_Inbox, PA_Reports). Talks to you on Telegram with
buttons + slash-commands. Thinks with a crew of FREE AI providers (auto-fallback) and can
research the web when asked.
"""
import os, sys, json, time, html, re, datetime, traceback
import requests

# ----------------------------------------------------------------------------- ENV / SECRETS
TG_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT    = os.environ.get("TELEGRAM_CHAT_ID", "")
SHEET_ID   = os.environ.get("SHEET_ID", "")
SA_JSON    = os.environ.get("GOOGLE_SA_JSON", "")
TZ_OFFSET  = 5.5  # IST; overridden by PA_Config TIMEZONE if present

TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# ----------------------------------------------------------------------------- AI ROUTER (deep free bench)
# A purpose walks a long fallback chain across many FREE providers. Each layer is
# skipped if its key is missing, retried on transient errors (429/5xx), and rolled
# past on any failure. If EVERY layer is down, ai() returns "" and callers degrade
# gracefully — so the assistant itself never hard-fails. Add a provider by dropping
# its key in as a GitHub Secret; no code change needed.

# OpenAI-compatible providers:  name -> (endpoint, key_env_var)
OAI = {
    "groq":       ("https://api.groq.com/openai/v1/chat/completions",        "GROQ_API_KEY"),
    "cerebras":   ("https://api.cerebras.ai/v1/chat/completions",            "CEREBRAS_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions",          "OPENROUTER_API_KEY"),
    "sambanova":  ("https://api.sambanova.ai/v1/chat/completions",           "SAMBANOVA_API_KEY"),
    "mistral":    ("https://api.mistral.ai/v1/chat/completions",             "MISTRAL_API_KEY"),
    "github":     ("https://models.inference.ai.azure.com/chat/completions", "GH_MODELS_KEY"),
    "nvidia":     ("https://integrate.api.nvidia.com/v1/chat/completions",   "NVIDIA_API_KEY"),
    "together":   ("https://api.together.xyz/v1/chat/completions",           "TOGETHER_API_KEY"),
}

# Purpose-built chains. Different jobs call different purposes; each chain is many
# layers deep, so the crew keeps answering even if several providers fail at once.
TIERS = {
    # QUICK — reminders, parsing, JSON extraction, quick chat
    "quick": [
        ("groq",       "llama-3.1-8b-instant"),
        ("cerebras",   "llama3.1-8b"),
        ("groq",       "llama-3.3-70b-versatile"),
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
        ("github",     "gpt-4o-mini"),
        ("mistral",    "mistral-small-latest"),
        ("together",   "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        ("sambanova",  "Meta-Llama-3.3-70B-Instruct"),
        ("gemini",     "gemini-2.5-flash"),
    ],
    # DEEP — reasoning, planning, research ("thinks to depth")
    "deep": [
        ("groq",       "deepseek-r1-distill-llama-70b"),
        ("openrouter", "deepseek/deepseek-r1:free"),
        ("nvidia",     "deepseek-ai/deepseek-r1"),
        ("cerebras",   "llama-3.3-70b"),
        ("github",     "gpt-4o"),
        ("gemini",     "gemini-2.5-flash"),
        ("groq",       "llama-3.3-70b-versatile"),
    ],
    # LONG — big-context summaries + reports
    "long": [
        ("gemini",     "gemini-2.5-flash"),
        ("cerebras",   "llama-3.3-70b"),
        ("mistral",    "mistral-large-latest"),
        ("groq",       "llama-3.3-70b-versatile"),
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
        ("github",     "gpt-4o"),
    ],
}

def _k(env):
    return os.environ.get(env, "")

def _has_key(name):
    if name == "gemini":
        return bool(_k("GEMINI_API_KEY"))
    return name in OAI and bool(_k(OAI[name][1]))

def _post_oai(name, model, system, prompt):
    url, env = OAI[name]
    headers = {"Authorization": f"Bearer {_k(env)}", "Content-Type": "application/json"}
    if name == "openrouter":
        headers["HTTP-Referer"] = "https://github.com"; headers["X-Title"] = "Shivam PA"
    r = requests.post(url, headers=headers, timeout=60, json={
        "model": model, "temperature": 0.4,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}]})
    if r.status_code in (408, 409, 425, 429, 500, 502, 503, 529):
        raise RuntimeError(f"{name} transient {r.status_code}")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def _post_gemini(model, system, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={_k('GEMINI_API_KEY')}"
    r = requests.post(url, headers={"Content-Type": "application/json"}, timeout=60, json={
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}]})
    if r.status_code in (429, 500, 502, 503):
        raise RuntimeError(f"gemini transient {r.status_code}")
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

def ai(prompt, system="You are Shivam's sharp, concise work personal assistant.",
       purpose="quick", tier=None):
    """Walk the purpose's fallback chain (retrying transient errors). Returns (text, provider).
    On total failure returns ('', 'none') so callers can degrade gracefully."""
    if tier:  # backward-compat: old calls used tier="fast"/"deep"/"long"
        purpose = {"fast": "quick"}.get(tier, tier)
    for name, model in TIERS.get(purpose, TIERS["quick"]):
        if not _has_key(name):
            continue
        for attempt in range(2):
            try:
                out = _post_gemini(model, system, prompt) if name == "gemini" \
                      else _post_oai(name, model, system, prompt)
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
                if out:
                    return out, f"{name}:{model}"
                break  # empty answer -> next provider
            except Exception:
                time.sleep(1.0 * (attempt + 1))
                continue
    return "", "none"

# ----------------------------------------------------------------------------- TELEGRAM
def tg(method, **payload):
    r = requests.post(f"{TG_API}/{method}", json=payload, timeout=30)
    return r.json()

def _md_to_tg(t):
    """Clean model output into Telegram-safe HTML: drop code fences, convert markdown."""
    if not t:
        return t
    t = re.sub(r"```[a-zA-Z]*\n?", "", t).replace("```", "")   # code fences
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.DOTALL)  # **bold**
    t = re.sub(r"__(.+?)__", r"<b>\1</b>", t, flags=re.DOTALL)      # __bold__
    t = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"<b>\1</b>", t)             # ## headers
    t = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", t)                       # - / * bullets
    return t.strip()

def _plain(t):
    return re.sub(r"<[^>]+>", "", t or "")

def send(text, buttons=None, chat=None):
    text = _md_to_tg(text)
    payload = {"chat_id": chat or TG_CHAT, "text": text,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    if buttons:
        payload["reply_markup"] = {"inline_keyboard": buttons}
    r = tg("sendMessage", **payload)
    if not r.get("ok"):                       # bad HTML -> resend as plain text
        payload["text"] = _plain(text); payload.pop("parse_mode", None)
        r = tg("sendMessage", **payload)
    return r

def edit(chat, message_id, text, buttons=None):
    text = _md_to_tg(text)
    payload = {"chat_id": chat, "message_id": message_id, "text": text,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    if buttons:
        payload["reply_markup"] = {"inline_keyboard": buttons}
    r = tg("editMessageText", **payload)
    if not r.get("ok"):
        payload["text"] = _plain(text); payload.pop("parse_mode", None)
        r = tg("editMessageText", **payload)
    return r

def answer_cb(cb_id, text=""):
    return tg("answerCallbackQuery", callback_query_id=cb_id, text=text)

def btn(label, data):
    return {"text": label, "callback_data": data}

# ----------------------------------------------------------------------------- GOOGLE SHEET (the brain)
_gc = None
def sheet():
    global _gc
    import gspread
    if _gc is None:
        _gc = gspread.service_account_from_dict(json.loads(SA_JSON))
    return _gc.open_by_key(SHEET_ID)

def ws(tab):
    return sheet().worksheet(tab)

def config():
    try:
        rows = ws("PA_Config").get_all_records()
        return {r["Key"]: r["Value"] for r in rows if r.get("Key")}
    except Exception:
        return {}

def set_config(key, value):
    w = ws("PA_Config")
    cells = w.findall(str(key))
    if cells:
        w.update_cell(cells[0].row, 2, str(value))
    else:
        w.append_row([key, str(value), ""])

def open_tasks():
    try:
        rows = ws("PA_Tasks").get_all_records()
    except Exception:
        return []
    out = []
    for r in rows:
        st = str(r.get("Status", "")).strip().lower()
        if st in ("done", "cancelled", "complete", "completed"):
            continue
        if not str(r.get("Task", "")).strip():
            continue
        out.append(r)
    return out

def rank_tasks(rows):
    def key(r):
        p = str(r.get("Priority", "P3")).upper()
        prio = {"P1": 1, "P2": 2, "P3": 3}.get(p, 3)
        due = str(r.get("Due Date", "")).strip()
        try:
            d = datetime.date.fromisoformat(due)
        except Exception:
            d = datetime.date.max
        return (prio, d)
    return sorted(rows, key=key)

def log_event(kind, detail, goal=""):
    try:
        now = datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET)
        ws("PA_DailyLog").append_row(
            [now.strftime("%Y-%m-%d %H:%M"), now.strftime("%Y-%m-%d"), kind, detail, goal])
    except Exception:
        pass

def now_ist():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET)

# ----------------------------------------------------------------------------- WEB RESEARCH TOOL
def web_search(query, n=5):
    """Free web search via DuckDuckGo (no API key). Returns list of {title,url,body}."""
    try:
        from ddgs import DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return []
    try:
        with DDGS() as d:
            return list(d.text(query, max_results=n))
    except Exception:
        return []

def research(query):
    hits = web_search(query, 6)
    if not hits:
        return "I couldn't reach the web just now — try again in a moment."
    src = "\n\n".join(f"[{i+1}] {h.get('title','')}\n{h.get('body','')}\n{h.get('href') or h.get('url','')}"
                      for i, h in enumerate(hits))
    ans, _ = ai(
        system=("You are Shivam's research analyst (hospitality: hostels, audit, training in India). "
                "Answer his question using ONLY the sources. Be specific and practical. "
                "End with a short 'Sources:' list of the [n] you used. Format for Telegram HTML "
                "(use <b> for emphasis, no markdown)."),
        prompt=f"Question: {query}\n\nSources:\n{src}", tier="long")
    if not ans:  # AI down — still hand over the raw findings so research never returns nothing
        ans = "<b>Top results</b> (AI busy — raw sources):\n" + "\n".join(
            f"• {html.escape(h.get('title',''))} — {h.get('href') or h.get('url','')}" for h in hits[:5])
    return ans

# ----------------------------------------------------------------------------- AGENT (thinks + uses tools)
AGENT_SYS = (
    "You are Shivam Negi's personal work assistant. Shivam is an Internal Auditor & Trainer at "
    "Moustache (India Hostels Pvt Ltd). Think step by step and go to real depth. You have tools:\n"
    "  SEARCH: <query>   — search the web (use for anything current, factual, or industry research)\n"
    "  TASKS             — read Shivam's open task list from his sheet\n"
    "  DONE              — you have enough; write the final answer after it\n"
    "Respond with EXACTLY ONE line: either 'SEARCH: ...', 'TASKS', or 'DONE: <final answer>'.\n"
    "Prefer to search when a question touches current facts, prices, tools, or best practices."
)

def run_agent(user_text):
    context = f"User asked: {user_text}\n"
    for _ in range(3):
        step, _p = ai(system=AGENT_SYS, prompt=context, tier="deep")
        if not step.strip():           # reasoning layers down -> fall back to a direct web search
            return research(user_text)
        line = step.strip().splitlines()[0]
        if line.upper().startswith("SEARCH:"):
            q = line.split(":", 1)[1].strip()
            context += f"\nI searched '{q}'. Findings:\n{research(q)}\n"
        elif line.upper().startswith("TASKS"):
            t = rank_tasks(open_tasks())
            context += "\nOpen tasks:\n" + "\n".join(
                f"- [{x.get('Priority')}] {x.get('Task')} (due {x.get('Due Date')})" for x in t[:15]) + "\n"
        elif line.upper().startswith("DONE:"):
            return line.split(":", 1)[1].strip()
        else:
            return step.strip()
    # final synthesis if loop exhausted
    final, _ = ai(system="Summarise the answer for Shivam in Telegram HTML.", prompt=context, tier="long")
    return final or research(user_text)

# ----------------------------------------------------------------------------- JOBS
def job_ping():
    send("✅ <b>Your PA is alive.</b>\nThe engine, Sheet link and Telegram are all wired up. "
         "Tap below to test a button.",
         buttons=[[btn("👋 Say hi", "ping:hi")], [btn("📋 Today's tasks", "cmd:today")]])

def job_morning():
    tasks = rank_tasks(open_tasks())
    d = now_ist().strftime("%A, %d %b")
    if not tasks:
        send(f"☀️ <b>Good morning, Shivam</b> — {d}\nNo open tasks in your sheet. "
             "Add one with <code>/add</code> or enjoy the clear runway.")
        return
    top = tasks[:5]
    lines = [f"☀️ <b>Good morning, Shivam</b>", f"Your Top {len(top)} for today — {d}\n"]
    rows = []
    for i, t in enumerate(top, 1):
        due = str(t.get("Due Date", "")).strip()
        due_s = f" · due {due}" if due else ""
        lines.append(f"<b>{i}.</b> [{t.get('Priority','P3')}] {html.escape(str(t.get('Task','')))}{due_s}")
    for i, t in enumerate(top, 1):
        rows.append(btn(f"✅ {i}", f"done:{t.get('Task ID','')}"))
    nudge, _ = ai(prompt="One short, warm one-line nudge to start the workday. No emoji.", tier="fast")
    nudge = nudge or "One focused block at a time — you've got this."
    lines.append(f"\n<i>{html.escape(nudge)}</i>")
    send("\n".join(lines), buttons=[rows, [btn("➕ Add task", "cmd:addhelp")]])

def job_reflection():
    c = config()
    qs = [c.get(f"NIGHTLY_Q{i}") for i in range(1, 6) if c.get(f"NIGHTLY_Q{i}")]
    if not qs:
        qs = ["What did you finish today?", "What's blocked?", "What did you learn?",
              "Energy (1-5)?", "Tomorrow's #1 priority?"]
    body = "🌙 <b>Nightly check-in</b>\nReply in one message (a line each):\n\n" + \
           "\n".join(f"{i}. {html.escape(q)}" for i, q in enumerate(qs, 1))
    set_config("STATE_awaiting_reflection", "1")
    send(body, buttons=[[btn("😴 Skip tonight", "reflect:skip")]])

def job_collect():
    """Read any pending reflection reply from Telegram and log it."""
    c = config()
    if c.get("STATE_awaiting_reflection") != "1":
        return
    updates = tg("getUpdates", offset=int(c.get("STATE_tg_offset", 0)) or None, timeout=0).get("result", [])
    answer = None
    last_id = int(c.get("STATE_tg_offset", 0))
    for u in updates:
        last_id = max(last_id, u.get("update_id", 0) + 1)
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) == str(TG_CHAT) and msg.get("text") \
           and not msg["text"].startswith("/"):
            answer = msg["text"]
    set_config("STATE_tg_offset", last_id)
    if not answer:
        return
    parsed, _ = ai(system="Split the user's nightly reflection into a JSON object with keys: "
                          "finished, blocked, learned, energy, tomorrow. Values are short strings. "
                          "Return ONLY JSON.",
                   prompt=answer, tier="fast")
    try:
        j = json.loads(re.search(r"\{.*\}", parsed, re.DOTALL).group())
    except Exception:
        j = {"finished": answer, "blocked": "", "learned": "", "energy": "", "tomorrow": ""}
    ws("PA_Reflections").append_row([now_ist().strftime("%Y-%m-%d"),
        j.get("finished", ""), j.get("blocked", ""), j.get("learned", ""),
        str(j.get("energy", "")), j.get("tomorrow", "")])
    log_event("reflection", "Nightly check-in saved")
    set_config("STATE_awaiting_reflection", "0")
    send("📝 Logged your reflection. Rest well — I'll have your Top-5 ready at 7am.")

def _summary(period, rows_desc):
    txt, prov = ai(system="You are Shivam's chief-of-staff. Write a crisp, scannable "
                          f"{period} summary in Telegram HTML (<b> headers, short lines). "
                          "Sections: Done · In-progress/Blocked · Next.",
                   prompt=rows_desc, tier="long")
    if not txt:  # AI down — send the raw activity so the summary still lands
        txt = f"<b>{period.title()} activity</b> (AI busy — raw log):\n" + html.escape(rows_desc[:3000])
    return txt

def _log_rows(days):
    cutoff = (now_ist() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        rows = ws("PA_DailyLog").get_all_records()
    except Exception:
        rows = []
    keep = [r for r in rows if str(r.get("Date", "")) >= cutoff]
    return "\n".join(f"{r.get('Date')} · {r.get('Type')} · {r.get('Detail')}" for r in keep) or "No activity logged."

def job_daily():
    send(_summary("end-of-day", "Today's log:\n" + _log_rows(1)))

def job_weekly():
    send("📆 <b>Weekly report</b> — here's your draft for review 👇")
    job_report("weekly")

def job_monthly():
    send("🗓️ <b>Monthly rollup</b>\n\n" + _summary("monthly", "This month's log:\n" + _log_rows(31)))

# ---------- KRA scorecard (reads your PA_KRA tab in its monthly "<Month> KRA / <Month> Score" layout)
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

def kra_rows():
    try:
        return ws("PA_KRA").get_all_records()
    except Exception:
        return []

def kra_current(rows=None):
    rows = rows if rows is not None else kra_rows()
    mon = now_ist().strftime("%B")
    ck, cs = f"{mon} KRA", f"{mon} Score"
    items = [(str(r.get(ck, "")).strip(), str(r.get(cs, "")).strip())
             for r in rows if str(r.get(ck, "")).strip()]
    return mon, items

def kra_scorecard(rows=None):
    rows = rows if rows is not None else kra_rows()
    out = []
    for m in MONTHS:
        got = tot = 0
        for r in rows:
            if str(r.get(f"{m} KRA", "")).strip():
                tot += 1
                try:
                    if int(float(r.get(f"{m} Score", 0) or 0)) >= 1:
                        got += 1
                except Exception:
                    pass
        if tot:
            out.append((m, got, tot))
    return out

def _report_context(kind):
    days = 7 if kind == "weekly" else 31
    log = _log_rows(days)
    tasks = rank_tasks(open_tasks())
    open_list = "\n".join(
        f"- [{t.get('Priority')}] {t.get('Task')} (due {t.get('Due Date')}, {t.get('Status')})"
        for t in tasks[:30]) or "None open."
    mon, kras = kra_current()
    kra_txt = "\n".join(f"- {k}  [score: {s or 'pending'}]" for k, s in kras) \
              or "No KRAs found for this month in the PA_KRA tab."
    ytd = kra_scorecard()
    ytd_txt = " · ".join(f"{m}: {g}/{t}" for m, g, t in ytd) or "n/a"
    return log, open_list, kra_txt, ytd_txt, mon

REP_BTNS = [[btn("✅ Approve & Send", "rep:send"), btn("🔄 Regenerate", "rep:regen")],
            [btn("❌ Cancel", "rep:cancel")]]

def job_report(kind="monthly"):
    """Generate an interactive report draft: review & revise in chat, then auto-email on approval."""
    c = config()
    log, open_list, kra_txt, ytd_txt, mon = _report_context(kind)
    period = (f"Week ending {now_ist().strftime('%d %b %Y')}" if kind == "weekly"
              else f"{mon} {now_ist().year}")
    sysp = (f"You are Shivam Negi's chief-of-staff writing a {kind} progress report for his Director, "
            f"{c.get('DIRECTOR_NAME', 'the Director')}. Shivam is Internal Auditor & Trainer at "
            "Moustache (India Hostels). EXECUTIVE STYLE — keep it short and skimmable, NOT a long report. "
            "Start with a 2–3 line <b>Executive Summary</b>. Then concise one-line bullets (use •) under: "
            "<b>KRA progress</b> (each KRA + its score), <b>Key completions</b>, <b>In-progress / Blocked</b>, "
            "<b>Next period</b>. No paragraphs, no filler — each bullet ≤ 15 words. Telegram HTML only.")
    prompt = (f"Period: {period}\nYear-to-date KRA scores: {ytd_txt}\n\n"
              f"This month's KRAs:\n{kra_txt}\n\nOpen tasks:\n{open_list}\n\nActivity log:\n{log}")
    draft, _ = ai(system=sysp, prompt=prompt, purpose="long")
    if not draft:
        draft = (f"<b>{kind.title()} report — {period}</b>\n\n<b>KRAs</b>\n{kra_txt}\n\n"
                 f"<b>Activity</b>\n{_plain(log)[:1500]}")
    set_config("STATE_report_mode", kind)
    set_config("STATE_report_period", period)
    set_config("STATE_report_draft", draft)
    send(f"🧾 <b>{kind.title()} report — DRAFT (not sent yet)</b>\n\n{draft}\n\n"
         "— Reply with any changes in plain words (e.g. <i>“mark risk assessment done”</i>, "
         "<i>“remove the laundry line and add I closed the Udaipur audit”</i>), or use the buttons.",
         buttons=REP_BTNS)

def revise_report(instruction):
    c = config()
    draft = c.get("STATE_report_draft", "")
    new, _ = ai(system=("Revise Shivam's report exactly per his instruction. Keep it professional and "
                        "in Telegram HTML. Return the FULL revised report only, no preamble."),
                prompt=f"Current report:\n{draft}\n\nInstruction: {instruction}", purpose="long")
    new = new or draft
    set_config("STATE_report_draft", new)
    send(f"✏️ <b>Updated draft</b>\n\n{new}\n\nMore changes? Or approve to send.", buttons=REP_BTNS)

def _clear_report_state():
    for k in ("STATE_report_mode", "STATE_report_period", "STATE_report_draft"):
        set_config(k, "")

def _lat1(s):
    """Make text safe for fpdf's core (latin-1) font."""
    for a, b in {"—": "-", "–": "-", "•": "-", "’": "'", "‘": "'", "“": '"', "”": '"',
                 "…": "...", "₹": "Rs.", "→": "->", "·": "-"}.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "ignore").decode("latin-1")

def build_pdf(title, html_text):
    from fpdf import FPDF
    text = _lat1(_plain(html_text))
    pdf = FPDF(); pdf.add_page(); pdf.set_margins(15, 15, 15)
    def cell(txt, size, bold=False, h=6):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.multi_cell(0, h, txt if txt.strip() else " ", new_x="LMARGIN", new_y="NEXT")
    cell(_lat1("Moustache — India Hostels Pvt Ltd"), 15, True, 9)
    cell(_lat1(title), 12, True, 8)
    pdf.ln(2)
    for line in text.split("\n"):
        cell(line, 11, False, 6)
    path = f"/tmp/report_{int(time.time())}.pdf"; pdf.output(path); return path

def email_report(subject, html_body, pdf_path, to_list, cc=None):
    user, pw = os.environ.get("SMTP_USER", ""), os.environ.get("SMTP_PASS", "")
    if not (user and pw):
        return False, "SMTP_USER / SMTP_PASS secrets not set"
    if not to_list:
        return False, "No recipient — set DIRECTOR_EMAIL in PA_Config"
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"] = subject; msg["From"] = user; msg["To"] = ", ".join(to_list)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg.set_content(_plain(html_body))
    msg.add_alternative(f"<html><body>{html_body}</body></html>", subtype="html")
    try:
        with open(pdf_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf",
                               filename=pdf_path.rsplit("/", 1)[-1])
    except Exception:
        pass
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls(); s.login(user, pw); s.send_message(msg)
        return True, "ok"
    except Exception as e:
        return False, str(e)

def send_report():
    c = config()
    draft = c.get("STATE_report_draft", "")
    kind = c.get("STATE_report_mode", "monthly") or "monthly"
    period = c.get("STATE_report_period", "")
    if not draft:
        return send("There's no report draft to send. Say “generate report” to start one.")
    director = c.get("DIRECTOR_EMAIL", "").strip()
    me = os.environ.get("SMTP_USER", "").strip()
    to_list = [director] if director else []
    subject = f"{kind.title()} Report — {c.get('OWNER_NAME','Shivam Negi')} — {period}"
    pdf = build_pdf(subject, draft)
    ok, err = email_report(subject, draft, pdf, to_list, cc=[me] if me else None)
    try:
        ws("PA_Reports").append_row([f"R{int(time.time())%100000}", kind.title(), period,
            now_ist().strftime("%Y-%m-%d"), "Sent" if ok else "Failed", _plain(draft)[:400], ""])
    except Exception:
        pass
    if ok:
        _clear_report_state()
        send(f"📤 Sent the {kind} report to <b>{director or 'nobody set'}</b> "
             f"(copy to you). Archived in PA_Reports. ✅")
    else:
        send(f"⚠️ Couldn't email it: <code>{html.escape(err)}</code>\nDraft kept — fix and try again. "
             "(Check SMTP_USER/SMTP_PASS secrets and DIRECTOR_EMAIL in PA_Config.)")

# ----------------------------------------------------------------------------- COMMAND + BUTTON HANDLERS
def add_task(text):
    prio = "P2"
    m = re.search(r"\b(P[123])\b", text, re.I)
    if m:
        prio = m.group(1).upper(); text = text.replace(m.group(0), "")
    due = ""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        due = m.group(1); text = text.replace(due, "")
    task = text.strip(" ,.-")
    tid = f"T{int(time.time())%100000}"
    ws("PA_Tasks").append_row([tid, task, "Telegram", prio, due, "Open", "",
                               now_ist().strftime("%Y-%m-%d"), "", ""])
    log_event("task_added", task)
    return tid, task, prio, due

def mark_done(task_id):
    w = ws("PA_Tasks")
    cells = w.findall(str(task_id))
    if not cells:
        return None
    row = cells[0].row
    headers = w.row_values(1)
    scol = headers.index("Status") + 1
    ccol = headers.index("Completed") + 1
    task = w.cell(row, headers.index("Task") + 1).value
    w.update_cell(row, scol, "Done")
    w.update_cell(row, ccol, now_ist().strftime("%Y-%m-%d"))
    log_event("task_done", task or task_id)
    return task

# ----------------------------------------------------------------------------- NATURAL LANGUAGE BRAIN
# Turns plain messages into the right action — no command syntax, no Task IDs needed.
def understand(text):
    tasks = open_tasks()
    tlist = "\n".join(
        f"{r.get('Task ID')}: {r.get('Task')} [{r.get('Priority')}] due {r.get('Due Date')}"
        for r in tasks) or "(no open tasks)"
    today = now_ist().strftime("%Y-%m-%d (%A)")
    sysp = (
        "You are Shivam's PA. Convert his message into ONE JSON action. Intents:\n"
        '  {"intent":"add_task","task":"...","priority":"P1|P2|P3","due":"YYYY-MM-DD or empty"}\n'
        '  {"intent":"complete_task","task_id":"<exact id from OPEN TASKS he means, or empty>"}\n'
        '  {"intent":"list_tasks"}\n'
        '  {"intent":"reflect"}   {"intent":"report"}\n'
        '  {"intent":"ask","query":"..."}   (questions, research, anything needing thought)\n'
        '  {"intent":"chat","reply":"short friendly reply"}   (small talk / acknowledgements)\n'
        "Rules: if he says he finished/closed/did something, match it to the OPEN TASK he means "
        "and use complete_task with that Task ID. If he describes a new to-do, use add_task and "
        "infer priority + resolve relative dates (today/tomorrow/Friday) to YYYY-MM-DD. "
        "Default priority P2. Return ONLY the JSON, nothing else."
    )
    out, _ = ai(system=sysp,
                prompt=f"Today is {today}.\nOPEN TASKS:\n{tlist}\n\nMessage: {text}",
                purpose="quick")
    try:
        return json.loads(re.search(r"\{.*\}", out, re.DOTALL).group())
    except Exception:
        return {"intent": "ask", "query": text}

def fuzzy_complete(text):
    """Fallback if the model didn't return an id: match by shared words."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    best, score = None, 0
    for r in open_tasks():
        tw = set(re.findall(r"[a-z0-9]+", str(r.get("Task", "")).lower()))
        s = len(words & tw)
        if s > score:
            score, best = s, r
    return mark_done(best.get("Task ID")) if best and score >= 1 else None

def route_natural(text):
    a = understand(text)
    it = a.get("intent", "ask")
    if it == "add_task":
        task = (a.get("task") or "").strip()
        if not task:
            return send("What would you like me to add?")
        prio = (a.get("priority") or "P2").upper()
        due = a.get("due") or ""
        tid = f"T{int(time.time())%100000}"
        ws("PA_Tasks").append_row([tid, task, "Telegram", prio, due, "Open", "",
                                   now_ist().strftime("%Y-%m-%d"), "", ""])
        log_event("task_added", task)
        send(f"➕ Got it — added <b>{html.escape(task)}</b> [{prio}]"
             + (f" · due {due}" if due else "") + ".",
             buttons=[[btn("✅ Mark done", "done:" + tid), btn("📋 Today", "cmd:today")]])
    elif it == "complete_task":
        t = mark_done(a.get("task_id", "")) if a.get("task_id") else None
        if not t:
            t = fuzzy_complete(text)
        send(f"✅ Nice — marked <b>{html.escape(t)}</b> done." if t
             else "I couldn't tell which task you meant. Which one? (say a few words from it, or /today to see the list)")
    elif it == "list_tasks":
        job_morning()
    elif it == "reflect":
        job_reflection()
    elif it == "report":
        job_report()
    elif it == "chat":
        send(a.get("reply") or "👍")
    else:  # ask / research / anything needing depth
        send("🔎 <i>Thinking…</i>")
        send(run_agent(a.get("query") or text))

def handle_command(text, msg):
    cmd = text.split()[0].lower().lstrip("/")
    arg = text[len(cmd)+1:].strip() if " " in text else ""
    if cmd in ("start", "help"):
        send("🤖 <b>I'm your work PA — just talk to me naturally.</b>\n"
             "Say things like:\n"
             "• <i>“closed the Jaipur audit point”</i> → I mark it done\n"
             "• <i>“remind me to call the GM Friday”</i> → I add it with the date\n"
             "• <i>“what's on my plate?”</i> → your ranked Top-5\n"
             "• <i>“how should I audit housekeeping?”</i> → I think + research it\n\n"
             "Shortcuts if you prefer: /today · /add · /done · /reflect · /report · /ask · /research")
    elif cmd == "today":
        job_morning()
    elif cmd == "add":
        if not arg:
            send("Usage: <code>/add Close Jaipur audit point P1 2026-07-31</code>")
        else:
            tid, task, prio, due = add_task(arg)
            send(f"➕ Added <b>{html.escape(task)}</b> [{prio}]"
                 + (f" · due {due}" if due else "") + f"\nID <code>{tid}</code>")
    elif cmd == "done":
        t = mark_done(arg)
        send(f"✅ Done: <b>{html.escape(t)}</b>" if t else "Couldn't find that Task ID.")
    elif cmd == "reflect":
        job_reflection()
    elif cmd == "report":
        job_report()
    elif cmd in ("ask", "research"):
        if not arg:
            send("Ask me anything, e.g. <code>/ask what are 2026 hostel check-in best practices?</code>")
        else:
            send("🔎 <i>Thinking &amp; researching…</i>")
            send(run_agent(arg) if cmd == "ask" else research(arg))
    else:
        route_natural(text)

def handle_callback(cb):
    data = cb.get("data", "")
    chat = cb["message"]["chat"]["id"]
    mid = cb["message"]["message_id"]
    if data.startswith("done:"):
        t = mark_done(data.split(":", 1)[1])
        answer_cb(cb["id"], "Marked done ✅")
        if t:
            edit(chat, mid, cb["message"].get("text", "") + f"\n\n✅ <b>{html.escape(t)}</b> — done!")
    elif data == "cmd:today":
        answer_cb(cb["id"]); job_morning()
    elif data == "cmd:addhelp":
        answer_cb(cb["id"]); send("Add a task: <code>/add Close audit point P1 2026-07-31</code>")
    elif data == "ping:hi":
        answer_cb(cb["id"], "👋"); send("👋 Hello Shivam! Buttons work. We're in business.")
    elif data == "reflect:skip":
        set_config("STATE_awaiting_reflection", "0")
        answer_cb(cb["id"], "Skipped"); edit(chat, mid, "🌙 No worries — skipped tonight. See you at 7am.")
    elif data.startswith("rep:"):
        act = data.split(":", 1)[1]
        answer_cb(cb["id"], act)
        if act == "send":
            send_report()
        elif act == "regen":
            job_report(config().get("STATE_report_mode", "monthly") or "monthly")
        elif act == "cancel":
            _clear_report_state(); send("❌ Report cancelled — nothing sent.")
        else:
            send("✏️ Tell me the changes and I'll revise the draft.")
    else:
        answer_cb(cb["id"])

# ----------------------------------------------------------------------------- LISTENER LOOP (near real-time)
def job_listen(minutes=300):
    """Long-poll Telegram for commands/buttons. Runs ~5h then exits (workflow restarts it)."""
    c = config()
    offset = int(c.get("STATE_tg_offset", 0)) or None
    end = time.time() + minutes * 60
    send("🟢 PA listener online — commands and buttons are live.")
    while time.time() < end:
        try:
            res = tg("getUpdates", offset=offset, timeout=25).get("result", [])
        except Exception:
            time.sleep(5); continue
        for u in res:
            offset = u["update_id"] + 1
            set_config("STATE_tg_offset", offset)
            try:
                if "message" in u and u["message"].get("text"):
                    m = u["message"]
                    if str(m["chat"]["id"]) != str(TG_CHAT):
                        continue
                    txt = m["text"]
                    if txt.startswith("/"):
                        handle_command(txt, m)
                    else:
                        st = config()
                        if st.get("STATE_awaiting_reflection") == "1":
                            job_collect()
                        elif st.get("STATE_report_mode"):
                            low = txt.strip().lower()
                            if low in ("send", "send it", "approve", "approve & send", "ok send", "yes send"):
                                send_report()
                            elif low in ("cancel", "stop", "discard", "no"):
                                _clear_report_state(); send("❌ Report cancelled — nothing sent.")
                            else:
                                revise_report(txt)
                        else:
                            route_natural(txt)
                elif "callback_query" in u:
                    handle_callback(u["callback_query"])
            except Exception as e:
                send(f"⚠️ Hit an error: <code>{html.escape(str(e))}</code>")

# ----------------------------------------------------------------------------- MAIN
def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "ping"
    fn = {"ping": job_ping, "morning": job_morning, "reflection": job_reflection,
          "collect": job_collect, "daily": job_daily, "weekly": job_weekly,
          "monthly": job_monthly, "report": job_report, "listen": job_listen}.get(job)
    if not fn:
        print("Unknown job:", job); sys.exit(1)
    try:
        fn()
    except Exception:
        err = traceback.format_exc()
        print(err)
        try:
            send(f"⚠️ Job <b>{job}</b> failed:\n<code>{html.escape(err[-600:])}</code>")
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
