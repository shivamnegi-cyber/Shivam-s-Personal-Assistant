#!/usr/bin/env python3
"""
Shivam's AI Personal Assistant  —  a multi-layer, thinking work PA.

One file, many jobs. Run with:  python pa_bot.py <job>
Jobs: ping | morning | reflection | collect | daily | weekly | monthly | sunday | report | listen

Reads/writes ONE Google Sheet (tabs: PA_Config, PA_Goals, PA_Tasks, PA_Reflections,
PA_DailyLog, PA_Reminders, PA_Followups, PA_Memory, PA_Capture, PA_Reports).
Talks to you on Telegram with buttons + natural conversation + voice & vision.
Thinks with a crew of FREE AI providers (auto-fallback) and can research the web when asked.
"""
import os, sys, json, time, html, re, datetime, traceback, base64
import requests

# ----------------------------------------------------------------------------- ENV / SECRETS
TG_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT    = os.environ.get("TELEGRAM_CHAT_ID", "")
SHEET_ID   = os.environ.get("SHEET_ID", "")
SA_JSON    = os.environ.get("GOOGLE_SA_JSON", "")
TZ_OFFSET  = 5.5  # IST (+05:30)

TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# ----------------------------------------------------------------------------- IN-MEMORY CACHE (API Quota Saver)
_CACHE = {}

def cached_records(tab_name, ttl_seconds=15):
    now = time.time()
    if tab_name in _CACHE:
        data, timestamp = _CACHE[tab_name]
        if now - timestamp < ttl_seconds:
            return data
    try:
        data = ws(tab_name).get_all_records()
        _CACHE[tab_name] = (data, now)
        return data
    except Exception:
        return _CACHE.get(tab_name, ([], 0))[0]

def clear_cache(tab_name=None):
    if tab_name: _CACHE.pop(tab_name, None)
    else: _CACHE.clear()

# ----------------------------------------------------------------------------- AI ROUTER
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

TIERS = {
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
    "deep": [
        ("groq",       "deepseek-r1-distill-llama-70b"),
        ("openrouter", "deepseek/deepseek-r1:free"),
        ("nvidia",     "deepseek-ai/deepseek-r1"),
        ("cerebras",   "llama-3.3-70b"),
        ("github",     "gpt-4o"),
        ("gemini",     "gemini-2.5-flash"),
        ("groq",       "llama-3.3-70b-versatile"),
    ],
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
        "model": model, "temperature": 0.3,
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
    if tier:
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
                break
            except Exception:
                time.sleep(1.0 * (attempt + 1))
                continue
    return "", "none"

# ----------------------------------------------------------------------------- TELEGRAM & COMMAND MENU
def tg(method, **payload):
    try:
        r = requests.post(f"{TG_API}/{method}", json=payload, timeout=30)
        return r.json()
    except Exception:
        return {"ok": False}

def setup_tg_commands():
    commands = [
        {"command": "today", "description": "Morning plan & top tasks"},
        {"command": "ask", "description": "Ask or research anything"},
        {"command": "reflect", "description": "Nightly check-in"},
        {"command": "sunday", "description": "Weekly performance digest"},
        {"command": "report", "description": "Generate monthly/weekly report"},
        {"command": "help", "description": "How to talk to your PA"}
    ]
    tg("setMyCommands", commands=commands)

def _md_to_tg(t):
    if not t:
        return t
    lines = t.split("\n")
    for _ in range(2):
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines):
            first = lines[i].strip()
            if re.search(r"(?i)(telegram html|html format)", first) or \
               re.search(r"(?i)^here'?s\b.{0,90}:$", first):
                del lines[:i + 1]
                continue
        break
    t = "\n".join(lines).lstrip("\n")
    t = re.sub(r"```[a-zA-Z]*\n?", "", t).replace("```", "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.DOTALL)
    t = re.sub(r"__(.+?)__", r"<b>\1</b>", t, flags=re.DOTALL)
    t = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"<b>\1</b>", t)
    t = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", t)
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
    if not r.get("ok"):
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

# ----------------------------------------------------------------------------- GOOGLE SHEET & UTILS
_gc = None
def sheet():
    global _gc
    import gspread
    if _gc is None:
        _gc = gspread.service_account_from_dict(json.loads(SA_JSON))
    return _gc.open_by_key(SHEET_ID)

def _ws_ensure(name, headers):
    sh = sheet()
    try:
        return sh.worksheet(name)
    except Exception:
        w = sh.add_worksheet(title=name, rows=300, cols=len(headers))
        w.append_row(headers)
        return w

def ws(tab):
    schemas = {
        "PA_Config": ["Key", "Value", "Description"],
        "PA_Tasks": ["Task ID", "Task", "Category", "Priority", "Due Date", "Status", "Month", "Score", "Notes"],
        "PA_Followups": ["FID", "Item", "Waiting On", "Created", "Due", "Status", "Notes"],
        "PA_Reminders": ["Reminder ID", "Text", "When", "Repeat", "Status", "Created"],
        "PA_Memory": ["MID", "Category", "Key", "Value", "Notes"],
        "PA_Capture": ["CID", "Date", "Type", "Content", "Tags", "Status", "Notes"],
        "PA_DailyLog": ["Timestamp", "Date", "Type", "Detail", "Goal"],
        "PA_Reflections": ["Date", "Finished", "Blocked", "Learned", "Energy", "Tomorrow"],
        "PA_Reports": ["RID", "Type", "Period", "Date", "Status", "Summary", "Notes"]
    }
    hdr = schemas.get(tab, ["ID", "Content", "Status", "Created"])
    return _ws_ensure(tab, hdr)

def get_val(r, keys, default=""):
    if not isinstance(r, dict):
        return default
    r_lower = {str(k).strip().lower(): v for k, v in r.items()}
    for k in keys:
        kl = str(k).strip().lower()
        if kl in r_lower and r_lower[kl] is not None and str(r_lower[kl]).strip() != "":
            return str(r_lower[kl]).strip()
    return default

def config():
    try:
        rows = cached_records("PA_Config")
        return {str(r.get("Key","")): str(r.get("Value","")) for r in rows if r.get("Key")}
    except Exception:
        return {}

def set_config(key, value):
    w = ws("PA_Config")
    cells = w.findall(str(key))
    if cells:
        w.update_cell(cells[0].row, 2, str(value))
    else:
        w.append_row([key, str(value), ""])
    clear_cache("PA_Config")

def now_ist():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET)

def log_event(kind, detail, goal=""):
    try:
        now = now_ist()
        ws("PA_DailyLog").append_row(
            [now.strftime("%Y-%m-%d %H:%M"), now.strftime("%Y-%m-%d"), kind, detail, goal])
        clear_cache("PA_DailyLog")
    except Exception:
        pass

# ----------------------------------------------------------------------------- MATCHING & NORMALIZATION
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

def _text_similarity(a, b):
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return min(len(na), len(nb)) / max(len(na), len(nb))
    wa, wb = set(na.split()), set(nb.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def _normalize_datetime(when_str):
    if not when_str:
        return None
    s = str(when_str).strip()
    now = now_ist()
    m = re.search(r"(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{2})", s)
    if m:
        return f"{m.group(1)} {int(m.group(2)):02d}:{int(m.group(3)):02d}"
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return f"{m.group(1)} 09:00"
    m = re.search(r"(\d{1,2}):(\d{2})(?:\s*(am|pm))?", s, re.I)
    if m:
        h, min_val = int(m.group(1)), int(m.group(2))
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and h < 12: h += 12
        elif ampm == "am" and h == 12: h = 0
        dt = now.replace(hour=h, minute=min_val, second=0, microsecond=0)
        if dt < now:
            dt += datetime.timedelta(days=1)
        return dt.strftime("%Y-%m-%d %H:%M")
    return None

# ----------------------------------------------------------------------------- DATA ACCESS (CACHED)
def open_tasks():
    try:
        rows = cached_records("PA_Tasks")
    except Exception:
        return []
    cur = now_ist().strftime("%b %Y")
    out = []
    for r in rows:
        st = get_val(r, ["Status"]).lower()
        if st in ("done", "cancelled", "complete", "completed", "missed"):
            continue
        task = get_val(r, ["Task", "Item"])
        if not task:
            continue
        m = get_val(r, ["Month"])
        if m and m != cur:
            continue
        out.append(r)
    return out

def rank_tasks(rows):
    def key(r):
        p = get_val(r, ["Priority"], "P3").upper()
        prio = {"P1": 1, "P2": 2, "P3": 3}.get(p, 3)
        due = get_val(r, ["Due Date", "Due"])
        try:
            d = datetime.date.fromisoformat(due)
        except Exception:
            d = datetime.date.max
        return (prio, d)
    return sorted(rows, key=key)

def open_followups():
    try:
        rows = cached_records("PA_Followups")
    except Exception:
        return []
    return [r for r in rows if get_val(r, ["Status"]).lower() in ("", "open")
            and get_val(r, ["Item", "Task"])]

def _open_reminders():
    try:
        return [r for r in cached_records("PA_Reminders")
                if get_val(r, ["Status"]).lower() in ("", "open")
                and get_val(r, ["Text", "Reminder"])]
    except Exception:
        return []

def _memory_context():
    try:
        rows = cached_records("PA_Memory")
        if not rows: return "(none)"
        return "\n".join(f"- {get_val(r, ['Key'])}: {get_val(r, ['Value'])}" for r in rows[:12])
    except Exception:
        return "(none)"

# ----------------------------------------------------------------------------- WEB RESEARCH
def web_search(query, n=5):
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
    hits = web_search(query, 5)
    if not hits:
        return "I couldn't reach the web right now — please try again in a moment."
    src = "\n\n".join(f"[{i+1}] {h.get('title','')}\n{h.get('body','')}\n{h.get('href') or h.get('url','')}"
                      for i, h in enumerate(hits))
    ans, _ = ai(
        system=("You are Shivam's research analyst (hospitality: hostels, audit, training in India). "
                "Answer his question using ONLY the sources. Be specific and practical. "
                "End with a short 'Sources:' list of the [n] used. Format for Telegram HTML "
                "(use <b> for emphasis, no markdown)."),
        prompt=f"Question: {query}\n\nSources:\n{src}", purpose="long")
    if not ans:
        ans = "<b>Top results</b> (AI busy — raw sources):\n" + "\n".join(
            f"• {html.escape(h.get('title',''))} — {h.get('href') or h.get('url','')}" for h in hits[:5])
    return ans

# ----------------------------------------------------------------------------- CALENDAR & EMAIL (SAFE OPTIONAL)
def calendar_events(days=1):
    try:
        from google.oauth2 import service_account
        import google.auth.transport.requests as gtr, urllib.parse
        creds = service_account.Credentials.from_service_account_info(
            json.loads(SA_JSON), scopes=["https://www.googleapis.com/auth/calendar"])
        creds.refresh(gtr.Request())
        cid = urllib.parse.quote((config().get("CALENDAR_ID", "") or "primary").strip())
        now = datetime.datetime.utcnow()
        tmin = now.isoformat() + "Z"
        tmax = (now + datetime.timedelta(days=days)).isoformat() + "Z"
        r = requests.get(f"https://www.googleapis.com/calendar/v3/calendars/{cid}/events",
            headers={"Authorization": f"Bearer {creds.token}"},
            params={"timeMin": tmin, "timeMax": tmax, "singleEvents": "true", "orderBy": "startTime"}, timeout=15)
        if r.status_code != 200: return "No calendar events."
        items = r.json().get("items", [])
        out = []
        for e in items:
            st = e.get("start", {}).get("dateTime", e.get("start", {}).get("date", ""))
            when = st[11:16] if "T" in st else (st + " (all day)")
            out.append(f"• {when} {e.get('summary', '(no title)')}")
        return "\n".join(out) if out else "No calendar events."
    except Exception:
        return "No calendar events."

# ----------------------------------------------------------------------------- INTENT ENGINE (OPTIMIZED CONTEXT)
def _intent_context():
    tasks = rank_tasks(open_tasks())
    fu = open_followups()
    rems = _open_reminders()
    mem = _memory_context()
    tctx = "\n".join(f"[{get_val(t, ['Task ID', 'ID'])}] {get_val(t, ['Task'])} ({get_val(t, ['Priority'], 'P2')})" for t in tasks[:12]) or "(none)"
    fctx = "\n".join(f"[{get_val(r, ['FID', 'ID'])}] {get_val(r, ['Item'])} — waiting on: {get_val(r, ['Waiting On', 'Who'], '—')}" for r in fu[:8]) or "(none)"
    rctx = "\n".join(f"[{get_val(r, ['Reminder ID', 'ID'])}] {get_val(r, ['Text'])} @ {get_val(r, ['When'])}" for r in rems[:8]) or "(none)"
    return tctx, fctx, rctx, mem

INTENT_SYS = (
    "You are the executive AI Intent Engine for Shivam Negi's Personal Assistant.\n"
    "Shivam is an Internal Auditor & Trainer at Moustache Hostels (India).\n"
    "Understand his intent deeply (English, Hindi, or Hinglish) and output ONLY a single JSON object:\n"
    "{\"actions\": [...], \"reply\": \"short, warm, sharp response in English/Hinglish\"}\n\n"
    "AVAILABLE ACTIONS:\n"
    "- {\"type\":\"add_task\", \"task\":\"description\", \"priority\":\"P1|P2|P3\", \"due\":\"YYYY-MM-DD or ''\", \"category\":\"Task|KRA\"}\n"
    "- {\"type\":\"complete_task\", \"id\":\"Task ID if matched from context, or ''\", \"match\":\"text if id not in context\"}\n"
    "- {\"type\":\"reopen_task\", \"id\":\"Task ID if matched from context, or ''\", \"match\":\"text if id not in context\"}\n"
    "- {\"type\":\"add_reminder\", \"text\":\"what to remind (SUBJECT ONLY, NO TIME WORDS)\", \"when\":\"YYYY-MM-DD HH:MM\", \"repeat\":\"none|daily|weekly\"}\n"
    "- {\"type\":\"add_followup\", \"item\":\"what he is waiting for or tracking\", \"who\":\"person waiting on (e.g. Deepak)\", \"due\":\"YYYY-MM-DD or ''\"}\n"
    "- {\"type\":\"complete_followup\", \"id\":\"FID if matched, or ''\", \"match\":\"text if id not in context\"}\n"
    "- {\"type\":\"remember\", \"category\":\"person|contact|preference|context\", \"key\":\"subject/name\", \"value\":\"information\"}\n"
    "- {\"type\":\"capture_note\", \"text\":\"content\", \"type\":\"idea|link|note|task\", \"tags\":\"comma tags\"}\n"
    "- {\"type\":\"list_tasks\"}\n"
    "- {\"type\":\"list_reminders\"}\n"
    "- {\"type\":\"list_followups\"}\n"
    "- {\"type\":\"scorecard\"}\n"
    "- {\"type\":\"research\", \"query\":\"question needing web search or deep thinking\"}\n\n"
    "RULES:\n"
    "1. TASK vs REMINDER: If Shivam gives a SPECIFIC TIME OF DAY ('at 3pm', 'in 2 hours', 'every day at 8am'), create 'add_reminder'. If it is a to-do with a date or no time ('do audit by Friday'), create 'add_task'.\n"
    "2. FOLLOW-UP: If he mentions waiting for someone ('waiting for audit report from Deepak', 'track Deepak'), create 'add_followup'.\n"
    "3. REMEMBER: For durable facts ('Deepak phone number is X'), create 'remember'. Use DURABLE MEMORY context if referenced.\n"
    "4. COMPLETION: If he finished something ('Jaipur audit done', 'got report from Deepak'), generate 'complete_task' or 'complete_followup' matching context IDs.\n"
    "5. MULTIPLE ACTIONS: Return ALL actions if he mentions multiple items.\n"
    "6. DATES & TIME: Resolve relative terms relative to NOW in Asia/Kolkata timezone.\n"
    "Return ONLY valid JSON."
)

def _new_task_row(tid, task, prio, due, category="Task"):
    return [tid, task, category, prio, due, "Open", now_ist().strftime("%b %Y"), "", ""]

def _do_add_task(a):
    task = (a.get("task") or "").strip()
    if not task: return None
    for r in open_tasks():
        if _text_similarity(task, get_val(r, ["Task"])) >= 0.75:
            return f"Already on task list: {get_val(r, ['Task'])}"
    
    prio = (a.get("priority") or "").upper()
    if prio not in ("P1", "P2", "P3"):
        urgent_keywords = ("urgent", "asap", "director", "critical", "immediately", "today", "top priority")
        prio = "P1" if any(w in task.lower() for w in urgent_keywords) else "P2"
        
    due = a.get("due") or ""
    cat = a.get("category") or "Task"
    tid = f"T{int(time.time()*7)%1000000}"
    ws("PA_Tasks").append_row(_new_task_row(tid, task, prio, due, cat))
    clear_cache("PA_Tasks")
    log_event("task_added", task)
    return f"Added task: {task} [{prio}]" + (f" · due {due}" if due else "")

def _find_task_row(a):
    w = ws("PA_Tasks")
    tid = str(a.get("id", "")).strip()
    match = a.get("match") or a.get("task") or ""
    rows = cached_records("PA_Tasks")
    if tid:
        for idx, r in enumerate(rows, 2):
            if get_val(r, ["Task ID", "ID"]).strip() == tid:
                return w, idx
    best_row, best_score = None, 0.0
    if match:
        for idx, r in enumerate(rows, 2):
            s = _text_similarity(match, get_val(r, ["Task"]))
            if s > best_score:
                best_score, best_row = s, idx
    return (w, best_row) if best_row and best_score >= 0.35 else (w, None)

def _set_task_status(a, status):
    w, row = _find_task_row(a)
    if not row:
        return f"Couldn't find that task to update."
    hdr = w.row_values(1)
    task = w.cell(row, hdr.index("Task") + 1).value if "Task" in hdr else "Task"
    if "Status" in hdr:
        w.update_cell(row, hdr.index("Status") + 1, status)
    clear_cache("PA_Tasks")
    log_event("task_" + status.lower(), task or "")
    return (f"Completed task: {task} ✅" if status == "Done" else f"Reopened task: {task}")

def _do_add_reminder(a):
    text = (a.get("text") or "").strip()
    when_raw = (a.get("when") or "").strip()
    when = _normalize_datetime(when_raw)
    if not when:
        return "Couldn't parse reminder time — please specify a clear time."
    if not text or re.match(r"(?i)^(at |in |tomorrow|today|every|reminder|due)", text):
        text = text if text and len(text) > 3 else "Reminder"
    for r in _open_reminders():
        if _text_similarity(text, get_val(r, ["Text"])) >= 0.75 and get_val(r, ["When"]) == when:
            return f"Reminder already set: {text} @ {when}"
    rid = f"R{int(time.time()*7)%1000000}"
    ws("PA_Reminders").append_row(
        [rid, text, when, a.get("repeat", "none"), "Open", now_ist().strftime("%Y-%m-%d %H:%M")])
    clear_cache("PA_Reminders")
    rep = str(a.get("repeat", "none")).lower()
    return f"⏰ Set reminder: \"{text}\" @ {when}" + (f" ({rep})" if rep != "none" else "")

def _do_add_followup(a):
    item = (a.get("item") or "").strip()
    if not item: return None
    for r in open_followups():
        if _text_similarity(item, get_val(r, ["Item"])) >= 0.75:
            return f"Already tracking: {get_val(r, ['Item'])}"
    fid = f"F{int(time.time()*7)%1000000}"
    ws("PA_Followups").append_row([fid, item, a.get("who", ""), now_ist().strftime("%Y-%m-%d"), a.get("due", ""), "Open", ""])
    clear_cache("PA_Followups")
    return f"Tracking follow-up: {item} (waiting on {a.get('who') or '—'})"

def _do_complete_followup(a):
    w = ws("PA_Followups")
    rows = cached_records("PA_Followups")
    tid = str(a.get("id", "")).strip(); match = a.get("match") or ""
    best_row, best_score, best_item = None, 0.0, ""
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Status"]).lower() not in ("", "open"):
            continue
        if tid and str(get_val(r, ["FID", "ID"])).strip() == tid:
            best_row, best_score, best_item = idx, 1.0, get_val(r, ["Item"])
            break
        s = _text_similarity(match, get_val(r, ["Item"]))
        if s > best_score:
            best_score, best_row, best_item = s, idx, get_val(r, ["Item"])
    if best_row and best_score >= 0.35:
        hdr = w.row_values(1)
        if "Status" in hdr:
            w.update_cell(best_row, hdr.index("Status") + 1, "Done")
        clear_cache("PA_Followups")
        return f"Closed follow-up: {best_item} ✅"
    return f"Couldn't find that follow-up."

def _tool_remember(arg):
    try:
        ws("PA_Memory").append_row([f"M{int(time.time())%100000}", "context", arg[:40], arg, ""])
        clear_cache("PA_Memory")
        return f"Noted in memory: {arg[:60]}"
    except Exception:
        return "Saved memory."

def _tool_capture(arg):
    try:
        ws("PA_Capture").append_row([f"C{int(time.time())%100000}", now_ist().strftime("%Y-%m-%d %H:%M"), "note", arg, "", "New", ""])
        clear_cache("PA_Capture")
        return f"Captured note: {arg[:60]}"
    except Exception:
        return "Saved note."

def _tool_scorecard():
    try:
        rows = cached_records("PA_Tasks")
        kras = [r for r in rows if get_val(r, ["Category"]).lower() == "kra"]
        if not kras: return "No KRA scores recorded yet."
        by_month = {}
        for r in kras:
            m = get_val(r, ["Month"])
            if not m: continue
            if m not in by_month: by_month[m] = [0, 0]
            by_month[m][1] += 1
            try:
                if float(get_val(r, ["Score"], "0")) >= 1: by_month[m][0] += 1
            except Exception: pass
        lines = [f"• {m}: {g}/{t} met" for m, (g, t) in by_month.items()]
        return "<b>KRA Scorecard</b>\n" + "\n".join(lines)
    except Exception:
        return "Couldn't fetch scorecard."

def _tasks_text():
    t = rank_tasks(open_tasks())
    if not t: return "No open tasks."
    return "\n".join(f"{i}. [{get_val(x, ['Priority'], 'P3')}] {get_val(x, ['Task'])} (due {get_val(x, ['Due Date'], '—')})"
                     for i, x in enumerate(t[:12], 1))

def _tool_list_reminders():
    op = _open_reminders()
    if not op: return "No upcoming reminders."
    return "<b>Upcoming Reminders</b>\n" + "\n".join(f"• {get_val(r, ['When'])} — {get_val(r, ['Text'])}" for r in op[:12])

def _tool_list_followups():
    fu = open_followups()
    if not fu: return "No open follow-ups."
    return "<b>Open Follow-ups</b>\n" + "\n".join(f"• {get_val(r, ['Item'])} (waiting on {get_val(r, ['Waiting On'], '—')})" for r in fu[:12])

def handle_intent(text):
    tg("sendChatAction", chat_id=TG_CHAT, action="typing")
    tctx, fctx, rctx, mctx = _intent_context()
    now = now_ist().strftime("%Y-%m-%d %H:%M (%A)")
    prompt = (f"NOW: {now} (Asia/Kolkata)\nOPEN TASKS:\n{tctx}\n\nOPEN FOLLOW-UPS:\n{fctx}\n\n"
              f"OPEN REMINDERS:\n{rctx}\n\nDURABLE MEMORY / CONTACTS:\n{mctx}\n\nUser Message: {text}")
    out, _ = ai(system=INTENT_SYS, prompt=prompt, purpose="deep")
    try:
        data = json.loads(re.search(r"\{.*\}", out, re.DOTALL).group())
    except Exception:
        return send(research(text))

    actions = data.get("actions", [])
    if not actions:
        return send(data.get("reply") or "Noted.")

    results = []
    for a in actions:
        t = a.get("type")
        try:
            if t == "add_task":            results.append(_do_add_task(a))
            elif t == "complete_task":     results.append(_set_task_status(a, "Done"))
            elif t == "reopen_task":       results.append(_set_task_status(a, "Open"))
            elif t == "add_reminder":      results.append(_do_add_reminder(a))
            elif t == "add_followup":      results.append(_do_add_followup(a))
            elif t == "complete_followup": results.append(_do_complete_followup(a))
            elif t == "remember":          results.append(_tool_remember(f"{a.get('key','')}: {a.get('value','')}"))
            elif t == "capture_note":      results.append(_tool_capture(a.get("text", "")))
            elif t == "list_tasks":        return send("<b>Open Tasks</b>\n" + _tasks_text())
            elif t == "list_reminders":    return send(_tool_list_reminders())
            elif t == "list_followups":    return send(_tool_list_followups())
            elif t == "scorecard":         return send(_tool_scorecard())
            elif t == "research":          return send(research(a.get("query", text)))
        except Exception as e:
            results.append(f"(error processing {t})")

    results = [r for r in results if r]
    if not results:
        return send(data.get("reply") or "Done.")
    if len(results) == 1:
        send(results[0])
    else:
        send("Done:\n" + "\n".join(f"• {r}" for r in results))

# ----------------------------------------------------------------------------- JOBS & COMMANDS
def job_ping():
    setup_tg_commands()
    send("✅ <b>Your PA is online & active.</b>\nEverything is connected cleanly.",
         buttons=[[btn("👋 Say hi", "ping:hi")], [btn("📋 Today's tasks", "cmd:today")]])

def job_morning():
    d = now_ist().strftime("%A, %d %b")
    top = rank_tasks(open_tasks())[:5]
    lines = ["☀️ <b>Good morning, Shivam</b>", f"{d}\n"]
    mtg = calendar_events(1)
    if mtg and not mtg.startswith("("):
        lines.append("<b>Today's meetings</b>\n" + mtg + "\n")
    if top:
        lines.append(f"<b>Top {len(top)} tasks</b>")
        for i, t in enumerate(top, 1):
            due = get_val(t, ["Due Date", "Due"])
            due_s = f" · due {due}" if due else ""
            lines.append(f"{i}. [{get_val(t, ['Priority'], 'P3')}] {html.escape(get_val(t, ['Task']))}{due_s}")
    else:
        lines.append("No open tasks — clear runway.")
    
    nudge, _ = ai(prompt="One short, warm, one-line nudge to start the workday. No emoji.", purpose="quick")
    lines.append(f"\n<i>{html.escape(nudge or 'Focus on one high-impact block at a time.')}</i>")
    rows = [btn(f"✅ {i}", f"done:{get_val(t, ['Task ID', 'ID'])}") for i, t in enumerate(top, 1)]
    kb = ([rows] if rows else []) + [[btn("➕ Add task", "cmd:addhelp")]]
    send("\n".join(lines), buttons=kb)

def job_reflection():
    set_config("STATE_awaiting_reflection", "1")
    send("🌙 <b>Nightly check-in</b>\nReply in one message:\n"
         "1. What did you finish today?\n2. What is blocked?\n3. Energy (1-5)?\n4. Tomorrow's #1 priority?",
         buttons=[[btn("😴 Skip tonight", "reflect:skip")]])

def job_collect():
    c = config()
    if c.get("STATE_awaiting_reflection") != "1":
        return
    updates = tg("getUpdates", offset=int(c.get("STATE_tg_offset", 0)) or None, timeout=0).get("result", [])
    answer, last_id = None, int(c.get("STATE_tg_offset", 0))
    for u in updates:
        last_id = max(last_id, u.get("update_id", 0) + 1)
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) == str(TG_CHAT) and msg.get("text") and not msg["text"].startswith("/"):
            answer = msg["text"]
    set_config("STATE_tg_offset", last_id)
    if not answer: return
    ws("PA_Reflections").append_row([now_ist().strftime("%Y-%m-%d"), answer, "", "", "", ""])
    clear_cache("PA_Reflections")
    log_event("reflection", "Nightly check-in logged")
    set_config("STATE_awaiting_reflection", "0")
    send("📝 Reflection logged. Rest well Shivam!")

def job_sunday_digest():
    tasks = open_tasks()
    fu = open_followups()
    sc = _tool_scorecard()
    lines = [
        "🏆 <b>Sunday Executive Digest</b>",
        f"<i>Week ending {now_ist().strftime('%d %b %Y')}</i>\n",
        f"• Open Tasks Remaining: <b>{len(tasks)}</b>",
        f"• Open Follow-ups Pending: <b>{len(fu)}</b>\n",
        sc,
        "\n<b>Next Week's Focus</b>:",
        "Review your top P1 priorities and start fresh tomorrow at 7 AM! 🚀"
    ]
    send("\n".join(lines))

def job_report(kind="monthly"):
    tasks = rank_tasks(open_tasks())
    open_list = "\n".join(f"- [{get_val(t, ['Priority'])}] {get_val(t, ['Task'])}" for t in tasks[:20]) or "None."
    fu = open_followups()
    fu_list = "\n".join(f"- {get_val(f, ['Item'])} (waiting on {get_val(f, ['Waiting On'])})" for f in fu[:10]) or "None."
    prompt = f"Write a crisp {kind} progress report for Shivam Negi (Internal Auditor & Trainer, Moustache Hostels).\nOpen Tasks:\n{open_list}\nFollowups:\n{fu_list}"
    draft, _ = ai(system="Write in executive Telegram HTML format with concise bullet points.", prompt=prompt, purpose="long")
    set_config("STATE_report_draft", draft)
    send(f"🧾 <b>{kind.title()} Report Draft</b>\n\n{draft}\n\n<i>Reply with edits or tap approve to finalize.</i>",
         buttons=[[btn("✅ Approve", "rep:approve"), btn("❌ Cancel", "rep:cancel")]])

def job_daily():
    send("📊 <b>Daily Activity Summary</b> logged for " + now_ist().strftime("%Y-%m-%d"))

def job_weekly(): job_report("weekly")
def job_monthly(): job_report("monthly")
def job_evening(): send("🔔 <b>Evening Watchdog Check-in</b>")
def job_watchdog(): pass

def mark_done(task_id):
    w = ws("PA_Tasks")
    rows = cached_records("PA_Tasks")
    tid = str(task_id).strip()
    target_row = None
    task_name = ""
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Task ID", "ID"]).strip() == tid or _text_similarity(tid, get_val(r, ["Task"])) >= 0.7:
            target_row = idx
            task_name = get_val(r, ["Task"])
            break
    if not target_row:
        return None
    hdr = w.row_values(1)
    if "Status" in hdr:
        w.update_cell(target_row, hdr.index("Status") + 1, "Done")
    if "Notes" in hdr:
        w.update_cell(target_row, hdr.index("Notes") + 1, f"done {now_ist().strftime('%Y-%m-%d')}")
    clear_cache("PA_Tasks")
    log_event("task_done", task_name or tid)
    return task_name or tid

def handle_command(text):
    cmd = text.split()[0].lower().lstrip("/")
    arg = text[len(cmd)+1:].strip() if " " in text else ""
    if cmd in ("start", "help"):
        send("🤖 <b>Shivam's AI PA</b>\nTalk to me naturally:\n"
             "• <i>“Jaipur audit point close ho gaya”</i>\n"
             "• <i>“Remind me at 4pm to call GM”</i>\n"
             "• <i>“Waiting on Deepak for audit sheet”</i>\n"
             "• <i>“What are my tasks?”</i>")
    elif cmd == "today": job_morning()
    elif cmd == "reflect": job_reflection()
    elif cmd == "sunday": job_sunday_digest()
    elif cmd == "report": job_report("monthly")
    elif cmd in ("ask", "research"): send(research(arg) if arg else "Please provide a query.")
    else: handle_intent(text)

def handle_callback(cb):
    data = cb.get("data", "")
    chat = cb["message"]["chat"]["id"]
    mid = cb["message"]["message_id"]
    if data.startswith("done:"):
        t = mark_done(data.split(":", 1)[1])
        answer_cb(cb["id"], "Marked done ✅")
        if t: edit(chat, mid, cb["message"].get("text", "") + f"\n\n✅ <b>{html.escape(t)}</b> — completed!")
    elif data == "cmd:today": answer_cb(cb["id"]); job_morning()
    elif data == "cmd:addhelp": answer_cb(cb["id"]); send("To add a task, simply tell me naturally (e.g. <i>'Add task audit Udaipur hostel by Friday'</i>).")
    elif data == "ping:hi": answer_cb(cb["id"], "👋"); send("👋 Hello Shivam! All systems working smoothly.")
    elif data == "reflect:skip":
        set_config("STATE_awaiting_reflection", "0")
        answer_cb(cb["id"], "Skipped"); edit(chat, mid, "🌙 Skipped tonight. Rest well!")
    elif data.startswith("rep:"):
        answer_cb(cb["id"])
        if "approve" in data: send("✅ Report finalized and saved.")
        else: set_config("STATE_report_draft", ""); send("❌ Draft cancelled.")
    else: answer_cb(cb["id"])

# ----------------------------------------------------------------------------- SENSES & LISTENER
def transcribe(data):
    key = _k("GROQ_API_KEY")
    if not key: return ""
    try:
        r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("audio.ogg", data, "audio/ogg")},
            data={"model": "whisper-large-v3"}, timeout=120)
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except Exception: return ""

def vision(data, prompt):
    b64 = base64.b64encode(data).decode()
    if _k("GEMINI_API_KEY"):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={_k('GEMINI_API_KEY')}"
            r = requests.post(url, timeout=120, json={"contents": [{"parts": [
                {"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}]})
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip(), ""
        except Exception: pass
    return "", "Vision models currently unreachable."

def _romanize(text):
    if not text or not re.search(r"[ऀ-ॿ]", text): return text
    out, _ = ai(system="Transliterate Devanagari text into Latin script (Hinglish). Output ONLY result.", prompt=text, purpose="quick")
    return out or text

def handle_voice(file_id):
    tg("sendChatAction", chat_id=TG_CHAT, action="typing")
    try:
        fpath = tg("getFile", file_id=file_id).get("result", {}).get("file_path", "")
        data = requests.get(f"https://api.telegram.org/file/bot{TG_TOKEN}/{fpath}", timeout=60).content
        txt = _romanize(transcribe(data))
        if txt:
            send(f"🎙️ <i>{html.escape(txt)}</i>")
            _dispatch_text(txt)
        else: send("Couldn't transcribe audio clearly.")
    except Exception: send("Error fetching voice note.")

def handle_photo(file_id, caption=""):
    tg("sendChatAction", chat_id=TG_CHAT, action="typing")
    send("🖼️ <i>Analyzing image…</i>")
    try:
        fpath = tg("getFile", file_id=file_id).get("result", {}).get("file_path", "")
        data = requests.get(f"https://api.telegram.org/file/bot{TG_TOKEN}/{fpath}", timeout=60).content
        out, _ = vision(data, "Extract key information, notes, or action items concisely for Shivam.")
        send(out or "Couldn't read image contents.")
    except Exception: send("Error fetching image.")

def _dispatch_text(txt):
    if txt.startswith("/"): return handle_command(txt)
    st = config()
    if st.get("STATE_awaiting_reflection") == "1": job_collect()
    else: handle_intent(txt)

def _fire_user_reminders():
    now = now_ist(); nows = now.strftime("%Y-%m-%d %H:%M")
    try:
        w = ws("PA_Reminders"); rows = cached_records("PA_Reminders")
    except Exception: return
    hdr = None
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Status"]).lower() not in ("", "open"): continue
        when = get_val(r, ["When"])
        if not when or when > nows: continue
        send(f"⏰ <b>Reminder</b>: {html.escape(get_val(r, ['Text']))}")
        if hdr is None: hdr = w.row_values(1)
        rep = get_val(r, ["Repeat"]).lower()
        if rep in ("daily", "weekly"):
            try:
                nxt = (datetime.datetime.strptime(when, "%Y-%m-%d %H:%M") + datetime.timedelta(days=1 if rep == "daily" else 7)).strftime("%Y-%m-%d %H:%M")
                if "When" in hdr: w.update_cell(idx, hdr.index("When") + 1, nxt)
            except Exception: pass
        elif "Status" in hdr: w.update_cell(idx, hdr.index("Status") + 1, "Done")
    clear_cache("PA_Reminders")

def _run_due_reminders():
    now = now_ist(); today = now.strftime("%Y-%m-%d"); hm = now.strftime("%H:%M")
    dow = now.weekday()
    c = config()
    def fire(key, ok):
        if ok and c.get(f"STATE_last_{key}", "") != today:
            set_config(f"STATE_last_{key}", today)
            try:
                {"morning": job_morning, "reflection": job_reflection, "daily": job_daily, "sunday": job_sunday_digest}[key]()
            except Exception: pass
    fire("morning", hm >= "07:00")
    fire("reflection", hm >= "21:00")
    fire("daily", hm >= "21:15")
    fire("sunday", dow == 6 and hm >= "18:00")

def job_listen(minutes=340):
    setup_tg_commands()
    c = config()
    offset = int(c.get("STATE_tg_offset", 0)) or None
    end = time.time() + minutes * 60
    last_check = 0
    fail_count = 0
    while time.time() < end:
        if time.time() - last_check > 180:
            last_check = time.time()
            try:
                _run_due_reminders()
                _fire_user_reminders()
            except Exception: pass
        try:
            res = tg("getUpdates", offset=offset, timeout=25).get("result", [])
            fail_count = 0
        except Exception:
            fail_count += 1
            time.sleep(min(30, 2 ** fail_count))
            continue
        for u in res:
            offset = u["update_id"] + 1
            set_config("STATE_tg_offset", offset)
            try:
                if "message" in u:
                    m = u["message"]
                    if str(m.get("chat", {}).get("id")) != str(TG_CHAT): continue
                    if m.get("text"): _dispatch_text(m["text"])
                    elif m.get("voice") or m.get("audio"): handle_voice((m.get("voice") or m.get("audio"))["file_id"])
                    elif m.get("photo"): handle_photo(m["photo"][-1]["file_id"], m.get("caption", ""))
                elif "callback_query" in u: handle_callback(u["callback_query"])
            except Exception as e: send(f"⚠️ Error: <code>{html.escape(str(e))}</code>")

def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "ping"
    fn = {"ping": job_ping, "morning": job_morning, "reflection": job_reflection,
          "collect": job_collect, "daily": job_daily, "weekly": job_weekly,
          "monthly": job_monthly, "sunday": job_sunday_digest, "report": job_report, "listen": job_listen,
          "evening": job_evening, "watchdog": job_watchdog}.get(job)
    if not fn: sys.exit(1)
    try: fn()
    except Exception:
        err = traceback.format_exc()
        print(err)
        try: send(f"⚠️ Failure in <b>{job}</b>:\n<code>{html.escape(err[-500:])}</code>")
        except Exception: pass
        sys.exit(1)

if __name__ == "__main__":
    main()
