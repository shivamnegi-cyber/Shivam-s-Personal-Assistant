#!/usr/bin/env python3
"""
Shivam's AI Personal Assistant — intent-driven, self-healing, always-on.
Run: python pa_bot.py <job>   (jobs: ping|morning|reflection|collect|daily|sunday|report|listen)
One Google Sheet (tabs: PA_Config, PA_Tasks, PA_Reflections, PA_DailyLog, PA_Reminders,
PA_Followups, PA_Memory, PA_Capture, PA_Reports). Telegram + voice + vision. Free AI crew.
Hardened so a single error can NEVER kill the 24/7 listener. Nothing is ever deleted.
"""
import os, sys, json, time, html, re, datetime, traceback, base64
import requests

# ----------------------------------------------------------------------------- ENV
TG_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT   = os.environ.get("TELEGRAM_CHAT_ID", "")
SHEET_ID  = os.environ.get("SHEET_ID", "")
SA_JSON   = os.environ.get("GOOGLE_SA_JSON", "")
TZ_OFFSET = 5.5
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# ----------------------------------------------------------------------------- IN-MEMORY CACHE
_CACHE = {}
def cached_records(tab, ttl=15):
    now = time.time()
    if tab in _CACHE and now - _CACHE[tab][1] < ttl:
        return _CACHE[tab][0]
    try:
        data = ws(tab).get_all_records()
        _CACHE[tab] = (data, now)
        return data
    except Exception:
        return _CACHE.get(tab, ([], 0))[0]
def clear_cache(tab=None):
    if tab: _CACHE.pop(tab, None)
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
    "quick": [("groq", "llama-3.1-8b-instant"), ("cerebras", "llama3.1-8b"),
              ("groq", "llama-3.3-70b-versatile"), ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
              ("github", "gpt-4o-mini"), ("mistral", "mistral-small-latest"),
              ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
              ("sambanova", "Meta-Llama-3.3-70B-Instruct"), ("gemini", "gemini-2.5-flash")],
    "deep":  [("groq", "llama-3.3-70b-versatile"), ("cerebras", "llama-3.3-70b"),
              ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"), ("github", "gpt-4o"),
              ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"), ("gemini", "gemini-2.5-flash"),
              ("groq", "llama-3.1-8b-instant")],
    "long":  [("gemini", "gemini-2.5-flash"), ("cerebras", "llama-3.3-70b"),
              ("mistral", "mistral-large-latest"), ("groq", "llama-3.3-70b-versatile"),
              ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"), ("github", "gpt-4o")],
}
def _k(env): return os.environ.get(env, "")
def _has_key(name):
    return bool(_k("GEMINI_API_KEY")) if name == "gemini" else (name in OAI and bool(_k(OAI[name][1])))
def _post_oai(name, model, system, prompt):
    url, env = OAI[name]
    h = {"Authorization": f"Bearer {_k(env)}", "Content-Type": "application/json"}
    if name == "openrouter":
        h["HTTP-Referer"] = "https://github.com"; h["X-Title"] = "Shivam PA"
    r = requests.post(url, headers=h, timeout=60, json={"model": model, "temperature": 0.3,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
    if r.status_code in (408, 409, 425, 429, 500, 502, 503, 529):
        raise RuntimeError(f"{name} {r.status_code}")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
def _post_gemini(model, system, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={_k('GEMINI_API_KEY')}"
    r = requests.post(url, headers={"Content-Type": "application/json"}, timeout=60, json={
        "systemInstruction": {"parts": [{"text": system}]}, "contents": [{"parts": [{"text": prompt}]}]})
    if r.status_code in (429, 500, 502, 503):
        raise RuntimeError(f"gemini {r.status_code}")
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
def ai(prompt, system="You are Shivam's sharp, concise work assistant.", purpose="quick", tier=None):
    if tier: purpose = {"fast": "quick"}.get(tier, tier)
    for name, model in TIERS.get(purpose, TIERS["quick"]):
        if not _has_key(name):
            continue
        for attempt in range(2):
            try:
                out = _post_gemini(model, system, prompt) if name == "gemini" else _post_oai(name, model, system, prompt)
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
                if out:
                    return out, f"{name}:{model}"
                break
            except Exception:
                time.sleep(1.0 * (attempt + 1)); continue
    return "", "none"

# ----------------------------------------------------------------------------- TELEGRAM
def tg(method, **payload):
    try:
        return requests.post(f"{TG_API}/{method}", json=payload, timeout=30).json()
    except Exception:
        return {"ok": False}
def setup_tg_commands():
    tg("setMyCommands", commands=[
        {"command": "today", "description": "Morning plan & top tasks"},
        {"command": "tasks", "description": "List open tasks"},
        {"command": "ask", "description": "Ask or research anything"},
        {"command": "reflect", "description": "Nightly check-in"},
        {"command": "report", "description": "Generate a report"},
        {"command": "help", "description": "How to talk to your PA"}])
def _md_to_tg(t):
    if not t: return t
    lines = t.split("\n")
    for _ in range(2):
        i = 0
        while i < len(lines) and not lines[i].strip(): i += 1
        if i < len(lines):
            f = lines[i].strip()
            if re.search(r"(?i)(telegram html|html format)", f) or re.search(r"(?i)^here'?s\b.{0,90}:$", f):
                del lines[:i + 1]; continue
        break
    t = "\n".join(lines).lstrip("\n")
    t = re.sub(r"```[a-zA-Z]*\n?", "", t).replace("```", "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.DOTALL)
    t = re.sub(r"__(.+?)__", r"<b>\1</b>", t, flags=re.DOTALL)
    t = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"<b>\1</b>", t)
    t = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", t)
    return t.strip()
def _plain(t): return re.sub(r"<[^>]+>", "", t or "")
def send(text, buttons=None, chat=None):
    text = _md_to_tg(text)
    p = {"chat_id": chat or TG_CHAT, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    if buttons: p["reply_markup"] = {"inline_keyboard": buttons}
    r = tg("sendMessage", **p)
    if not r.get("ok"):
        p["text"] = _plain(text); p.pop("parse_mode", None)
        r = tg("sendMessage", **p)
    return r
def edit(chat, mid, text, buttons=None):
    text = _md_to_tg(text)
    p = {"chat_id": chat, "message_id": mid, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    if buttons: p["reply_markup"] = {"inline_keyboard": buttons}
    r = tg("editMessageText", **p)
    if not r.get("ok"):
        p["text"] = _plain(text); p.pop("parse_mode", None)
        r = tg("editMessageText", **p)
    return r
def answer_cb(cb_id, text=""): return tg("answerCallbackQuery", callback_query_id=cb_id, text=text)
def btn(label, data): return {"text": label, "callback_data": data}

# ----------------------------------------------------------------------------- SHEET + UTIL
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
        w = sh.add_worksheet(title=name, rows=400, cols=max(6, len(headers)))
        w.append_row(headers)
        return w
SCHEMAS = {
    "PA_Config":      ["Key", "Value", "Description"],
    "PA_Tasks":       ["Task ID", "Task", "Category", "Priority", "Due Date", "Status", "Month", "Score", "Notes"],
    "PA_Followups":   ["FID", "Item", "Waiting On", "Created", "Due", "Status", "Notes"],
    "PA_Reminders":   ["Reminder ID", "Text", "When", "Repeat", "Status", "Created"],
    "PA_Memory":      ["MID", "Category", "Key", "Value", "Notes"],
    "PA_Capture":     ["CID", "Date", "Type", "Content", "Tags", "Status", "Notes"],
    "PA_DailyLog":    ["Timestamp", "Date", "Type", "Detail", "Goal"],
    "PA_Reflections": ["Date", "Finished", "Blocked", "Learned", "Energy", "Tomorrow"],
    "PA_Reports":     ["RID", "Type", "Period", "Date", "Status", "Summary", "Notes"],
}
def ws(tab):
    return _ws_ensure(tab, SCHEMAS.get(tab, ["ID", "Content", "Status", "Created"]))
def get_val(r, keys, default=""):
    if not isinstance(r, dict): return default
    low = {str(k).strip().lower(): v for k, v in r.items()}
    for k in keys:
        v = low.get(str(k).strip().lower())
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default
def now_ist(): return datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET)
def config():
    try:
        return {str(r.get("Key", "")): str(r.get("Value", "")) for r in cached_records("PA_Config") if r.get("Key")}
    except Exception:
        return {}
def set_config(key, value):
    """Hardened: never raises (so it can't crash the listener); no gspread.findall."""
    try:
        w = ws("PA_Config")
        vals = w.get_all_values()
        row_idx = None
        for i, r in enumerate(vals, 1):
            if r and str(r[0]).strip() == str(key):
                row_idx = i; break
        if row_idx:
            w.update_cell(row_idx, 2, str(value))
        else:
            w.append_row([str(key), str(value), ""])
        clear_cache("PA_Config")
    except Exception:
        pass
def log_event(kind, detail, goal=""):
    try:
        n = now_ist()
        ws("PA_DailyLog").append_row([n.strftime("%Y-%m-%d %H:%M"), n.strftime("%Y-%m-%d"), kind, detail, goal])
        clear_cache("PA_DailyLog")
    except Exception:
        pass

# ----------------------------------------------------------------------------- MATCH / TIME
def _norm(s): return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
def _sim(a, b):
    na, nb = _norm(a), _norm(b)
    if not na or not nb: return 0.0
    if na == nb: return 1.0
    if na in nb or nb in na: return min(len(na), len(nb)) / max(len(na), len(nb))
    wa, wb = set(na.split()), set(nb.split())
    return len(wa & wb) / len(wa | wb) if (wa and wb) else 0.0
def _normalize_datetime(s):
    if not s: return None
    s = str(s).strip(); now = now_ist()
    m = re.search(r"(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{2})", s)
    if m: return f"{m.group(1)} {int(m.group(2)):02d}:{int(m.group(3)):02d}"
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m: return f"{m.group(1)} 09:00"
    m = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", s, re.I)      # 5:00 pm / 17:00
    if not m: m = re.search(r"(\d{1,2})\s*(am|pm)", s, re.I)      # bare 9am
    if m:
        h = int(m.group(1)); mn = int(m.group(2)) if (m.lastindex and m.group(2) and m.group(2).isdigit()) else 0
        ap = (m.groups()[-1] or "").lower()
        if ap == "pm" and h < 12: h += 12
        elif ap == "am" and h == 12: h = 0
        dt = now.replace(hour=h % 24, minute=mn, second=0, microsecond=0)
        if "tomorrow" in s.lower(): dt += datetime.timedelta(days=1)
        elif dt < now: dt += datetime.timedelta(days=1)
        return dt.strftime("%Y-%m-%d %H:%M")
    return None

# ----------------------------------------------------------------------------- DATA ACCESS
def open_tasks():
    cur = now_ist().strftime("%b %Y")
    out = []
    for r in cached_records("PA_Tasks"):
        if get_val(r, ["Status"]).lower() in ("done", "cancelled", "complete", "completed", "missed"): continue
        if not get_val(r, ["Task", "Item"]): continue
        m = get_val(r, ["Month"])
        if m and m != cur: continue
        out.append(r)
    return out
def rank_tasks(rows):
    def key(r):
        prio = {"P1": 1, "P2": 2, "P3": 3}.get(get_val(r, ["Priority"], "P3").upper(), 3)
        try: d = datetime.date.fromisoformat(get_val(r, ["Due Date", "Due"]))
        except Exception: d = datetime.date.max
        return (prio, d)
    return sorted(rows, key=key)
def open_followups():
    return [r for r in cached_records("PA_Followups")
            if get_val(r, ["Status"]).lower() in ("", "open") and get_val(r, ["Item", "Task"])]
def unique_followups():
    seen, out = set(), []
    for r in open_followups():
        k = _norm(get_val(r, ["Item"]))
        if k and k not in seen: seen.add(k); out.append(r)
    return out
def _open_reminders():
    return [r for r in cached_records("PA_Reminders")
            if get_val(r, ["Status"]).lower() in ("", "open") and get_val(r, ["Text"])]
def open_kras():
    cur = now_ist().strftime("%b %Y"); seen, out = set(), []
    for r in cached_records("PA_Tasks"):
        if get_val(r, ["Category"]).lower() != "kra": continue
        if get_val(r, ["Month"]) != cur: continue
        if get_val(r, ["Score"]).strip(): continue
        t = get_val(r, ["Task"]); k = _norm(t)
        if t and k not in seen: seen.add(k); out.append(t)
    return out
def _memory_context():
    try:
        rows = cached_records("PA_Memory")
        return "\n".join(f"- {get_val(r, ['Key'])}: {get_val(r, ['Value'])}" for r in rows[:15]) or "(none)"
    except Exception:
        return "(none)"

# ----------------------------------------------------------------------------- RESEARCH
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
        return "I couldn't reach the web right now — try again in a moment."
    src = "\n\n".join(f"[{i+1}] {h.get('title','')}\n{h.get('body','')}\n{h.get('href') or h.get('url','')}"
                      for i, h in enumerate(hits))
    ans, _ = ai(system=("You are Shivam's research analyst (hospitality/audit/training in India). Answer using ONLY "
                        "the sources; be specific and practical. End with a short 'Sources:' list. Telegram HTML, English."),
                prompt=f"Question: {query}\n\nSources:\n{src}", purpose="long")
    return ans or ("<b>Top results</b>\n" + "\n".join(f"• {html.escape(h.get('title',''))} — {h.get('href') or h.get('url','')}" for h in hits[:5]))

# ----------------------------------------------------------------------------- CALENDAR (optional read)
def calendar_events(days=1):
    try:
        from google.oauth2 import service_account
        import google.auth.transport.requests as gtr, urllib.parse
        creds = service_account.Credentials.from_service_account_info(
            json.loads(SA_JSON), scopes=["https://www.googleapis.com/auth/calendar"])
        creds.refresh(gtr.Request())
        cid = urllib.parse.quote((config().get("CALENDAR_ID", "") or "primary").strip())
        now = datetime.datetime.utcnow()
        r = requests.get(f"https://www.googleapis.com/calendar/v3/calendars/{cid}/events",
            headers={"Authorization": f"Bearer {creds.token}"},
            params={"timeMin": now.isoformat() + "Z", "timeMax": (now + datetime.timedelta(days=days)).isoformat() + "Z",
                    "singleEvents": "true", "orderBy": "startTime"}, timeout=15)
        if r.status_code != 200: return ""
        out = []
        for e in r.json().get("items", []):
            st = e.get("start", {}).get("dateTime", e.get("start", {}).get("date", ""))
            out.append(f"• {st[11:16] if 'T' in st else st} {e.get('summary', '(no title)')}")
        return "\n".join(out)
    except Exception:
        return ""

# ----------------------------------------------------------------------------- SHORT-TERM MEMORY
_HISTORY = []   # recent (role, text) turns within this listener run — gives cross-message context
def _remember_turn(role, text):
    _HISTORY.append((role, str(text)[:400]))
    if len(_HISTORY) > 8:
        del _HISTORY[:len(_HISTORY) - 8]
def _history_text():
    return "\n".join(f"{r}: {t}" for r, t in _HISTORY[-6:]) or "(new conversation)"

# ----------------------------------------------------------------------------- INTENT ENGINE
def _intent_context():
    t = rank_tasks(open_tasks()); fu = unique_followups(); rem = _open_reminders()
    tctx = "\n".join(f"[{get_val(x, ['Task ID', 'ID'])}] {get_val(x, ['Task'])} ({get_val(x, ['Priority'], 'P2')})" for x in t[:15]) or "(none)"
    fctx = "\n".join(f"[{get_val(x, ['FID', 'ID'])}] {get_val(x, ['Item'])} — {get_val(x, ['Waiting On'], '—')}" for x in fu[:10]) or "(none)"
    rctx = "\n".join(f"[{get_val(x, ['Reminder ID', 'ID'])}] {get_val(x, ['Text'])} @ {get_val(x, ['When'])}" for x in rem[:10]) or "(none)"
    return tctx, fctx, rctx, _memory_context()
INTENT_SYS = (
    "You are the intent engine for Shivam Negi's assistant (Internal Auditor & Trainer, Moustache hostels, India). "
    "Understand his intent (English/Hindi/Hinglish) and output ONLY a JSON object: "
    "{\"actions\":[...], \"reply\":\"one short warm line in English\"}.\n"
    "ACTIONS:\n"
    " {\"type\":\"add_task\",\"task\":..,\"priority\":\"P1|P2|P3\",\"due\":\"YYYY-MM-DD or ''\",\"category\":\"Task|KRA\"}\n"
    " {\"type\":\"edit_task\",\"id\":\"<Task ID or ''>\",\"match\":\"text\",\"new_task\":\"'' unless renamed\",\"new_priority\":\"'' or P1|P2|P3\",\"new_due\":\"'' or YYYY-MM-DD\"}\n"
    " {\"type\":\"complete_task\",\"id\":\"<Task ID from context or ''>\",\"match\":\"text if no id\"}\n"
    " {\"type\":\"reopen_task\",\"id\":..,\"match\":..}\n"
    " {\"type\":\"set_score\",\"id\":\"<KRA/Task ID>\",\"match\":\"text\",\"score\":\"1 or 0\"}\n"
    " {\"type\":\"add_reminder\",\"text\":\"SUBJECT only, never a time phrase\",\"when\":\"YYYY-MM-DD HH:MM\",\"repeat\":\"none|daily|weekly\"}\n"
    " {\"type\":\"cancel_reminder\",\"id\":\"<Reminder ID or ''>\",\"match\":\"text\"}\n"
    " {\"type\":\"add_followup\",\"item\":..,\"who\":..,\"due\":\"YYYY-MM-DD or ''\"}\n"
    " {\"type\":\"complete_followup\",\"id\":\"<FID or ''>\",\"match\":..}\n"
    " {\"type\":\"remember\",\"key\":..,\"value\":..}\n"
    " {\"type\":\"note\",\"text\":..}\n"
    " {\"type\":\"list_tasks\"} {\"type\":\"list_reminders\"} {\"type\":\"list_followups\"} {\"type\":\"scorecard\"}\n"
    " {\"type\":\"research\",\"query\":\"a question needing web search / thought\"}\n"
    "RULES:\n"
    "1. Specific clock time ('at 3pm','in 2 hours','every day at 8am') -> add_reminder. A to-do with a date or no time -> add_task.\n"
    "2. Waiting on someone -> add_followup. Finished something -> complete_task / complete_followup (match the [ID] from context by MEANING).\n"
    "3. 'mark not completed'/'reopen'/'undo' -> reopen_task. 'move X to Friday'/'change priority' -> edit_task.\n"
    "4. 'mark KRA X as met/1' or 'score X 0' -> set_score. 'cancel/remove the reminder about Y' -> cancel_reminder.\n"
    "5. add_reminder.text is the SUBJECT only (e.g. 'call Manish'), NEVER 'at 9am'.\n"
    "6. Return ALL actions if he lists several. If he lists duplicates to remove, one action per [ID].\n"
    "7. Use CONVERSATION so far to resolve references like 'add that', 'remind me about it', 'the second one'.\n"
    "8. Resolve relative dates/times to Asia/Kolkata. Greeting/chit-chat -> actions:[] with a friendly reply.\n"
    "Return ONLY valid JSON."
)
def _new_task_row(tid, task, prio, due, cat="Task"):
    return [tid, task, cat, prio, due, "Open", now_ist().strftime("%b %Y"), "", ""]
def _do_add_task(a):
    task = (a.get("task") or "").strip()
    if not task: return None
    for r in open_tasks():
        if _sim(task, get_val(r, ["Task"])) >= 0.75:
            return f"Already on your list: {get_val(r, ['Task'])}"
    prio = (a.get("priority") or "").upper()
    if prio not in ("P1", "P2", "P3"):
        prio = "P1" if any(w in task.lower() for w in ("urgent", "asap", "director", "critical", "today", "top priority")) else "P2"
    tid = f"T{int(time.time()*7)%1000000}"
    ws("PA_Tasks").append_row(_new_task_row(tid, task, prio, a.get("due") or "", a.get("category") or "Task"))
    clear_cache("PA_Tasks"); log_event("task_added", task)
    return f"Added: {task} [{prio}]" + (f" · due {a.get('due')}" if a.get("due") else "")
def _find_task_row(a):
    w = ws("PA_Tasks"); rows = cached_records("PA_Tasks")
    tid = str(a.get("id", "")).strip()
    if tid:
        for idx, r in enumerate(rows, 2):
            if get_val(r, ["Task ID", "ID"]).strip() == tid: return w, idx
    match = a.get("match") or a.get("task") or ""
    best, bs = None, 0.0
    if match:
        for idx, r in enumerate(rows, 2):
            s = _sim(match, get_val(r, ["Task"]))
            if s > bs: bs, best = s, idx
    return (w, best) if best and bs >= 0.35 else (w, None)
def _set_task_status(a, status):
    w, row = _find_task_row(a)
    if not row: return "Couldn't find that task."
    hdr = w.row_values(1)
    task = w.cell(row, hdr.index("Task") + 1).value if "Task" in hdr else ""
    if "Status" in hdr: w.update_cell(row, hdr.index("Status") + 1, status)
    clear_cache("PA_Tasks"); log_event("task_" + status.lower(), task or "")
    return (f"Completed: {task} ✅" if status == "Done" else f"Reopened: {task}")
def _do_edit_task(a):
    w, row = _find_task_row(a)
    if not row: return "Couldn't find that task to edit."
    hdr = w.row_values(1); changed = []
    def setcol(name, val):
        if val and name in hdr:
            w.update_cell(row, hdr.index(name) + 1, val); changed.append(name.lower())
    if a.get("new_task"): setcol("Task", a["new_task"])
    if a.get("new_priority"): setcol("Priority", str(a["new_priority"]).upper())
    if a.get("new_due"): setcol("Due Date", a["new_due"])
    clear_cache("PA_Tasks")
    task = w.cell(row, hdr.index("Task") + 1).value if "Task" in hdr else ""
    return f"Updated {task} ({', '.join(changed)})" if changed else "Nothing changed."
def _do_set_score(a):
    w, row = _find_task_row(a)
    if not row: return "Couldn't find that KRA/task to score."
    hdr = w.row_values(1); score = str(a.get("score", "")).strip()
    if "Score" in hdr: w.update_cell(row, hdr.index("Score") + 1, score)
    clear_cache("PA_Tasks")
    task = w.cell(row, hdr.index("Task") + 1).value if "Task" in hdr else ""
    return f"Scored '{task}': {score}"
def _do_cancel_reminder(a):
    w = ws("PA_Reminders"); rows = cached_records("PA_Reminders")
    tid = str(a.get("id", "")).strip(); match = a.get("match") or a.get("text") or ""
    best, bs, txt = None, 0.0, ""
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Status"]).lower() not in ("", "open"): continue
        if tid and get_val(r, ["Reminder ID", "ID"]).strip() == tid:
            best, bs, txt = idx, 1.0, get_val(r, ["Text"]); break
        s = _sim(match, get_val(r, ["Text"]))
        if s > bs: bs, best, txt = s, idx, get_val(r, ["Text"])
    if best and bs >= 0.35:
        hdr = w.row_values(1)
        if "Status" in hdr: w.update_cell(best, hdr.index("Status") + 1, "Cancelled")
        clear_cache("PA_Reminders")
        return f"Cancelled reminder: {txt}"
    return "Couldn't find that reminder."
def _snooze_reminder(rid, mins):
    w = ws("PA_Reminders"); rows = cached_records("PA_Reminders")
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Reminder ID", "ID"]).strip() == str(rid):
            hdr = w.row_values(1)
            nxt = (now_ist() + datetime.timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M")
            if "When" in hdr: w.update_cell(idx, hdr.index("When") + 1, nxt)
            if "Status" in hdr: w.update_cell(idx, hdr.index("Status") + 1, "Open")
            clear_cache("PA_Reminders"); send(f"😴 Snoozed to {nxt}"); return
    send("Couldn't find that reminder to snooze.")

def _do_add_reminder(a):
    text = (a.get("text") or "").strip()
    when = _normalize_datetime(a.get("when") or "")
    if not when: return "Couldn't work out the time — try 'remind me tomorrow 9am to ...'."
    if not text or re.match(r"(?i)^(at |in |tomorrow|today|every|reminder|due)\b", text):
        text = text if len(text) > 3 else "Reminder"
    for r in _open_reminders():
        if _sim(text, get_val(r, ["Text"])) >= 0.75 and get_val(r, ["When"]) == when:
            return f"Reminder already set: {text} @ {when}"
    ws("PA_Reminders").append_row([f"R{int(time.time()*7)%1000000}", text, when, a.get("repeat", "none"), "Open",
                                   now_ist().strftime("%Y-%m-%d %H:%M")])
    clear_cache("PA_Reminders")
    rep = str(a.get("repeat", "none")).lower()
    return f"⏰ Reminder: {text} @ {when}" + (f" ({rep})" if rep != "none" else "")
def _do_add_followup(a):
    item = (a.get("item") or "").strip()
    if not item: return None
    for r in open_followups():
        if _sim(item, get_val(r, ["Item"])) >= 0.75:
            return f"Already tracking: {get_val(r, ['Item'])}"
    ws("PA_Followups").append_row([f"F{int(time.time()*7)%1000000}", item, a.get("who", ""),
                                   now_ist().strftime("%Y-%m-%d"), a.get("due", ""), "Open", ""])
    clear_cache("PA_Followups")
    return f"Tracking: {item} (waiting on {a.get('who') or '—'})"
def _do_complete_followup(a):
    w = ws("PA_Followups"); rows = cached_records("PA_Followups")
    tid = str(a.get("id", "")).strip(); match = a.get("match") or ""
    best, bs, item = None, 0.0, ""
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Status"]).lower() not in ("", "open"): continue
        if tid and get_val(r, ["FID", "ID"]).strip() == tid:
            best, bs, item = idx, 1.0, get_val(r, ["Item"]); break
        s = _sim(match, get_val(r, ["Item"]))
        if s > bs: bs, best, item = s, idx, get_val(r, ["Item"])
    if best and bs >= 0.35:
        hdr = w.row_values(1)
        if "Status" in hdr: w.update_cell(best, hdr.index("Status") + 1, "Done")
        clear_cache("PA_Followups")
        return f"Closed follow-up: {item} ✅"
    return "Couldn't find that follow-up."
def _do_remember(a):
    try:
        ws("PA_Memory").append_row([f"M{int(time.time())%100000}", a.get("category", "context"),
                                    a.get("key", "")[:60], a.get("value", ""), ""])
        clear_cache("PA_Memory")
        return f"Noted: {a.get('key','')}"
    except Exception:
        return "Saved to memory."
def _do_note(a):
    try:
        ws("PA_Capture").append_row([f"C{int(time.time())%100000}", now_ist().strftime("%Y-%m-%d %H:%M"),
                                     "note", a.get("text", ""), "", "New", ""])
        clear_cache("PA_Capture")
        return f"Captured: {a.get('text','')[:60]}"
    except Exception:
        return "Saved note."
def _tasks_text():
    t = rank_tasks(open_tasks())
    if not t: return "No open tasks."
    return "\n".join(f"{i}. [{get_val(x, ['Priority'], 'P3')}] {get_val(x, ['Task'])} (due {get_val(x, ['Due Date', 'Due'], '—')})"
                     for i, x in enumerate(t[:15], 1))
def _list_reminders():
    op = _open_reminders()
    if not op: return "No upcoming reminders."
    return "<b>Upcoming reminders</b>\n" + "\n".join(f"• {get_val(r, ['When'])} — {get_val(r, ['Text'])}" for r in op[:15])
def _list_followups():
    fu = unique_followups()
    if not fu: return "No open follow-ups."
    return "<b>Waiting on</b>\n" + "\n".join(f"• {get_val(r, ['Item'])} — {get_val(r, ['Waiting On'], '—')}" for r in fu[:15])
def _scorecard():
    rows = [r for r in cached_records("PA_Tasks") if get_val(r, ["Category"]).lower() == "kra"]
    if not rows: return "No KRA scores yet."
    bm = {}
    for r in rows:
        m = get_val(r, ["Month"])
        if not m: continue
        bm.setdefault(m, [0, 0]); bm[m][1] += 1
        try:
            if float(get_val(r, ["Score"], "0")) >= 1: bm[m][0] += 1
        except Exception: pass
    return "<b>KRA scorecard</b>\n" + "\n".join(f"• {m}: {g}/{t} met" for m, (g, t) in bm.items())
def handle_intent(text):
    tg("sendChatAction", chat_id=TG_CHAT, action="typing")
    _remember_turn("user", text)
    def _say(t):
        _remember_turn("assistant", _plain(t)); return send(t)
    tctx, fctx, rctx, mctx = _intent_context()
    now = now_ist().strftime("%Y-%m-%d %H:%M (%A)")
    prompt = (f"NOW: {now} (Asia/Kolkata)\nCONVERSATION SO FAR:\n{_history_text()}\n\n"
              f"OPEN TASKS:\n{tctx}\n\nFOLLOW-UPS:\n{fctx}\n\nREMINDERS:\n{rctx}\n\n"
              f"MEMORY:\n{mctx}\n\nLatest message: {text}")
    out, _ = ai(system=INTENT_SYS, prompt=prompt, purpose="deep")
    try:
        data = json.loads(re.search(r"\{.*\}", out, re.DOTALL).group())
    except Exception:
        return _say(research(text))
    actions = data.get("actions", [])
    if not actions:
        return _say(data.get("reply") or "Noted.")
    results = []
    for a in actions:
        t = a.get("type")
        try:
            if t == "add_task":            results.append(_do_add_task(a))
            elif t == "edit_task":         results.append(_do_edit_task(a))
            elif t == "complete_task":     results.append(_set_task_status(a, "Done"))
            elif t == "reopen_task":       results.append(_set_task_status(a, "Open"))
            elif t == "set_score":         results.append(_do_set_score(a))
            elif t == "add_reminder":      results.append(_do_add_reminder(a))
            elif t == "cancel_reminder":   results.append(_do_cancel_reminder(a))
            elif t == "add_followup":      results.append(_do_add_followup(a))
            elif t == "complete_followup": results.append(_do_complete_followup(a))
            elif t == "remember":          results.append(_do_remember(a))
            elif t == "note":              results.append(_do_note(a))
            elif t == "list_tasks":        return _say("<b>Open tasks</b>\n" + _tasks_text())
            elif t == "list_reminders":    return _say(_list_reminders())
            elif t == "list_followups":    return _say(_list_followups())
            elif t == "scorecard":         return _say(_scorecard())
            elif t == "research":          return _say(research(a.get("query", text)))
        except Exception:
            results.append(f"(couldn't do {t})")
    results = [r for r in results if r]
    if not results:
        return _say(data.get("reply") or "Done.")
    _say(results[0] if len(results) == 1 else "Done:\n" + "\n".join(f"• {r}" for r in results))

# ----------------------------------------------------------------------------- JOBS
def job_ping():
    setup_tg_commands()
    send("✅ <b>Your PA is online.</b>", buttons=[[btn("👋 Say hi", "ping:hi")], [btn("📋 Today", "cmd:today")]])
def job_morning():
    d = now_ist().strftime("%A, %d %b")
    top = rank_tasks(open_tasks())[:5]
    lines = ["☀️ <b>Good morning, Shivam</b>", d, ""]
    mtg = calendar_events(1)
    if mtg: lines += ["<b>Today's meetings</b>", mtg, ""]
    if top:
        lines.append("<b>Top tasks</b>")
        for i, t in enumerate(top, 1):
            due = get_val(t, ["Due Date", "Due"])
            lines.append(f"{i}. [{get_val(t, ['Priority'], 'P3')}] {html.escape(get_val(t, ['Task']))}" + (f" · due {due}" if due else ""))
    else:
        lines.append("No open tasks — clear runway.")
    kras = open_kras()
    if kras: lines += ["", "🔔 <b>KRAs open this month</b>"] + [f"• {html.escape(k)}" for k in kras[:6]]
    fu = unique_followups()
    if fu: lines += ["", "<b>Waiting on</b>"] + [f"• {html.escape(get_val(r, ['Item']))} — {html.escape(get_val(r, ['Waiting On'], '—'))}" for r in fu[:6]]
    nudge, _ = ai(prompt="One short warm one-line nudge to start the workday. No emoji.", purpose="quick")
    lines += ["", f"<i>{html.escape(nudge or 'One focused block at a time.')}</i>"]
    rows = [btn(f"✅ {i}", f"done:{get_val(t, ['Task ID', 'ID'])}") for i, t in enumerate(top, 1)]
    send("\n".join(lines), buttons=([rows] if rows else []) + [[btn("➕ Add task", "cmd:addhelp")]])
def job_reflection():
    set_config("STATE_awaiting_reflection", "1")
    send("🌙 <b>Nightly check-in</b>\nReply in one message:\n1. Finished today?\n2. Blocked / waiting on?\n"
         "3. Learned?\n4. Energy (1-5)?\n5. Tomorrow's #1?", buttons=[[btn("😴 Skip", "reflect:skip")]])
def job_collect():
    c = config()
    if c.get("STATE_awaiting_reflection") != "1": return
    ans = None
    for u in tg("getUpdates", offset=int(c.get("STATE_tg_offset", 0)) or None, timeout=0).get("result", []):
        m = u.get("message") or {}
        if str(m.get("chat", {}).get("id")) == str(TG_CHAT) and m.get("text") and not m["text"].startswith("/"):
            ans = m["text"]
    if not ans: return
    ws("PA_Reflections").append_row([now_ist().strftime("%Y-%m-%d"), ans, "", "", "", ""])
    clear_cache("PA_Reflections"); log_event("reflection", "logged"); set_config("STATE_awaiting_reflection", "0")
    send("📝 Logged. Rest well, Shivam.")
def _watchdog_text():
    parts, today = [], now_ist().date()
    overdue = []
    for t in open_tasks():
        due = get_val(t, ["Due Date", "Due"])
        try:
            if due and datetime.date.fromisoformat(due) < today: overdue.append(t)
        except Exception: pass
    if overdue:
        parts.append("<b>Overdue</b>\n" + "\n".join(f"• {get_val(t, ['Task'])} (due {get_val(t, ['Due Date', 'Due'])})" for t in overdue[:6]))
    kr = open_kras()
    if kr and today.day >= 18:
        parts.append("<b>KRAs still open</b>\n" + "\n".join(f"• {k}" for k in kr[:6]))
    fu = unique_followups()
    if fu:
        parts.append("<b>Waiting on</b>\n" + "\n".join(f"• {get_val(r, ['Item'])} — {get_val(r, ['Waiting On'], '—')}" for r in fu[:6]))
    return "\n\n".join(parts)
def job_evening():
    txt = _watchdog_text()
    send("🔔 <b>Evening check-in</b>\n\n" + txt if txt else "🔔 Evening check-in — nothing overdue or pending. Nicely on top of it.")
def job_daily():
    log_event("daily_digest", f"{len(open_tasks())} open tasks")
def job_sunday():
    send(f"🏆 <b>Sunday digest</b> — {now_ist().strftime('%d %b')}\n"
         f"• Open tasks: <b>{len(open_tasks())}</b>\n• Waiting on: <b>{len(unique_followups())}</b>\n\n{_scorecard()}")
def job_report(kind="monthly"):
    tasks = rank_tasks(open_tasks())
    tl = "\n".join(f"- [{get_val(t, ['Priority'])}] {get_val(t, ['Task'])}" for t in tasks[:20]) or "None."
    fl = "\n".join(f"- {get_val(f, ['Item'])} (waiting on {get_val(f, ['Waiting On'])})" for f in unique_followups()[:10]) or "None."
    draft, _ = ai(system="Executive Telegram-HTML report, concise bullets, English.",
                  prompt=f"Write a {kind} progress report for Shivam Negi (Internal Auditor & Trainer, Moustache).\n"
                         f"Open tasks:\n{tl}\n\nWaiting on:\n{fl}\n\n{_scorecard()}", purpose="long")
    send(f"🧾 <b>{kind.title()} report</b>\n\n{draft or 'No data yet.'}")

# ----------------------------------------------------------------------------- COMMANDS / BUTTONS
def mark_done(task_id):
    w = ws("PA_Tasks"); rows = cached_records("PA_Tasks"); tid = str(task_id).strip()
    row, name = None, ""
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Task ID", "ID"]).strip() == tid:
            row, name = idx, get_val(r, ["Task"]); break
    if not row: return None
    hdr = w.row_values(1)
    if "Status" in hdr: w.update_cell(row, hdr.index("Status") + 1, "Done")
    clear_cache("PA_Tasks"); log_event("task_done", name or tid)
    return name or tid
def handle_command(text):
    cmd = text.split()[0].lower().lstrip("/")
    arg = text[len(cmd) + 1:].strip() if " " in text else ""
    if cmd in ("start", "help"):
        send("🤖 <b>Talk to me naturally:</b>\n• “Jaipur audit point done”\n• “Remind me at 4pm to call GM”\n"
             "• “Waiting on Deepak for the sign-off”\n• “What are my tasks?”")
    elif cmd == "today": job_morning()
    elif cmd == "tasks": send("<b>Open tasks</b>\n" + _tasks_text())
    elif cmd == "reflect": job_reflection()
    elif cmd == "report": job_report("monthly")
    elif cmd in ("ask", "research"): send(research(arg) if arg else "Ask me anything.")
    else: handle_intent(text)
def handle_callback(cb):
    data = cb.get("data", ""); chat = cb["message"]["chat"]["id"]; mid = cb["message"]["message_id"]
    if data.startswith("done:"):
        t = mark_done(data.split(":", 1)[1]); answer_cb(cb["id"], "Done ✅")
        if t: edit(chat, mid, cb["message"].get("text", "") + f"\n\n✅ <b>{html.escape(t)}</b> — done!")
    elif data == "cmd:today": answer_cb(cb["id"]); job_morning()
    elif data == "cmd:addhelp": answer_cb(cb["id"]); send("Just tell me, e.g. “add task audit Udaipur by Friday P1”.")
    elif data == "ping:hi": answer_cb(cb["id"], "👋"); send("👋 Hello Shivam! All systems working.")
    elif data == "reflect:skip":
        set_config("STATE_awaiting_reflection", "0"); answer_cb(cb["id"], "Skipped"); edit(chat, mid, "🌙 Skipped. Rest well!")
    elif data.startswith("snooze:"):
        try:
            _, rid, mins = data.split(":"); answer_cb(cb["id"], f"Snoozed {mins}m"); _snooze_reminder(rid, int(mins))
        except Exception:
            answer_cb(cb["id"])
    else: answer_cb(cb["id"])

# ----------------------------------------------------------------------------- SENSES
def transcribe(data):
    key = _k("GROQ_API_KEY")
    if not key: return ""
    try:
        r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"}, files={"file": ("audio.ogg", data, "audio/ogg")},
            data={"model": "whisper-large-v3"}, timeout=120)
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except Exception:
        return ""
def vision(data, prompt):
    b64 = base64.b64encode(data).decode()
    order = [("github", "gpt-4o"), ("together", "meta-llama/Llama-Vision-Free"), ("mistral", "pixtral-12b-2409")]
    for prov, model in order:
        if prov not in OAI or not _k(OAI[prov][1]): continue
        try:
            r = requests.post(OAI[prov][0], timeout=120,
                headers={"Authorization": f"Bearer {_k(OAI[prov][1])}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    if _k("GEMINI_API_KEY"):
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={_k('GEMINI_API_KEY')}"
            r = requests.post(url, timeout=120, json={"contents": [{"parts": [
                {"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}]})
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
    return ""
def _romanize(text):
    if not text or not re.search(r"[ऀ-ॿ]", text): return text
    out, _ = ai(system="Transliterate Devanagari to Latin/Hinglish. Output only the result.", prompt=text, purpose="quick")
    return out or text
def _tg_file(file_id):
    fp = tg("getFile", file_id=file_id).get("result", {}).get("file_path", "")
    return requests.get(f"https://api.telegram.org/file/bot{TG_TOKEN}/{fp}", timeout=90).content
def handle_voice(file_id):
    tg("sendChatAction", chat_id=TG_CHAT, action="typing")
    try:
        txt = _romanize(transcribe(_tg_file(file_id)))
    except Exception:
        return send("Couldn't fetch that voice note.")
    if not txt: return send("Couldn't transcribe that — try again or type it.")
    send(f"🎙️ <i>{html.escape(txt)}</i>"); _dispatch_text(txt)
def handle_photo(file_id, caption=""):
    tg("sendChatAction", chat_id=TG_CHAT, action="typing"); send("🖼️ <i>reading the image…</i>")
    try:
        out = vision(_tg_file(file_id), "Extract the useful content/notes/action items for Shivam concisely."
                     + (f" Caption: {caption}." if caption else ""))
    except Exception:
        return send("Couldn't fetch that image.")
    send(out + "\n\n<i>Want any of these as tasks? Just say which.</i>" if out else "Couldn't read that image — try a clearer photo.")

# ----------------------------------------------------------------------------- DISPATCH + REMINDERS
def _dispatch_text(txt):
    if txt.startswith("/"): return handle_command(txt)
    if config().get("STATE_awaiting_reflection") == "1":
        return job_collect()
    handle_intent(txt)
def _fire_user_reminders():
    now = now_ist(); nows = now.strftime("%Y-%m-%d %H:%M")
    w = ws("PA_Reminders"); rows = cached_records("PA_Reminders")
    hdr, fired = None, set()
    for idx, r in enumerate(rows, 2):
        if get_val(r, ["Status"]).lower() not in ("", "open"): continue
        when = get_val(r, ["When"])
        if not when or when > nows: continue
        text = get_val(r, ["Text"]); rid = get_val(r, ["Reminder ID", "ID"]); key = (_norm(text), when)
        if key not in fired:                       # collapse duplicate rows into one ping
            send(f"⏰ <b>Reminder</b>: {html.escape(text)}",
                 buttons=[[btn("😴 +30m", f"snooze:{rid}:30"), btn("😴 +1h", f"snooze:{rid}:60")]])
            fired.add(key)
        if hdr is None: hdr = w.row_values(1)
        rep = get_val(r, ["Repeat"]).lower()
        if rep in ("daily", "weekly"):
            try:
                nxt = (datetime.datetime.strptime(when, "%Y-%m-%d %H:%M")
                       + datetime.timedelta(days=1 if rep == "daily" else 7)).strftime("%Y-%m-%d %H:%M")
                if "When" in hdr: w.update_cell(idx, hdr.index("When") + 1, nxt)
            except Exception:
                if "Status" in hdr: w.update_cell(idx, hdr.index("Status") + 1, "Done")
        elif "Status" in hdr:
            w.update_cell(idx, hdr.index("Status") + 1, "Done")
    if fired: clear_cache("PA_Reminders")
def _run_due_reminders():
    now = now_ist(); today = now.strftime("%Y-%m-%d"); hm = now.strftime("%H:%M"); dow = now.weekday()
    c = config()
    JOBS = {"morning": job_morning, "evening": job_evening, "reflection": job_reflection,
            "daily": job_daily, "sunday": job_sunday}
    def fire(key, ok):
        if ok and c.get(f"STATE_last_{key}", "") != today:
            set_config(f"STATE_last_{key}", today)
            try: JOBS[key]()
            except Exception: pass
    fire("morning", hm >= "07:00")
    fire("evening", hm >= "18:00")
    fire("reflection", hm >= "21:00")
    fire("daily", hm >= "21:15")
    fire("sunday", dow == 6 and hm >= "18:00")

# ----------------------------------------------------------------------------- LISTENER (bullet-proof)
def job_listen(minutes=340):
    try: setup_tg_commands()
    except Exception: pass
    offset = int(config().get("STATE_tg_offset", 0)) or None
    end = time.time() + minutes * 60
    last_check = last_save = 0
    while time.time() < end:
        if time.time() - last_check > 60:
            last_check = time.time()
            try: _run_due_reminders()
            except Exception: pass
            try: _fire_user_reminders()
            except Exception: pass
        try:
            res = tg("getUpdates", offset=offset, timeout=25).get("result", [])
        except Exception:
            time.sleep(3); continue
        for u in res:
            try:
                offset = u["update_id"] + 1
                if "message" in u:
                    m = u["message"]
                    if str(m.get("chat", {}).get("id")) != str(TG_CHAT): continue
                    if m.get("text"): _dispatch_text(m["text"])
                    elif m.get("voice") or m.get("audio"): handle_voice((m.get("voice") or m.get("audio"))["file_id"])
                    elif m.get("photo"): handle_photo(m["photo"][-1]["file_id"], m.get("caption", ""))
                elif "callback_query" in u:
                    handle_callback(u["callback_query"])
            except Exception as e:
                try: send(f"⚠️ Hit a snag: <code>{html.escape(str(e))[:150]}</code>")
                except Exception: pass
        if offset and time.time() - last_save > 10:      # persist offset in batches, not per-message
            last_save = time.time(); set_config("STATE_tg_offset", offset)
    if offset: set_config("STATE_tg_offset", offset)

# ----------------------------------------------------------------------------- MAIN
def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "ping"
    fn = {"ping": job_ping, "morning": job_morning, "evening": job_evening, "reflection": job_reflection,
          "collect": job_collect, "daily": job_daily, "sunday": job_sunday,
          "weekly": lambda: job_report("weekly"), "monthly": lambda: job_report("monthly"),
          "report": job_report, "listen": job_listen}.get(job)
    if not fn:
        print("Unknown job:", job); sys.exit(1)
    try:
        fn()
    except Exception:
        err = traceback.format_exc(); print(err)
        try: send(f"⚠️ Job <b>{job}</b> failed:\n<code>{html.escape(err[-500:])}</code>")
        except Exception: pass
        sys.exit(1)

if __name__ == "__main__":
    main()
