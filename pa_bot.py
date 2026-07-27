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
    """Actionable tasks: not done/missed, and either no month or the current month
    (past-month KRA rows stay in the sheet as history but don't clutter the daily list)."""
    try:
        rows = ws("PA_Tasks").get_all_records()
    except Exception:
        return []
    cur = now_ist().strftime("%b %Y")
    out = []
    for r in rows:
        st = str(r.get("Status", "")).strip().lower()
        if st in ("done", "cancelled", "complete", "completed", "missed"):
            continue
        if not str(r.get("Task", "")).strip():
            continue
        m = str(r.get("Month", "")).strip()
        if m and m != cur:
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

# ============================================================================= MULTI-AGENT FRAMEWORK
# A small crew of specialised agents coordinated by a Chief-of-Staff orchestrator.
# Each agent has a persona, an allowed tool subset, and can hand off to another agent.
# Adding an agent or tool is a few lines below — the runner is generic.

# ---- TOOLS (shared registry) -----------------------------------------------
def _tasks_text(_arg=""):
    t = rank_tasks(open_tasks())
    if not t:
        return "No open tasks."
    return "\n".join(f"{i}. [{x.get('Priority')}] {x.get('Task')} "
                     f"(due {x.get('Due Date') or '—'})" for i, x in enumerate(t[:12], 1))

def _tool_add(arg):
    j, _ = ai(system=("Extract a task as JSON {\"task\":..,\"priority\":\"P1|P2|P3\",\"due\":\"YYYY-MM-DD or ''\"}. "
                      f"Today is {now_ist().strftime('%Y-%m-%d')}. Resolve relative dates. Return ONLY JSON."),
              prompt=arg, tier="quick")
    try:
        d = json.loads(re.search(r"\{.*\}", j, re.DOTALL).group())
    except Exception:
        d = {"task": arg, "priority": "P2", "due": ""}
    task = (d.get("task") or arg).strip()
    prio = (d.get("priority") or "P2").upper()
    due = d.get("due") or ""
    tid = f"T{int(time.time())%100000}"
    ws("PA_Tasks").append_row([tid, task, "Task", prio, due, "Open",
                               now_ist().strftime("%b %Y"), "", ""])
    log_event("task_added", task)
    return f"Added '{task}' [{prio}]" + (f" due {due}" if due else "")

# ---- Quick-capture tools ---------------------------------------------------
def _tool_capture(arg):
    j, _ = ai(system=("Classify this captured note as JSON {\"type\":\"idea|link|note|task\","
                      "\"tags\":\"comma,tags\"}. Return ONLY JSON."), prompt=arg, tier="quick")
    try:
        d = json.loads(re.search(r"\{.*\}", j, re.DOTALL).group())
    except Exception:
        d = {"type": "note", "tags": ""}
    cid = f"C{int(time.time())%100000}"
    try:
        ws("PA_Capture").append_row([cid, now_ist().strftime("%Y-%m-%d %H:%M"),
                                     d.get("type", "note"), arg, d.get("tags", ""), "New", ""])
    except Exception:
        return "Saved (couldn't reach PA_Capture — check the tab exists)."
    return f"Captured as {d.get('type','note')} [{d.get('tags','')}] ✅"

def _tool_recall(arg):
    try:
        rows = ws("PA_Capture").get_all_records()
    except Exception:
        return "Nothing captured yet."
    if not rows:
        return "Nothing captured yet."
    words = set(re.findall(r"[a-z0-9]+", arg.lower()))
    scored = []
    for r in rows:
        blob = f"{r.get('Content','')} {r.get('Tags','')}".lower()
        s = len(words & set(re.findall(r"[a-z0-9]+", blob)))
        scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    picks = [r for s, r in scored if s > 0][:6] or [r for _, r in scored[-6:]]
    return "\n".join(f"• [{r.get('Type')}] {r.get('Content')}" for r in picks)

def _tool_complete(arg):
    tasks = open_tasks()
    tlist = "\n".join(f"{r.get('Task ID')}: {r.get('Task')}" for r in tasks) or "(none)"
    j, _ = ai(system="Return ONLY the single Task ID from the list that best matches, or NONE.",
              prompt=f"Tasks:\n{tlist}\n\nWhich did he finish: {arg}", tier="quick")
    tid = (j.strip().split() or ["NONE"])[0]
    t = mark_done(tid) if tid and tid != "NONE" else None
    if not t:
        t = fuzzy_complete(arg)
    return f"Marked '{t}' done ✅" if t else "Couldn't find a matching open task."

def _tool_scorecard(_arg=""):
    sc = kra_scorecard()
    if not sc:
        return "No KRA data yet (KRAs are PA_Tasks rows with Category=KRA)."
    lines = [f"{m}: {g}/{t} met" for m, g, t in sc]
    tot_g = sum(g for _, g, _ in sc); tot_t = sum(t for _, _, t in sc)
    return "KRA scorecard —\n" + "\n".join(lines) + f"\nYear-to-date: {tot_g}/{tot_t}"

TOOLS = {
    "web_search":    (lambda a: research(a),   "Search the web; returns a synthesised answer. Arg = the query."),
    "list_tasks":    (_tasks_text,             "List Shivam's open tasks, ranked. Arg ignored."),
    "add_task":      (_tool_add,               "Add a task. Arg = natural description incl. priority/date."),
    "complete_task": (_tool_complete,          "Mark a task done. Arg = a few words identifying the task."),
    "kra_scorecard": (_tool_scorecard,         "KRA scores per month + year-to-date. Arg ignored."),
    "capture":       (_tool_capture,           "Save a thought/link/idea/note for later. Arg = the thing to remember."),
    "recall":        (_tool_recall,            "Find previously captured notes. Arg = what he's looking for."),
}

# ---- AGENTS (registry) -----------------------------------------------------
AGENTS = {
    "chief": {
        "desc": "General assistant, planner and router. Handles anything not clearly another agent's job: "
                "questions, advice, chit-chat, mixed requests.",
        "tools": ["web_search", "list_tasks", "add_task", "complete_task", "kra_scorecard"],
        "tier": "deep",
        "persona": "You are Shivam Negi's Chief of Staff. He's an Internal Auditor & Trainer at Moustache "
                   "(India Hostels). Be sharp, proactive and concise.",
    },
    "tasks": {
        "desc": "To-do management: add, complete, list, or prioritise tasks.",
        "tools": ["list_tasks", "add_task", "complete_task"],
        "tier": "quick",
        "persona": "You are Shivam's Task manager. Capture and close tasks precisely. Confirm briefly.",
    },
    "capture": {
        "desc": "Quick-capture: he dumps a thought/idea/link/note to save for later, or asks to recall past notes.",
        "tools": ["capture", "recall"],
        "tier": "quick",
        "persona": "You are Shivam's Quick-capture agent. Save whatever he dumps, tagged; recall on request. "
                   "Confirm in one short line.",
    },
    "research": {
        "desc": "Questions needing web research: industry best practices, current facts, how-to, benchmarks.",
        "tools": ["web_search"],
        "tier": "deep",
        "persona": "You are Shivam's Research Analyst for hospitality (hostels), internal audit and training "
                   "in India. Always search before answering; be specific and practical; cite sources.",
    },
    "coach": {
        "desc": "Nightly reflection / check-in and light motivation.",
        "tools": [], "tier": "quick", "action": "reflect",
        "persona": "You are Shivam's Coach.",
    },
    "reporter": {
        "desc": "Weekly/monthly progress reports for the Director. (Report engine currently paused.)",
        "tools": ["kra_scorecard", "list_tasks"], "tier": "long", "action": "report",
        "persona": "You are Shivam's Reporter.",
    },
}

def agent_run(name, text, depth=0):
    a = AGENTS[name]
    if a.get("action") == "reflect":
        job_reflection(); return ""
    if a.get("action") == "report":
        job_report("monthly"); return ""
    tool_desc = "\n".join(f"  {t}: {TOOLS[t][1]}" for t in a["tools"]) or "  (none)"
    others = ", ".join(n for n in AGENTS if n != name)
    sysp = (a["persona"] + "\n\nTOOLS:\n" + tool_desc +
            "\n\nRespond with EXACTLY ONE line, one of:\n"
            "  USE <tool> | <argument>\n"
            f"  HANDOFF <agent> | <subtask>   (agents: {others})\n"
            "  REPLY <your final answer to Shivam, Telegram HTML, concise>\n"
            "Call a tool when it helps; hand off if another agent fits better; else REPLY.")
    ctx = f"Shivam said: {text}\n"
    for _ in range(4):
        step, _p = ai(system=sysp, prompt=ctx, tier=a["tier"])
        if not step.strip():
            return research(text) if "web_search" in a["tools"] else ""
        line = step.strip().splitlines()[0]
        head, _, rest = line.partition(" ")
        head = head.upper().strip()
        if head == "USE":
            tool, _, arg = rest.partition("|")
            tool = tool.strip()
            if tool in TOOLS:
                ctx += f"\n[{tool} → {TOOLS[tool][0](arg.strip())}]\n"
            else:
                ctx += f"\n[no tool '{tool}']\n"
        elif head == "HANDOFF" and depth < 2:
            ag, _, sub = rest.partition("|")
            ag = ag.strip().lower()
            if ag in AGENTS and ag != name:
                return agent_run(ag, (sub.strip() or text), depth + 1)
            ctx += "\n[handoff failed]\n"
        elif head == "REPLY":
            return rest.strip()
        else:
            return step.strip()   # model replied plainly
    final, _ = ai(system="Give Shivam the final answer, concise, Telegram HTML.", prompt=ctx, tier="long")
    return final or "Done."

def orchestrate(text):
    """Chief-of-Staff routing: pick the best agent, run it, send the reply."""
    tg("sendChatAction", chat_id=TG_CHAT, action="typing")
    desc = "\n".join(f"- {n}: {a['desc']}" for n, a in AGENTS.items())
    pick, _ = ai(system=("Route Shivam's message to ONE agent. Reply with ONLY the agent name.\n"
                         "Agents:\n" + desc), prompt=text, tier="quick")
    name = (pick.strip().split() or ["chief"])[0].lower()
    if name not in AGENTS:
        name = "chief"
    out = agent_run(name, text)
    if out:
        send(out)

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

# ---------- KRA scorecard (KRAs are just tasks: rows in PA_Tasks with Category=KRA, a Month + a 1/0 Score)
def kra_task_rows():
    try:
        rows = ws("PA_Tasks").get_all_records()
    except Exception:
        return []
    return [r for r in rows if str(r.get("Category", "")).strip().lower() == "kra"
            and str(r.get("Task", "")).strip()]

def kra_current(rows=None):
    rows = rows if rows is not None else kra_task_rows()
    mon = now_ist().strftime("%b %Y")
    items = [(str(r.get("Task", "")).strip(), str(r.get("Score", "")).strip())
             for r in rows if str(r.get("Month", "")).strip() == mon]
    return mon, items

def kra_scorecard(rows=None):
    rows = rows if rows is not None else kra_task_rows()
    by_month = {}
    order = []
    for r in rows:
        m = str(r.get("Month", "")).strip()
        if not m:
            continue
        if m not in by_month:
            by_month[m] = [0, 0]; order.append(m)
        by_month[m][1] += 1
        try:
            if int(float(r.get("Score", 0) or 0)) >= 1:
                by_month[m][0] += 1
        except Exception:
            pass
    return [(m, by_month[m][0], by_month[m][1]) for m in order]

def _report_context(kind):
    days = 7 if kind == "weekly" else 31
    log = _log_rows(days)
    tasks = rank_tasks(open_tasks())
    open_list = "\n".join(
        f"- [{t.get('Priority')}] {t.get('Task')} (due {t.get('Due Date')}, {t.get('Status')})"
        for t in tasks[:30]) or "None open."
    mon, kras = kra_current()
    kra_txt = "\n".join(f"- {k}  [score: {s or 'pending'}]" for k, s in kras) \
              or "No KRAs logged for this month."
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
# PA_Tasks columns: Task ID | Task | Category | Priority | Due Date | Status | Month | Score | Notes
def _new_task_row(tid, task, prio, due, category="Task"):
    return [tid, task, category, prio, due, "Open", now_ist().strftime("%b %Y"), "", ""]

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
    ws("PA_Tasks").append_row(_new_task_row(tid, task, prio, due))
    log_event("task_added", task)
    return tid, task, prio, due

def mark_done(task_id):
    w = ws("PA_Tasks")
    cell = w.find(str(task_id), in_column=1)   # match Task ID column only
    if not cell:
        return None
    row = cell.row
    headers = w.row_values(1)
    def col(name):
        return headers.index(name) + 1 if name in headers else None
    task = w.cell(row, col("Task")).value if col("Task") else None
    if col("Status"):
        w.update_cell(row, col("Status"), "Done")
    if col("Notes"):
        w.update_cell(row, col("Notes"), f"done {now_ist().strftime('%Y-%m-%d')}")
    log_event("task_done", task or task_id)
    return task

# ----------------------------------------------------------------------------- helper used by tools
def fuzzy_complete(text):
    """Fallback task matcher: pick the open task with the most shared words."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    best, score = None, 0
    for r in open_tasks():
        tw = set(re.findall(r"[a-z0-9]+", str(r.get("Task", "")).lower()))
        s = len(words & tw)
        if s > score:
            score, best = s, r
    return mark_done(best.get("Task ID")) if best and score >= 1 else None

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
            send("Ask me anything, e.g. <code>/ask 2026 hostel check-in best practices</code>")
        elif cmd == "research":
            send(research(arg))
        else:
            orchestrate(arg)
    else:
        orchestrate(text)

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
                            orchestrate(txt)
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
