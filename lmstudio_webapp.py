#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmstudio_webapp.py - v2.1
==========================
Web app AI voi giao dien moi + luu lich su chat.

Database:
  - Local / LAN : SQLite (mac dinh, khong can cai them)
  - Render cloud: Turso (libSQL cloud, mien phi 5GB)

Cai dat:
  pip install flask openai
  pip install libsql-client   # Chi can khi dung Turso

Chay local:
  python lmstudio_webapp.py

Deploy Render (dung Turso):
  Bien moi truong can set:
    TURSO_URL      = libsql://ten-db.turso.io
    TURSO_TOKEN    = eyJhbGci...
    GROQ_API_KEY   = gsk_...
    OPENROUTER_API_KEY = sk-or-...
    APP_PASSWORD   = matkhau  (tuy chon)
"""

from flask import Flask, request, jsonify, Response
from openai import OpenAI
import json, os, re, sqlite3, uuid
from datetime import datetime
from pathlib import Path

# ===== CONFIG =====
try:
    from config import LM_STUDIO_URL, REQUEST_TIMEOUT, GROQ_API_KEY, OPENROUTER_API_KEY
except ImportError:
    LM_STUDIO_URL      = "http://127.0.0.1:1234"
    REQUEST_TIMEOUT    = 120
    GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()
app = Flask(__name__)

PROVIDERS = {
    "lmstudio":   {"name": "LM Studio (Local)",       "base_url": f"{LM_STUDIO_URL}/v1",             "api_key": "lm-studio",        "default_model": "local-model"},
    "groq":       {"name": "Groq (Cloud - Fast)",      "base_url": "https://api.groq.com/openai/v1",  "api_key": GROQ_API_KEY,       "default_model": "llama3-70b-8192"},
    "openrouter": {"name": "OpenRouter (Multi-Model)", "base_url": "https://openrouter.ai/api/v1",    "api_key": OPENROUTER_API_KEY, "default_model": "google/gemini-pro-1.5"},
}
# ===== DATABASE - tu dong chon SQLite hoac Turso =====
TURSO_URL   = os.environ.get("TURSO_URL", "").strip()
TURSO_TOKEN = os.environ.get("TURSO_TOKEN", "").strip()
USE_TURSO   = bool(TURSO_URL and TURSO_TOKEN)

_CREATE_SESSIONS = """CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, title TEXT NOT NULL,
    system_prompt TEXT DEFAULT '', provider TEXT DEFAULT 'lmstudio',
    model TEXT DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""

_CREATE_MESSAGES_SQLITE = """CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"""

_CREATE_MESSAGES_TURSO = """CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"""


# --- SQLite (local) ---
_db_path = Path(os.environ.get("DB_PATH", "") or (Path(__file__).parent / "chat_history.db"))

def _sqlite_conn():
    c = sqlite3.connect(str(_db_path))
    c.row_factory = sqlite3.Row
    return c

def _sqlite_init():
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    with _sqlite_conn() as c:
        c.execute(_CREATE_SESSIONS)
        c.execute(_CREATE_MESSAGES_SQLITE)
        c.commit()


# --- Turso (cloud) ---
_turso_client = None

def _get_turso():
    global _turso_client
    if _turso_client is None:
        try:
            import libsql_client
            _turso_client = libsql_client.create_client_sync(
                url=TURSO_URL, auth_token=TURSO_TOKEN)
        except ImportError:
            raise RuntimeError(
                "Thu vien libsql-client chua duoc cai.\n"
                "Chay: pip install libsql-client")
    return _turso_client

def _turso_init():
    c = _get_turso()
    c.execute(_CREATE_SESSIONS)
    c.execute(_CREATE_MESSAGES_TURSO)

def _turso_rows(result):
    cols = [col.name for col in result.columns] if result.columns else []
    return [dict(zip(cols, row)) for row in (result.rows or [])]


# --- Interface thong nhat (dung cho ca SQLite va Turso) ---
def init_db():
    if USE_TURSO:
        _turso_init()
        print(f"  DB : Turso cloud ({TURSO_URL})")
    else:
        _sqlite_init()
        print(f"  DB : SQLite ({_db_path})")

def db_execute(sql, params=()):
    if USE_TURSO:
        _get_turso().execute(sql, list(params))
    else:
        with _sqlite_conn() as c:
            c.execute(sql, params)
            c.commit()

def db_fetchall(sql, params=()):
    if USE_TURSO:
        result = _get_turso().execute(sql, list(params))
        return _turso_rows(result)
    else:
        with _sqlite_conn() as c:
            rows = c.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

def db_fetchone(sql, params=()):
    rows = db_fetchall(sql, params)
    return rows[0] if rows else None

# Chay ngay khi module load (can cho gunicorn/Render)
init_db()

# ===== AUTH =====
@app.before_request
def check_login():
    if APP_PASSWORD:
        auth = request.authorization
        if not auth or auth.password != APP_PASSWORD:
            return Response('Nhập mật khẩu để tiếp tục.', 401, {'WWW-Authenticate': 'Basic realm="AI Apps"'})

# ===== HELPERS =====
def get_client(provider_key="lmstudio"):
    cfg = PROVIDERS.get(provider_key, PROVIDERS["lmstudio"])
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

def parse_reply(content):
    if not content: return ""
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    for tok in ["<|END_RESPONSE|>","<|end_of_turn|>","<|eot_id|>","<|endoftext|>","<|im_end|>"]:
        text = text.replace(tok, "")
    return text.strip()

# ===== GROQ RATE LIMITER =====
# Groq free: 8000 tokens/phut, 30 req/phut
import threading, time as _time
_groq_lock     = threading.Lock()
_groq_tokens   = 0
_groq_reqs     = 0
_groq_reset_at = 0.0
GROQ_MAX_TPM   = 7500   # gia tri an toan duoi gioi han 8000
GROQ_MAX_RPM   = 28

def _groq_wait(estimated_tokens=500):
    global _groq_tokens, _groq_reqs, _groq_reset_at
    with _groq_lock:
        now = _time.time()
        if now >= _groq_reset_at:           # Reset moi phut
            _groq_tokens   = 0
            _groq_reqs     = 0
            _groq_reset_at = now + 60.0
        wait = 0.0
        if _groq_tokens + estimated_tokens > GROQ_MAX_TPM:
            wait = max(wait, _groq_reset_at - now)
        if _groq_reqs >= GROQ_MAX_RPM:
            wait = max(wait, _groq_reset_at - now)
        if wait > 0:
            print(f"  [Groq RL] cho {wait:.1f}s (tokens={_groq_tokens}, reqs={_groq_reqs})")
        _groq_tokens += estimated_tokens
        _groq_reqs   += 1
    if wait > 0:
        _time.sleep(wait)

def _estimate_tokens(messages):
    return sum(len(m.get("content","")) // 3 for m in messages) + 200

def llm_call(messages, d, max_tokens=1024, temperature=0.7):
    try:
        provider = d.get('provider', 'lmstudio')
        model    = d.get('model') or PROVIDERS[provider]["default_model"]
        if provider == 'groq':
            _groq_wait(_estimate_tokens(messages) + max_tokens)
        resp = get_client(provider).chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        return parse_reply(resp.choices[0].message.content or "")
    except Exception as e:
        err = str(e)
        if '429' in err and 'groq' in d.get('provider',''):
            _time.sleep(15)
            try:
                resp = get_client(d.get('provider')).chat.completions.create(
                    model=d.get('model') or PROVIDERS[d.get('provider')]["default_model"],
                    messages=messages, max_tokens=max_tokens, temperature=temperature)
                return parse_reply(resp.choices[0].message.content or "")
            except Exception as e2:
                return f"Lỗi sau retry: {str(e2)}"
        return f"Lỗi API: {err}"

# ===== FILE PROCESSING =====
import base64, mimetypes

ALLOWED_EXT = {'.txt','.md','.py','.js','.ts','.json','.csv','.html','.css',
               '.xml','.yaml','.yml','.ini','.sh','.bat','.sql','.log',
               '.pdf','.png','.jpg','.jpeg','.gif','.webp'}
MAX_FILE_MB  = 5
MAX_TEXT_CHARS = 12000   # cat ngan neu qua dai

def _read_file_content(file_storage):
    """Doc noi dung file, tra ve (text|None, mime_type, ten_file)."""
    fname = file_storage.filename or "file"
    ext   = os.path.splitext(fname)[1].lower()
    if ext not in ALLOWED_EXT:
        return None, None, fname
    data  = file_storage.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, "too_large", fname
    mime  = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    # File van ban
    if mime.startswith("text/") or ext in {'.py','.js','.ts','.json','.csv',
            '.xml','.yaml','.yml','.ini','.sh','.bat','.sql','.log','.md'}:
        try:
            text = data.decode("utf-8", errors="replace")
            return text[:MAX_TEXT_CHARS], "text", fname
        except:
            return None, None, fname
    # File anh
    if mime.startswith("image/"):
        b64 = base64.b64encode(data).decode()
        return b64, mime, fname
    return None, None, fname

def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===== HTML =====
HTML = r"""<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Apps</title>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<style>
/* Markdown rendering trong bubble chat */
.bbl.md-content h1,.bbl.md-content h2,.bbl.md-content h3{font-weight:600;margin:.6em 0 .3em;line-height:1.3}
.bbl.md-content h1{font-size:1.1em}.bbl.md-content h2{font-size:1em}.bbl.md-content h3{font-size:.95em}
.bbl.md-content p{margin:.4em 0;line-height:1.65}
.bbl.md-content ul,.bbl.md-content ol{padding-left:1.4em;margin:.4em 0}
.bbl.md-content li{margin:.2em 0;line-height:1.6}
.bbl.md-content table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:.85em}
.bbl.md-content th,.bbl.md-content td{border:1px solid var(--bd2);padding:6px 10px;text-align:left}
.bbl.md-content th{background:var(--bg3);font-weight:600;color:var(--tx)}
.bbl.md-content td{background:var(--bg2);color:var(--tx2)}
.bbl.md-content tr:hover td{background:var(--bg3)}
.bbl.md-content code{background:var(--bg3);border:1px solid var(--bd);border-radius:4px;padding:1px 5px;font-family:var(--f1);font-size:.85em;color:var(--ac2)}
.bbl.md-content pre{background:var(--bg0);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;overflow-x:auto;margin:.5em 0}
.bbl.md-content pre code{background:none;border:none;padding:0;font-size:.82em;color:var(--tx)}
.bbl.md-content blockquote{border-left:3px solid var(--ac);padding:.3em .8em;margin:.4em 0;color:var(--tx2);background:var(--bg2);border-radius:0 6px 6px 0}
.bbl.md-content strong{font-weight:600;color:var(--tx)}
.bbl.md-content em{font-style:italic;color:var(--tx2)}
.bbl.md-content hr{border:none;border-top:1px solid var(--bd);margin:.6em 0}
.bbl.md-content a{color:var(--ac2);text-decoration:underline}
:root{
  --bg0:#09090b;--bg1:#111115;--bg2:#18181c;--bg3:#222228;
  --bd:#2e2e36;--bd2:#3a3a44;
  --tx:#f0eeff;--tx2:#9896aa;--tx3:#5a5870;
  --ac:#8b74ff;--ac2:#b09eff;--ac3:#6455dd;
  --gr:#34d399;--rd:#f87171;--am:#fbbf24;
  --f0:'Geist',sans-serif;--f1:'Geist Mono',monospace;
  --r:10px;--r2:14px;
}
[data-theme="light"]{
  --bg0:#f4f3fa;--bg1:#ffffff;--bg2:#f0eff8;--bg3:#e8e7f4;
  --bd:#d4d2e8;--bd2:#c0bedd;
  --tx:#1a1826;--tx2:#5a5870;--tx3:#9896aa;
  --ac:#7c6af7;--ac2:#5a4de0;--ac3:#9e8fff;
  --gr:#059669;--rd:#dc2626;--am:#d97706;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg0);color:var(--tx);font-family:var(--f0);height:100vh;display:flex;flex-direction:column;overflow:hidden}
.topbar{height:52px;background:var(--bg1);border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;padding:0 16px;flex-shrink:0;gap:12px;z-index:50}
.logo{font-family:var(--f1);font-size:.95rem;color:var(--ac2);font-weight:500;white-space:nowrap}
.logo em{color:var(--tx2);font-style:normal}
.topbar-mid{display:flex;align-items:center;gap:8px;flex:1;min-width:0;justify-content:center}
.topbar-right{display:flex;align-items:center;gap:8px;flex-shrink:0}
select.sel{background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:7px;padding:5px 8px;font-size:.8rem;outline:none;font-family:var(--f0);max-width:185px}
select.sel:focus{border-color:var(--ac)}
.badge{display:flex;align-items:center;gap:6px;background:var(--bg2);border:1px solid var(--bd);padding:4px 10px;border-radius:20px;font-size:.75rem;font-family:var(--f1);color:var(--tx2);white-space:nowrap}
.dot{width:7px;height:7px;border-radius:50%;background:var(--tx3);flex-shrink:0}
.dot.on{background:var(--gr);box-shadow:0 0 7px var(--gr);animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}
.icon-btn{background:var(--bg2);border:1px solid var(--bd);color:var(--tx2);width:32px;height:32px;border-radius:7px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:.9rem;transition:all .15s}
.icon-btn:hover{border-color:var(--ac);color:var(--ac2)}
.layout{display:flex;flex:1;overflow:hidden}
.apnav{width:56px;background:var(--bg1);border-right:1px solid var(--bd);display:flex;flex-direction:column;align-items:center;padding:10px 0;gap:4px;flex-shrink:0}
.anv{width:40px;height:40px;border-radius:9px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:1.05rem;color:var(--tx3);transition:all .15s;position:relative}
.anv:hover{background:var(--bg3);color:var(--tx)}
.anv.active{background:var(--ac3);color:#fff;box-shadow:0 2px 10px rgba(139,116,255,.3)}
.anv .tip{position:absolute;left:52px;top:50%;transform:translateY(-50%);background:var(--bg3);border:1px solid var(--bd2);color:var(--tx);padding:4px 10px;border-radius:6px;font-size:.78rem;white-space:nowrap;pointer-events:none;opacity:0;transition:opacity .1s;z-index:99}
.anv:hover .tip{opacity:1}
.anv-sep{width:32px;height:1px;background:var(--bd);margin:4px 0}
.hpanel{width:232px;background:var(--bg1);border-right:1px solid var(--bd);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0;transition:width .2s ease}
.hpanel.hidden{width:0;border:none}
.hp-head{padding:10px 12px;border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.hp-title{font-size:.72rem;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--tx3)}
.hp-body{flex:1;overflow-y:auto;padding:4px}
.hg-label{font-size:.69rem;color:var(--tx3);padding:8px 8px 3px;font-weight:500;letter-spacing:.3px}
.hi{padding:7px 10px;border-radius:7px;cursor:pointer;font-size:.81rem;color:var(--tx2);transition:all .15s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border:1px solid transparent;margin-bottom:1px}
.hi:hover{background:var(--bg2);color:var(--tx)}
.hi.active{background:var(--bg3);color:var(--tx);border-color:var(--bd2)}
.hi-time{font-size:.67rem;color:var(--tx3);margin-top:1px}
.h-empty{padding:24px 12px;text-align:center;color:var(--tx3);font-size:.8rem;line-height:1.6}
.main{flex:1;overflow:hidden;display:flex;flex-direction:column}
.panel{display:none;flex:1;overflow:hidden;flex-direction:column;animation:fi .18s ease}
.panel.active{display:flex}
@keyframes fi{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
.chat-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden}
.chat-toolbar{padding:8px 16px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:8px;flex-shrink:0;background:var(--bg1)}
.chat-toolbar span{font-size:.78rem;color:var(--tx3);white-space:nowrap}
.chat-toolbar input{flex:1;background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:7px;padding:5px 10px;font-size:.82rem;font-family:var(--f0);outline:none;transition:border-color .15s}
.chat-toolbar input:focus{border-color:var(--ac)}
.chat-msgs{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.mrow{display:flex;gap:10px;animation:fi .15s ease}
.mrow.user{justify-content:flex-end}
.av{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.8rem;flex-shrink:0;margin-top:2px}
.av.ai{background:linear-gradient(135deg,var(--ac3),var(--ac));color:#fff}
.av.user{background:var(--bg3);border:1px solid var(--bd2)}
.bbl{padding:9px 13px;border-radius:13px;font-size:.87rem;line-height:1.65;max-width:76%;white-space:pre-wrap;word-break:break-word}
.mrow.user .bbl{background:var(--ac);color:#fff;border-radius:13px 13px 3px 13px}
.mrow.ai .bbl{background:var(--bg2);border:1px solid var(--bd);border-radius:13px 13px 13px 3px}
.mrow.sys{justify-content:center}
.mrow.sys .bbl{background:transparent;color:var(--tx3);font-size:.77rem;font-style:italic;border:none;padding:3px 0}
.msg-t{font-size:.67rem;color:var(--tx3);margin-top:3px;text-align:right}
.tyd{display:flex;gap:4px;align-items:center;padding:2px 0}
.tyd span{width:6px;height:6px;border-radius:50%;background:var(--tx3);animation:td 1s infinite}
.tyd span:nth-child(2){animation-delay:.15s}.tyd span:nth-child(3){animation-delay:.3s}
@keyframes td{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
.chat-footer{padding:10px 16px;border-top:1px solid var(--bd);background:var(--bg1);flex-shrink:0}
.cin-wrap{display:flex;gap:8px;align-items:flex-end}
.cin-wrap textarea{flex:1;background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:10px;padding:9px 12px;font-family:var(--f0);font-size:.87rem;resize:none;outline:none;min-height:40px;max-height:130px;line-height:1.5;transition:border-color .15s}
.cin-wrap textarea:focus{border-color:var(--ac)}
.sbtn{width:40px;height:40px;background:var(--ac);border:none;border-radius:10px;color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:.95rem;transition:all .15s;flex-shrink:0}
.sbtn:hover{background:var(--ac2)}.sbtn:disabled{opacity:.5;cursor:not-allowed}
.chint{font-size:.71rem;color:var(--tx3);margin-top:5px;text-align:center}
.twrap{flex:1;overflow-y:auto;padding:22px}
.tinner{max-width:800px;margin:0 auto}
.th2{font-size:1.1rem;font-weight:600;margin-bottom:3px;display:flex;align-items:center;gap:8px}
.tsub{color:var(--tx2);font-size:.82rem;margin-bottom:16px}
.card{background:var(--bg1);border:1px solid var(--bd);border-radius:var(--r2);padding:15px;margin-bottom:10px}
.lbl{font-size:.71rem;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:var(--tx3);display:block;margin-bottom:5px}
textarea.inp,input.inp,select.inp{width:100%;background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:8px;padding:8px 10px;font-family:var(--f0);font-size:.87rem;resize:vertical;outline:none;transition:border-color .15s}
textarea.inp{min-height:88px}
textarea.inp:focus,input.inp:focus,select.inp:focus{border-color:var(--ac)}
.row{display:flex;gap:7px;margin-top:9px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 13px;border-radius:8px;border:none;font-family:var(--f0);font-size:.83rem;font-weight:500;cursor:pointer;transition:all .15s}
.bp{background:var(--ac);color:#fff}.bp:hover{background:var(--ac2)}.bp:disabled{opacity:.5;cursor:not-allowed}
.bg{background:var(--bg2);color:var(--tx);border:1px solid var(--bd)}.bg:hover{border-color:var(--ac);color:var(--ac2)}
.out{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:12px;font-family:var(--f1);font-size:.82rem;line-height:1.7;color:var(--tx);white-space:pre-wrap;word-break:break-word;min-height:80px;max-height:440px;overflow-y:auto}
.out.ph{color:var(--tx3);font-style:italic}
.lgrow{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:end;margin-bottom:10px}
.swp{height:34px;width:34px;display:flex;align-items:center;justify-content:center;background:var(--bg2);border:1px solid var(--bd);border-radius:7px;cursor:pointer;color:var(--tx2);transition:all .15s}
.swp:hover{border-color:var(--ac);color:var(--ac2)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}
.chip{display:inline-flex;align-items:center;gap:3px;padding:3px 8px;border-radius:20px;font-size:.73rem;background:var(--bg3);border:1px solid var(--bd);color:var(--tx2);cursor:pointer;transition:all .15s}
.chip:hover{border-color:var(--ac);color:var(--ac2)}
.sp{display:inline-block;width:12px;height:12px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:3px}
</style>
</head>
<body>
<div class="topbar">
  <div class="logo">AI<em>.apps</em></div>
  <div class="topbar-mid">
    <select id="gp" class="sel" onchange="fetchModels()"></select>
    <select id="gm" class="sel"></select>
  </div>
  <div class="topbar-right">
    <div class="badge"><div class="dot" id="sdot"></div><span id="stxt">Kết nối...</span></div>
    <div class="icon-btn" onclick="toggleTheme()" title="Đổi giao diện">🌙</div>
    <div class="icon-btn" onclick="toggleHP()" title="Lịch sử">📋</div>
  </div>
</div>

<div class="layout">
  <nav class="apnav">
    <div class="anv active" id="anav-chat" onclick="switchApp('chat')">💬<span class="tip">Chatbot</span></div>
    <div class="anv" id="anav-optimizer" onclick="switchApp('optimizer')">✨<span class="tip">Tối ưu Prompt</span></div>
    <div class="anv-sep"></div>
    <div class="anv" id="anav-translate" onclick="switchApp('translate')">🌐<span class="tip">Dịch thuật</span></div>
    <div class="anv" id="anav-review" onclick="switchApp('review')">🔍<span class="tip">Code Review</span></div>
    <div class="anv" id="anav-summary" onclick="switchApp('summary')">📄<span class="tip">Tóm tắt</span></div>
    <div class="anv" id="anav-mockdata" onclick="switchApp('mockdata')">🗄️<span class="tip">Mock Data</span></div>
    <div class="anv" id="anav-terminal" onclick="switchApp('terminal')">⌨️<span class="tip">Terminal</span></div>
  </nav>

  <div class="hpanel" id="hpanel">
    <div class="hp-head">
      <span class="hp-title">Lịch sử chat</span>
      <div class="icon-btn" onclick="newSession()" title="Tạo mới">✏️</div>
    </div>
    <div class="hp-body" id="hlist"></div>
  </div>

  <div class="main">

    <div class="panel active" id="panel-chat">
      <div class="chat-wrap">
        <div class="chat-toolbar">
          <span>System:</span>
          <input id="chat-system" placeholder="Bạn là trợ lý AI hữu ích, trả lời bằng tiếng Việt.">
        </div>
        <div class="chat-msgs" id="chat-msgs">
          <div class="mrow sys"><div class="bbl">Chọn cuộc trò chuyện bên trái hoặc nhấn ✏️ để bắt đầu mới</div></div>
        </div>
        <div class="chat-footer">
          <div id="file-preview" style="display:none;padding:6px 0 4px;display:flex;align-items:center;gap:8px"></div>
          <div class="cin-wrap">
            <label class="fbtn" title="Đính kèm file" style="width:36px;height:36px;background:var(--bg2);border:1px solid var(--bd);border-radius:9px;display:flex;align-items:center;justify-content:center;cursor:pointer;color:var(--tx2);font-size:.95rem;flex-shrink:0;transition:all .15s" onmouseover="this.style.borderColor='var(--ac)'" onmouseout="this.style.borderColor='var(--bd)'">
              📎<input type="file" id="file-input" style="display:none" onchange="onFileSelect(this)" accept=".txt,.md,.py,.js,.ts,.json,.csv,.html,.css,.xml,.yaml,.yml,.sh,.sql,.log,.pdf,.png,.jpg,.jpeg,.gif,.webp">
            </label>
            <textarea id="chat-input" rows="1" placeholder="Nhập tin nhắn... (Enter gửi, Shift+Enter xuống dòng)"
              oninput="arz(this)"
              onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
            <button class="sbtn" id="sbtn" onclick="sendChat()">➤</button>
          </div>
          <div class="chint">Enter gửi · Shift+Enter xuống dòng · 📎 Đính kèm file</div>
        </div>
      </div>
    </div>

    <div class="panel" id="panel-optimizer">
      <div class="twrap"><div class="tinner">
        <div class="th2">✨ Tối ưu Prompt</div>
        <div class="tsub">Biến ý tưởng thô thành prompt chuyên nghiệp</div>
        <div class="card">
          <label class="lbl">Ý tưởng ban đầu</label>
          <textarea class="inp" id="opt-in" rows="4" placeholder="VD: Viết bài post facebook bán áo thun mùa hè"></textarea>
          <div class="row"><button class="btn bp" id="opt-btn" onclick="tc('optimize_prompt',{text:g('opt-in')},'opt-btn','opt-out')">✨ Tối ưu hóa</button></div>
        </div>
        <div class="card">
          <label class="lbl">Prompt đã tối ưu</label>
          <div class="out ph" id="opt-out">Prompt chuyên nghiệp sẽ hiện ở đây...</div>
          <div class="row"><button class="btn bg" onclick="cp('opt-out')">📋 Sao chép</button></div>
        </div>
      </div></div>
    </div>

    <div class="panel" id="panel-translate">
      <div class="twrap"><div class="tinner">
        <div class="th2">🌐 Dịch thuật</div>
        <div class="tsub">Dịch văn bản đa ngôn ngữ chuyên sâu</div>
        <div class="card">
          <div class="lgrow">
            <div><label class="lbl">Từ</label>
              <select class="inp" id="tr-src" style="resize:none">
                <option value="English">English</option><option value="Vietnamese">Tiếng Việt</option>
                <option value="Chinese">Tiếng Trung</option><option value="Japanese">Tiếng Nhật</option>
                <option value="Korean">Tiếng Hàn</option><option value="French">Tiếng Pháp</option>
                <option value="German">Tiếng Đức</option>
              </select>
            </div>
            <div style="padding-top:18px"><div class="swp" onclick="swapL()">⇄</div></div>
            <div><label class="lbl">Sang</label>
              <select class="inp" id="tr-dst" style="resize:none">
                <option value="Vietnamese">Tiếng Việt</option><option value="English">English</option>
                <option value="Chinese">Tiếng Trung</option><option value="Japanese">Tiếng Nhật</option>
                <option value="Korean">Tiếng Hàn</option><option value="French">Tiếng Pháp</option>
                <option value="German">Tiếng Đức</option>
              </select>
            </div>
          </div>
          <label class="lbl">Văn bản</label>
          <textarea class="inp" id="tr-in" rows="5" placeholder="Nhập văn bản cần dịch..."></textarea>
          <div class="row">
            <button class="btn bp" id="tr-btn" onclick="tc('translate',{text:g('tr-in'),src:g('tr-src'),dst:g('tr-dst')},'tr-btn','tr-out')">🌐 Dịch</button>
            <button class="btn bg" onclick="cp('tr-out')">📋 Sao chép</button>
          </div>
        </div>
        <div class="card"><label class="lbl">Kết quả</label><div class="out ph" id="tr-out">Bản dịch sẽ hiện ở đây...</div></div>
      </div></div>
    </div>

    <div class="panel" id="panel-review">
      <div class="twrap"><div class="tinner">
        <div class="th2">🔍 Code Review</div>
        <div class="tsub">Phát hiện lỗi, tối ưu hiệu năng và bảo mật</div>
        <div class="card">
          <label class="lbl">Code cần review</label>
          <textarea class="inp" id="rv-in" rows="10" placeholder="Paste code vào đây..." style="font-family:var(--f1);font-size:.81rem"></textarea>
          <div class="row">
            <button class="btn bp" id="rv-btn" onclick="tc('review',{code:g('rv-in')},'rv-btn','rv-out')">🔍 Review</button>
            <button class="btn bg" onclick="cp('rv-out')">📋 Sao chép</button>
          </div>
        </div>
        <div class="card"><label class="lbl">Nhận xét</label><div class="out ph" id="rv-out">Kết quả review...</div></div>
      </div></div>
    </div>

    <div class="panel" id="panel-summary">
      <div class="twrap"><div class="tinner">
        <div class="th2">📄 Tóm tắt văn bản</div>
        <div class="tsub">Tóm tắt nhanh báo cáo, tài liệu dài</div>
        <div class="card">
          <label class="lbl">Văn bản</label>
          <textarea class="inp" id="sm-in" rows="9" placeholder="Paste văn bản cần tóm tắt..."></textarea>
          <div class="row">
            <button class="btn bp" id="sm-btn" onclick="tc('summary',{text:g('sm-in')},'sm-btn','sm-out')">📄 Tóm tắt</button>
            <button class="btn bg" onclick="cp('sm-out')">📋 Sao chép</button>
          </div>
        </div>
        <div class="card"><label class="lbl">Kết quả</label><div class="out ph" id="sm-out">Bản tóm tắt sẽ hiện ở đây...</div></div>
      </div></div>
    </div>

    <div class="panel" id="panel-mockdata">
      <div class="twrap"><div class="tinner">
        <div class="th2">🗄️ Sinh Mock Data JSON</div>
        <div class="tsub">Tạo dữ liệu mẫu cho testing</div>
        <div class="card">
          <label class="lbl">Mẫu nhanh</label>
          <div class="chips">
            <span class="chip" onclick="si('mk-in','5 users: email, role, avatar, created_at')">👤 Users</span>
            <span class="chip" onclick="si('mk-in','5 game items: name, type, damage, rarity, price')">⚔️ Game Items</span>
            <span class="chip" onclick="si('mk-in','5 products: name, price, category, stock, rating')">📦 Products</span>
            <span class="chip" onclick="si('mk-in','5 blog posts: title, author, content, tags, date')">📝 Blog</span>
            <span class="chip" onclick="si('mk-in','5 API endpoints: method, path, description, params')">🔌 API Docs</span>
          </div>
          <label class="lbl">Mô tả schema</label>
          <input class="inp" id="mk-in" placeholder="VD: 5 nhân vật game RPG với tên, class, level, kỹ năng">
          <div class="row">
            <button class="btn bp" id="mk-btn" onclick="tc('mockdata',{schema:g('mk-in')},'mk-btn','mk-out')">🗄️ Tạo dữ liệu</button>
            <button class="btn bg" onclick="cp('mk-out')">📋 Sao chép JSON</button>
          </div>
        </div>
        <div class="card"><label class="lbl">JSON Output</label><div class="out ph" id="mk-out" style="font-family:var(--f1);font-size:.79rem">Dữ liệu JSON sẽ hiện ở đây...</div></div>
      </div></div>
    </div>

    <div class="panel" id="panel-terminal">
      <div class="twrap"><div class="tinner">
        <div class="th2">⌨️ Trợ lý Terminal</div>
        <div class="tsub">Giải thích lỗi, đề xuất lệnh</div>
        <div class="card">
          <label class="lbl">Error log hoặc yêu cầu</label>
          <textarea class="inp" id="tm-in" rows="7" placeholder="Paste error log hoặc mô tả việc cần làm..." style="font-family:var(--f1);font-size:.81rem"></textarea>
          <div class="row">
            <button class="btn bp" id="tm-btn" onclick="tc('terminal',{input:g('tm-in')},'tm-btn','tm-out')">⌨️ Phân tích</button>
            <button class="btn bg" onclick="cp('tm-out')">📋 Sao chép</button>
          </div>
        </div>
        <div class="card"><label class="lbl">Kết quả</label><div class="out ph" id="tm-out">Kết quả phân tích...</div></div>
      </div></div>
    </div>

  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const g = id => ($(id)?.value||$(id)?.textContent||'').trim();
const si = (id,v) => { $(id).value=v; };
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const arz = el => { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,130)+'px'; };

let curSid=null, chatMsgs=[], hpVisible=true;

function toggleTheme(){
  const h=document.documentElement, dark=h.getAttribute('data-theme')==='dark';
  h.setAttribute('data-theme',dark?'light':'dark');
}
function toggleHP(){ hpVisible=!hpVisible; $('hpanel').classList.toggle('hidden',!hpVisible); }
function swapL(){ const s=$('tr-src'),d=$('tr-dst'); [s.value,d.value]=[d.value,s.value]; }

function switchApp(name){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.anv').forEach(n=>n.classList.remove('active'));
  $('panel-'+name).classList.add('active');
  $('anav-'+name).classList.add('active');
  $('hpanel').style.visibility = name==='chat'?'':'hidden';
  $('hpanel').style.width      = name==='chat'&&hpVisible?'':'0';
}

async function cp(id){
  const t=$(id).innerText;
  try{ await navigator.clipboard.writeText(t); }
  catch(e){ const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();document.execCommand('copy');document.body.removeChild(a); }
}

// Providers & Models
async function initProviders(){
  try{
    const d=await(await fetch('/api/providers')).json();
    const sel=$('gp'); sel.innerHTML='';
    for(const[k,v]of Object.entries(d)) sel.innerHTML+=`<option value="${k}">${v.name}</option>`;
    await fetchModels();
  }catch(e){}
}
async function fetchModels(){
  const p=$('gp').value, gm=$('gm');
  gm.innerHTML='<option>Đang tải...</option>';
  $('sdot').className='dot'; $('stxt').innerText='Đang tải...';
  try{
    const d=await(await fetch(`/api/models?provider=${p}`)).json();
    gm.innerHTML='';
    if(d.models?.length){
      d.models.forEach(m=>gm.innerHTML+=`<option value="${m}"${m===d.default?' selected':''}>${m}</option>`);
      $('sdot').className='dot on'; $('stxt').innerText='Sẵn sàng';
    } else {
      gm.innerHTML=`<option value="${d.default}">${d.default}</option>`;
      $('sdot').className='dot'; $('stxt').innerText='Kiểm tra API Key';
    }
  }catch(e){ $('stxt').innerText='Lỗi kết nối'; }
}

// History
async function loadHistory(){
  try{
    const ss=await(await fetch('/api/sessions')).json();
    renderHistory(ss);
  }catch(e){}
}
function renderHistory(ss){
  const el=$('hlist');
  if(!ss.length){ el.innerHTML='<div class="h-empty">Chưa có cuộc trò chuyện<br>Nhấn ✏️ để bắt đầu</div>'; return; }
  const today=new Date().toISOString().slice(0,10);
  const yest=new Date(Date.now()-86400000).toISOString().slice(0,10);
  const groups={};
  ss.forEach(s=>{
    const d=(s.updated_at||'').slice(0,10);
    const lbl=d===today?'Hôm nay':d===yest?'Hôm qua':d||'Cũ hơn';
    if(!groups[lbl])groups[lbl]=[];
    groups[lbl].push(s);
  });
  el.innerHTML='';
  for(const[lbl,items]of Object.entries(groups)){
    el.innerHTML+=`<div class="hg-label">${lbl}</div>`;
    items.forEach(s=>{
      const ac=s.id===curSid?'active':'';
      el.innerHTML+=`<div class="hi ${ac}" onclick="loadSess('${s.id}')" title="${esc(s.title)}">
        ${esc(s.title)}<div class="hi-time">${(s.updated_at||'').slice(11,16)}</div></div>`;
    });
  }
}

async function newSession(){
  curSid=null; chatMsgs=[];
  $('chat-msgs').innerHTML='<div class="mrow sys"><div class="bbl">Cuộc trò chuyện mới ✨</div></div>';
  $('chat-system').value='';
  await loadHistory();
}

async function loadSess(id){
  try{
    const d=await(await fetch(`/api/sessions/${id}`)).json();
    curSid=d.session.id; chatMsgs=d.messages.map(m=>({role:m.role,content:m.content}));
    $('chat-system').value=d.session.system_prompt||'';
    $('chat-msgs').innerHTML='';
    d.messages.forEach(m=>addBbl(m.role,m.content,(m.created_at||'').slice(11,16)));
    $('chat-msgs').scrollTop=$('chat-msgs').scrollHeight;
    await loadHistory();
  }catch(e){ console.error(e); }
}

function addBbl(role,content,time=''){
  const m=$('chat-msgs');
  if(role==='sys'||role==='system'){
    m.innerHTML+=`<div class="mrow sys"><div class="bbl">${esc(content)}</div></div>`;
    return;
  }
  const u=role==='user';
  const av=u?`<div class="av user">👤</div>`:`<div class="av ai">🤖</div>`;
  // User: escape HTML thuần | AI: render markdown
  const rendered = u
    ? esc(content)
    : (typeof marked !== 'undefined'
        ? marked.parse(content, {breaks:true, gfm:true})
        : esc(content));
  const mdClass = u ? '' : ' md-content';
  m.innerHTML+=`<div class="mrow ${u?'user':'ai'}">${u?'':av}
    <div><div class="bbl${mdClass}">${rendered}</div>${time?`<div class="msg-t">${time}</div>`:''}</div>
    ${u?av:''}</div>`;
  m.scrollTop=m.scrollHeight;
}

let _fileInfo = null;   // File dang duoc dinh kem

function onFileSelect(input) {
  const file = input.files[0];
  if (!file) return;
  const prev = $('file-preview');

  // Hien thi preview
  prev.style.display = 'flex';
  prev.innerHTML = `
    <span style="font-size:.8rem;background:var(--bg3);border:1px solid var(--bd);border-radius:6px;
      padding:3px 8px;color:var(--tx2);display:flex;align-items:center;gap:6px">
      📄 ${file.name} (${(file.size/1024).toFixed(1)}KB)
      <span onclick="clearFile()" style="cursor:pointer;color:var(--rd);font-weight:600" title="Xoa file">✕</span>
    </span>
    <span id="upload-status" style="font-size:.75rem;color:var(--tx3)">Dang tai...</span>`;

  // Upload len server
  const fd = new FormData();
  fd.append('file', file);
  fetch('/api/upload', {method:'POST', body:fd})
    .then(r => r.json())
    .then(d => {
      if (d.error) {
        $('upload-status').textContent = '❌ ' + d.error;
        $('upload-status').style.color = 'var(--rd)';
        _fileInfo = null;
      } else {
        $('upload-status').textContent = '✅ San sang';
        $('upload-status').style.color = 'var(--gr)';
        _fileInfo = d;
      }
    })
    .catch(e => {
      $('upload-status').textContent = '❌ Loi upload';
      _fileInfo = null;
    });
  input.value = '';
}

function clearFile() {
  _fileInfo = null;
  const prev = $('file-preview');
  prev.style.display = 'none';
  prev.innerHTML = '';
}

async function sendChat(){
  const inp=$('chat-input'), msg=inp.value.trim();
  if(!msg && !_fileInfo) return;
  if(!curSid){
    const title = msg || (_fileInfo ? 'File: ' + _fileInfo.name : 'Cuoc tro chuyen moi');
    const r=await fetch('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title:title.length>40?title.slice(0,40)+'...':title,
        system_prompt:$('chat-system').value.trim(),provider:$('gp').value,model:$('gm').value})});
    curSid=(await r.json()).id;
  }
  const now=new Date().toLocaleTimeString('vi',{hour:'2-digit',minute:'2-digit'});
  // Hien thi tin nhan user (co the co label file)
  const userLabel = msg + (_fileInfo ? `
📎 ${_fileInfo.name}` : '');
  addBbl('user', userLabel, now);
  chatMsgs.push({role:'user', content: msg || `[File: ${_fileInfo?.name}]`});
  inp.value=''; inp.style.height='auto';
  const fileToSend = _fileInfo;
  clearFile();
  $('sbtn').disabled=true;
  const m=$('chat-msgs');
  m.innerHTML+=`<div class="mrow ai" id="trow"><div class="av ai">🤖</div><div class="bbl"><div class="tyd"><span></span><span></span><span></span></div></div></div>`;
  m.scrollTop=m.scrollHeight;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg, history:chatMsgs, file:fileToSend,
        system:$('chat-system').value.trim()||'Ban la tro ly AI huu ich.',
        provider:$('gp').value, model:$('gm').value, session_id:curSid})});
    const d=await r.json();
    $('trow')?.remove();
    const reply=d.reply||'(Khong co phan hoi)';
    const t2=new Date().toLocaleTimeString('vi',{hour:'2-digit',minute:'2-digit'});
    addBbl('ai',reply,t2);
    chatMsgs.push({role:'assistant',content:reply});
    await loadHistory();
  }catch(e){ $('trow')?.remove(); addBbl('sys','❌ Loi: '+e.message); }
  $('sbtn').disabled=false;
}

async function tc(ep,payload,btnId,outId){
  if(Object.values(payload).some(v=>!v)) return;
  const btn=$(btnId), out=$(outId);
  btn.dataset.orig=btn.innerHTML;
  btn.innerHTML='<span class="sp"></span> Đang xử lý...';
  btn.disabled=true;
  out.textContent='⏳ Đang xử lý...'; out.classList.remove('ph');
  payload.provider=$('gp').value; payload.model=$('gm').value;
  try{
    const r=await fetch('/api/'+ep,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    out.textContent=d.result||d.error||'(Không có kết quả)';
  }catch(e){ out.textContent='❌ Lỗi: '+e.message; }
  btn.innerHTML=btn.dataset.orig; btn.disabled=false;
}

window.onload=async()=>{
  await initProviders();
  await loadHistory();
  const ss=await(await fetch('/api/sessions')).json();
  if(ss.length) await loadSess(ss[0].id);
};
</script>
</body>
</html>"""

# ===== ROUTES =====
@app.route('/')
def index(): return HTML

@app.route('/api/providers')
def api_providers():
    return jsonify({k:{"name":v["name"],"default":v["default_model"]} for k,v in PROVIDERS.items()})

@app.route('/api/models')
def api_models():
    p = request.args.get('provider','lmstudio')
    cfg = PROVIDERS.get(p, PROVIDERS["lmstudio"])
    try:
        ids = [m.id for m in get_client(p).models.list().data]
        d = cfg["default_model"]
        if d in ids: ids.insert(0, ids.pop(ids.index(d)))
        return jsonify({"models":ids,"default":d})
    except:
        return jsonify({"models":[],"default":cfg["default_model"]})

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    rows = db_fetchall("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 100")
    return jsonify(rows)

@app.route('/api/sessions', methods=['POST'])
def create_session():
    d=request.json; sid=str(uuid.uuid4()); ts=now_str()
    db_execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
        (sid, d.get('title','Cuoc tro chuyen moi'), d.get('system_prompt',''),
         d.get('provider','lmstudio'), d.get('model',''), ts, ts))
    return jsonify({"id":sid,"title":d.get('title'),"created_at":ts,"updated_at":ts})

@app.route('/api/sessions/<sid>', methods=['GET'])
def get_session(sid):
    s  = db_fetchone("SELECT * FROM sessions WHERE id=?", (sid,))
    ms = db_fetchall("SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,))
    if not s: return jsonify({"error":"Not found"}), 404
    return jsonify({"session": s, "messages": ms})

@app.route('/api/sessions/<sid>', methods=['DELETE'])
def delete_session(sid):
    db_execute("DELETE FROM messages WHERE session_id=?", (sid,))
    db_execute("DELETE FROM sessions WHERE id=?", (sid,))
    return jsonify({"ok": True})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Nhan file, tra ve noi dung da xu ly de chat."""
    if 'file' not in request.files:
        return jsonify({"error": "Khong co file"}), 400
    f = request.files['file']
    content_data, ctype, fname = _read_file_content(f)
    if ctype == "too_large":
        return jsonify({"error": f"File qua lon (toi da {MAX_FILE_MB}MB)"}), 400
    if content_data is None:
        return jsonify({"error": f"Dinh dang khong ho tro: {fname}"}), 400
    return jsonify({"content": content_data, "type": ctype,
                    "name": fname, "is_image": ctype not in ("text","too_large")})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    d         = request.json
    sid       = d.get('session_id')
    user_msg  = d.get('message', '')
    file_info = d.get('file')         # {"content":..., "type":..., "name":..., "is_image":...}
    provider  = d.get('provider', 'lmstudio')
    system    = d.get("system", "Ban la tro ly AI huu ich.")

    msgs = [{"role": "system", "content": system}]
    msgs.extend(d.get("history", []))

    # Xu ly file dinh kem
    if file_info:
        fname = file_info.get("name", "file")
        if file_info.get("is_image"):
            # Anh: dung vision API (ho tro LM Studio vision model, OpenRouter)
            mime = file_info["type"]
            b64  = file_info["content"]
            user_content = [
                {"type": "text",      "text": user_msg or f"Phan tich anh: {fname}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]
            msgs.append({"role": "user", "content": user_content})
        else:
            # Van ban/code: nhung noi dung vao message
            ftext  = file_info["content"]
            block  = f"[File: {fname}]\n```\n{ftext}\n```"
            prompt = block + "\n\n" + user_msg if user_msg else block + "\n\nHay phan tich noi dung file nay."
            msgs.append({"role": "user", "content": prompt})
    else:
        msgs.append({"role": "user", "content": user_msg})

    reply = llm_call(msgs, d, max_tokens=1024, temperature=0.7)

    if sid:
        ts = now_str()
        label = user_msg + (f" [+{file_info['name']}]" if file_info else "")
        db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)",
                   (sid, 'user', label, ts))
        db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)",
                   (sid, 'assistant', reply, ts))
        db_execute("UPDATE sessions SET updated_at=? WHERE id=?", (ts, sid))
    return jsonify({"reply": reply})

@app.route('/api/optimize_prompt', methods=['POST'])
def api_optimize():
    d=request.json
    r=llm_call([{"role":"system","content":"Bạn là chuyên gia Prompt Engineering. Viết lại thành prompt chuyên nghiệp, chi tiết. Chỉ trả về prompt, không giải thích."},
                {"role":"user","content":f"Yêu cầu: {d['text']}"}],d,max_tokens=1024,temperature=0.5)
    return jsonify({"result":r})

@app.route('/api/translate', methods=['POST'])
def api_translate():
    d=request.json
    r=llm_call([{"role":"system","content":f"Translate to {d['dst']}. Output ONLY the translation."},
                {"role":"user","content":f"Translate from {d['src']}:\n{d['text']}"}],d,max_tokens=1024,temperature=0.3)
    return jsonify({"result":r})

@app.route('/api/review', methods=['POST'])
def api_review():
    d=request.json
    r=llm_call([{"role":"system","content":"You are a senior developer. Review code, find bugs and suggest improvements in Vietnamese."},
                {"role":"user","content":f"Review:\n```\n{d['code']}\n```"}],d,max_tokens=2048,temperature=0.2)
    return jsonify({"result":r})

@app.route('/api/summary', methods=['POST'])
def api_summary():
    d=request.json
    r=llm_call([{"role":"system","content":"Tóm tắt súc tích, đầy đủ bằng tiếng Việt."},
                {"role":"user","content":f"Tóm tắt:\n{d['text'][:8000]}"}],d,max_tokens=1024,temperature=0.3)
    return jsonify({"result":r})

@app.route('/api/mockdata', methods=['POST'])
def api_mockdata():
    d=request.json
    r=llm_call([{"role":"system","content":"Output ONLY a valid JSON array. No markdown, no backticks."},
                {"role":"user","content":f"Generate JSON: {d['schema']}"}],d,max_tokens=2048,temperature=0.8)
    return jsonify({"result":r})

@app.route('/api/terminal', methods=['POST'])
def api_terminal():
    d=request.json
    r=llm_call([{"role":"system","content":"Chuyên gia DevOps. Giải thích lỗi hoặc đưa lệnh terminal chính xác bằng tiếng Việt."},
                {"role":"user","content":d['input']}],d,max_tokens=1024,temperature=0.2)
    return jsonify({"result":r})

if __name__=='__main__':
    # init_db() đã chạy ở module level — không cần gọi lại
    import socket
    ip=socket.gethostbyname(socket.gethostname())
    print(f"\n{'='*46}\n  AI Apps v2.0 — SQLite Chat History\n{'='*46}")
    print(f"  Local : http://localhost:5000")
    print(f"  LAN   : http://{ip}:5000")
    print(f"  DB    : {_db_path if not USE_TURSO else TURSO_URL}")
    print(f"\n  Deploy Render: Thêm biến môi trường:")
    print(f"    GROQ_API_KEY, OPENROUTER_API_KEY, APP_PASSWORD")
    print(f"    DB_PATH=/var/data/chat.db  (nếu dùng Render Disk)")
    print(f"{'='*46}\n")
    app.run(host='0.0.0.0',port=5000,debug=False)
