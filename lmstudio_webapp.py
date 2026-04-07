#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmstudio_webapp.py - v2.2 (Final UI & Turso Fix)
=================================================
Khắc phục lỗi Giao diện (Input Chat UI), giữ nguyên Topbar cũ
và Khắc phục lỗi Deadlock (Timeout Worker) của Turso trên Render.
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

# ===== DATABASE (TURSO / SQLITE SYNC FIX) =====
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

def _get_turso_url():
    url = TURSO_URL
    if url.startswith("libsql://"):
        url = url.replace("libsql://", "https://")
    return url

def init_db():
    if USE_TURSO:
        try:
            import libsql_client
            # Dùng 'with' để tự động mở và đóng kết nối -> Chống kẹt worker trên Render
            with libsql_client.create_client_sync(url=_get_turso_url(), auth_token=TURSO_TOKEN) as c:
                c.execute(_CREATE_SESSIONS)
                c.execute(_CREATE_MESSAGES_TURSO)
            print(f"  DB : Turso cloud ({TURSO_URL})")
        except Exception as e:
            print(f"  Turso Init Error: {e}")
    else:
        _sqlite_init()
        print(f"  DB : SQLite ({_db_path})")

def db_execute(sql, params=()):
    if USE_TURSO:
        import libsql_client
        with libsql_client.create_client_sync(url=_get_turso_url(), auth_token=TURSO_TOKEN) as c:
            c.execute(sql, list(params))
    else:
        with _sqlite_conn() as c:
            c.execute(sql, params)
            c.commit()

def db_fetchall(sql, params=()):
    if USE_TURSO:
        import libsql_client
        with libsql_client.create_client_sync(url=_get_turso_url(), auth_token=TURSO_TOKEN) as c:
            result = c.execute(sql, list(params))
            cols = [col.name for col in result.columns] if result.columns else []
            return [dict(zip(cols, row)) for row in (result.rows or [])]
    else:
        with _sqlite_conn() as c:
            rows = c.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

def db_fetchone(sql, params=()):
    rows = db_fetchall(sql, params)
    return rows[0] if rows else None

# Khởi tạo DB khi chạy
init_db()

# ===== AUTH =====
@app.before_request
def check_login():
    if APP_PASSWORD:
        auth = request.authorization
        if not auth or auth.password != APP_PASSWORD:
            return Response('Nhập mật khẩu để tiếp tục.', 401, {'WWW-Authenticate': 'Basic realm="AI Apps"'})

# ===== HELPERS & RATE LIMITER =====
def get_client(provider_key="lmstudio"):
    cfg = PROVIDERS.get(provider_key, PROVIDERS["lmstudio"])
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

def parse_reply(content):
    if not content: return ""
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    for tok in ["<|END_RESPONSE|>","<|end_of_turn|>","<|eot_id|>","<|endoftext|>","<|im_end|>"]:
        text = text.replace(tok, "")
    return text.strip()

import threading, time as _time
_groq_lock     = threading.Lock()
_groq_tokens   = 0
_groq_reqs     = 0
_groq_reset_at = 0.0
GROQ_MAX_TPM   = 7500   
GROQ_MAX_RPM   = 28

def _groq_wait(estimated_tokens=500):
    global _groq_tokens, _groq_reqs, _groq_reset_at
    with _groq_lock:
        now = _time.time()
        if now >= _groq_reset_at:
            _groq_tokens   = 0
            _groq_reqs     = 0
            _groq_reset_at = now + 60.0
        wait = 0.0
        if _groq_tokens + estimated_tokens > GROQ_MAX_TPM:
            wait = max(wait, _groq_reset_at - now)
        if _groq_reqs >= GROQ_MAX_RPM:
            wait = max(wait, _groq_reset_at - now)
        if wait > 0: print(f"  [Groq RL] chờ {wait:.1f}s")
        _groq_tokens += estimated_tokens
        _groq_reqs   += 1
    if wait > 0: _time.sleep(wait)

def _estimate_tokens(messages):
    return sum(len(m.get("content","")) // 3 for m in messages if isinstance(m.get("content"), str)) + 200

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
            except Exception as e2: return f"Lỗi sau retry: {str(e2)}"
        return f"Lỗi API: {err}"

# ===== FILE PROCESSING =====
import base64, mimetypes

ALLOWED_EXT = {'.txt','.md','.py','.js','.ts','.json','.csv','.html','.css',
               '.xml','.yaml','.yml','.ini','.sh','.bat','.sql','.log',
               '.pdf','.png','.jpg','.jpeg','.gif','.webp'}
MAX_FILE_MB  = 5
MAX_TEXT_CHARS = 12000

def _read_file_content(file_storage):
    fname = file_storage.filename or "file"
    ext   = os.path.splitext(fname)[1].lower()
    if ext not in ALLOWED_EXT: return None, None, fname
    data  = file_storage.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024: return None, "too_large", fname
    mime  = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    
    if mime.startswith("text/") or ext in {'.py','.js','.ts','.json','.csv',
            '.xml','.yaml','.yml','.ini','.sh','.bat','.sql','.log','.md'}:
        try:
            text = data.decode("utf-8", errors="replace")
            return text[:MAX_TEXT_CHARS], "text", fname
        except: return None, None, fname
        
    if mime.startswith("image/"):
        b64 = base64.b64encode(data).decode()
        return b64, mime, fname
        
    # PDF parsing (Cần pypdf)
    if ext == '.pdf':
        try:
            import io
            from pypdf import PdfReader
            pdf = PdfReader(io.BytesIO(data))
            text = "\n".join([page.extract_text() for page in pdf.pages])
            return text[:MAX_TEXT_CHARS], "text", fname
        except Exception: return None, None, fname
        
    return None, None, fname

def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===== HTML & CSS (NÂNG CẤP UI) =====
HTML = r"""<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Apps v2.2</title>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<style>
/* Markdown rendering */
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
.bbl.md-content a{color:var(--ac2);text-decoration:underline}

:root{
  --bg0:#09090b;--bg1:#111115;--bg2:#18181c;--bg3:#222228;
  --bd:#2e2e36;--bd2:#3a3a44;
  --tx:#f0eeff;--tx2:#9896aa;--tx3:#5a5870;
  --ac:#8b74ff;--ac2:#b09eff;--ac3:#6455dd;
  --gr:#34d399;--rd:#f87171;--am:#fbbf24;
  --f0:'Geist',sans-serif;--f1:'Geist Mono',monospace;
}
[data-theme="light"]{
  --bg0:#f4f3fa;--bg1:#ffffff;--bg2:#f0eff8;--bg3:#e8e7f4;
  --bd:#d4d2e8;--bd2:#c0bedd;
  --tx:#1a1826;--tx2:#5a5870;--tx3:#9896aa;
  --ac:#7c6af7;--ac2:#5a4de0;--ac3:#9e8fff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg0);color:var(--tx);font-family:var(--f0);height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* Layout & Nav */
.topbar{height:52px;background:var(--bg1);border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;padding:0 16px;z-index:50}
.logo{font-family:var(--f1);font-size:.95rem;color:var(--ac2);font-weight:500;}
.logo em{color:var(--tx2);font-style:normal}
.topbar-right{display:flex;align-items:center;gap:8px;}
select.sel{background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:7px;padding:5px 8px;font-size:.8rem;outline:none;font-family:var(--f0);}
.badge{display:flex;align-items:center;gap:6px;background:var(--bg2);border:1px solid var(--bd);padding:4px 10px;border-radius:20px;font-size:.75rem;font-family:var(--f1);color:var(--tx2);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--tx3);}
.dot.on{background:var(--gr);box-shadow:0 0 7px var(--gr);}
.icon-btn{background:var(--bg2);border:1px solid var(--bd);color:var(--tx2);width:32px;height:32px;border-radius:7px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:0.15s;}
.icon-btn:hover{border-color:var(--ac);color:var(--ac2)}

.layout{display:flex;flex:1;overflow:hidden}
.apnav{width:56px;background:var(--bg1);border-right:1px solid var(--bd);display:flex;flex-direction:column;align-items:center;padding:10px 0;gap:4px;}
.anv{width:40px;height:40px;border-radius:9px;display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:1.05rem;color:var(--tx3);transition:0.15s;position:relative}
.anv:hover{background:var(--bg3);color:var(--tx)}
.anv.active{background:var(--ac3);color:#fff;}
.anv .tip{position:absolute;left:52px;background:var(--bg3);border:1px solid var(--bd2);color:var(--tx);padding:4px 10px;border-radius:6px;font-size:.78rem;opacity:0;pointer-events:none;transition:0.1s;z-index:99;white-space:nowrap;}
.anv:hover .tip{opacity:1}
.anv-sep{width:32px;height:1px;background:var(--bd);margin:4px 0}

.hpanel{width:232px;background:var(--bg1);border-right:1px solid var(--bd);display:flex;flex-direction:column;transition:width 0.2s;overflow:hidden;}
.hpanel.hidden{width:0;border:none}
.hp-head{padding:10px 12px;border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;}
.hp-title{font-size:.72rem;font-weight:600;color:var(--tx3);text-transform:uppercase}
.hp-body{flex:1;overflow-y:auto;padding:4px}
.hg-label{font-size:.69rem;color:var(--tx3);padding:8px 8px 3px;font-weight:500;}
.hi{padding:7px 10px;border-radius:7px;cursor:pointer;font-size:.81rem;color:var(--tx2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:1px}
.hi:hover{background:var(--bg2);color:var(--tx)}
.hi.active{background:var(--bg3);color:var(--tx);}
.h-empty{padding:24px 12px;text-align:center;color:var(--tx3);font-size:.8rem;}

.main{flex:1;overflow:hidden;display:flex;flex-direction:column}
.panel{display:none;flex:1;overflow:hidden;flex-direction:column;}
.panel.active{display:flex}

/* Chat Area */
.chat-toolbar{padding:8px 16px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:8px;background:var(--bg1)}
.chat-toolbar span{font-size:.78rem;color:var(--tx3);}
.chat-toolbar input{flex:1;background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:7px;padding:5px 10px;font-size:.82rem;font-family:var(--f0);outline:none;}
.chat-toolbar input:focus{border-color:var(--ac)}
.chat-msgs{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.mrow{display:flex;gap:10px;}
.mrow.user{justify-content:flex-end}
.av{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.8rem;flex-shrink:0;}
.av.ai{background:linear-gradient(135deg,var(--ac3),var(--ac));color:#fff}
.bbl{padding:9px 13px;border-radius:13px;font-size:.87rem;line-height:1.65;max-width:76%;white-space:pre-wrap;word-break:break-word}
.mrow.user .bbl{background:var(--ac);color:#fff;border-radius:13px 13px 3px 13px}
.mrow.ai .bbl{background:var(--bg2);border:1px solid var(--bd);border-radius:13px 13px 13px 3px}
.mrow.sys{justify-content:center}
.mrow.sys .bbl{background:transparent;color:var(--tx3);font-size:.77rem;font-style:italic;}

/* CHAT INPUT GIAO DIỆN MỚI */
.chat-footer{padding:10px 16px 16px;background:var(--bg0);}
.chat-input-wrapper {
  display: flex; align-items: flex-end; gap: 8px;
  background: var(--bg2); border: 1px solid var(--bd);
  border-radius: 20px; padding: 6px 10px; transition: border-color 0.2s;
  max-width: 860px; margin: 0 auto;
}
.chat-input-wrapper:focus-within { border-color: var(--ac); box-shadow: 0 0 0 1px var(--ac); }
.chat-input-wrapper textarea {
  flex: 1; background: transparent; border: none; color: var(--tx);
  padding: 8px 4px; font-family: var(--f0); font-size: 0.95rem;
  resize: none; outline: none; min-height: 22px; max-height: 150px; line-height: 1.5;
}
.fbtn {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; color: var(--tx3); flex-shrink: 0; transition: 0.2s; margin-bottom: 1px;
}
.fbtn svg { width: 20px; height: 20px; }
.fbtn:hover { background: var(--bg3); color: var(--tx); }
.sbtn {
  width: 34px; height: 34px; border-radius: 50%;
  background: var(--ac); border: none; color: #fff;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; flex-shrink: 0; transition: 0.2s; margin-bottom: 2px;
}
.sbtn svg { width: 16px; height: 16px; margin-left: -2px; margin-top: 2px;}
.sbtn:hover { background: var(--ac2); transform: scale(1.05); }
.sbtn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.chint{font-size:.71rem;color:var(--tx3);margin-top:8px;text-align:center}

/* Cards & Other Apps */
.twrap{flex:1;overflow-y:auto;padding:22px}
.tinner{max-width:800px;margin:0 auto}
.th2{font-size:1.1rem;font-weight:600;margin-bottom:3px;display:flex;align-items:center;gap:8px}
.tsub{color:var(--tx2);font-size:.82rem;margin-bottom:16px}
.card{background:var(--bg1);border:1px solid var(--bd);border-radius:14px;padding:15px;margin-bottom:10px}
.lbl{font-size:.71rem;font-weight:600;color:var(--tx3);display:block;margin-bottom:5px;text-transform:uppercase}
.inp{width:100%;background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:8px;padding:8px 10px;font-family:var(--f0);font-size:.87rem;outline:none;}
.inp:focus{border-color:var(--ac)}
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 13px;border-radius:8px;border:none;font-family:var(--f0);font-size:.83rem;font-weight:500;cursor:pointer;}
.bp{background:var(--ac);color:#fff}.bp:hover{background:var(--ac2)}
.bg{background:var(--bg2);color:var(--tx);border:1px solid var(--bd)}.bg:hover{border-color:var(--ac);color:var(--ac2)}
.out{background:var(--bg2);border:1px solid var(--bd);border-radius:8px;padding:12px;font-family:var(--f1);font-size:.82rem;color:var(--tx);white-space:pre-wrap;min-height:80px;max-height:440px;overflow-y:auto}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px}
.chip{padding:3px 8px;border-radius:20px;font-size:.73rem;background:var(--bg3);border:1px solid var(--bd);color:var(--tx2);cursor:pointer;}
.chip:hover{border-color:var(--ac);color:var(--ac2)}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:3px}
</style>
</head>
<body>
<div class="topbar">
  <div class="logo">AI<em>.apps</em></div>
  <div class="topbar-right" style="justify-content:center;">
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
      <div class="chat-toolbar">
        <span>System:</span>
        <input id="chat-system" placeholder="Bạn là trợ lý AI hữu ích, trả lời bằng tiếng Việt.">
      </div>
      <div class="chat-msgs" id="chat-msgs">
        <div class="mrow sys"><div class="bbl">Chọn cuộc trò chuyện bên trái hoặc nhấn ✏️ để bắt đầu mới</div></div>
      </div>
      
      <div class="chat-footer">
        <div style="max-width:860px; margin:0 auto">
          <div id="file-preview" style="display:none; padding: 6px 12px; margin-bottom: 8px; background: var(--bg2); border: 1px solid var(--bd); border-radius: 12px; align-items: center; justify-content: space-between; gap: 12px; width: fit-content; max-width: 100%; box-sizing: border-box;"></div>
          
          <div class="chat-input-wrapper">
            <label class="fbtn" title="Đính kèm file (Text, Ảnh, PDF)">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
              <input type="file" id="file-input" style="display:none" onchange="onFileSelect(this)" accept=".txt,.md,.py,.js,.ts,.json,.csv,.html,.css,.xml,.yaml,.yml,.sh,.sql,.log,.pdf,.png,.jpg,.jpeg,.gif,.webp">
            </label>
            <textarea id="chat-input" rows="1" placeholder="Nhập tin nhắn..." oninput="arz(this)" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
            <button class="sbtn" id="sbtn" onclick="sendChat()" title="Gửi (Enter)">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
          </div>
        </div>
        <div class="chint">Enter gửi · Shift+Enter xuống dòng</div>
      </div>
    </div>

    <div class="panel" id="panel-optimizer">
      <div class="twrap"><div class="tinner"><div class="th2">✨ Tối ưu Prompt</div><div class="card"><textarea class="inp" id="opt-in" rows="4"></textarea><br><br><button class="btn bp" onclick="tc('optimize_prompt',{text:g('opt-in')},this.id,'opt-out')" id="ob">✨ Tối ưu hóa</button></div><div class="card"><div class="out" id="opt-out">...</div></div></div></div>
    </div>
    <div class="panel" id="panel-translate">
      <div class="twrap"><div class="tinner"><div class="th2">🌐 Dịch thuật</div><div class="card">
        <select class="inp" id="tr-src" style="width:40%;display:inline-block"><option value="English">English</option><option value="Vietnamese">Tiếng Việt</option></select> ⇄ 
        <select class="inp" id="tr-dst" style="width:40%;display:inline-block"><option value="Vietnamese">Tiếng Việt</option><option value="English">English</option></select>
        <br><br><textarea class="inp" id="tr-in" rows="4"></textarea><br><br><button class="btn bp" onclick="tc('translate',{text:g('tr-in'),src:g('tr-src'),dst:g('tr-dst')},this.id,'tr-out')" id="tb">Dịch</button></div>
        <div class="card"><div class="out" id="tr-out">...</div></div></div></div>
    </div>
    <div class="panel" id="panel-review">
      <div class="twrap"><div class="tinner"><div class="th2">🔍 Code Review</div><div class="card"><textarea class="inp" id="rv-in" rows="6"></textarea><br><br><button class="btn bp" onclick="tc('review',{code:g('rv-in')},this.id,'rv-out')" id="rb">Review</button></div><div class="card"><div class="out" id="rv-out">...</div></div></div></div>
    </div>
    <div class="panel" id="panel-summary">
      <div class="twrap"><div class="tinner"><div class="th2">📄 Tóm tắt</div><div class="card"><textarea class="inp" id="sm-in" rows="6"></textarea><br><br><button class="btn bp" onclick="tc('summary',{text:g('sm-in')},this.id,'sm-out')" id="sb">Tóm tắt</button></div><div class="card"><div class="out" id="sm-out">...</div></div></div></div>
    </div>
    <div class="panel" id="panel-mockdata">
      <div class="twrap"><div class="tinner"><div class="th2">🗄️ Mock Data</div><div class="card"><input class="inp" id="mk-in" placeholder="VD: 5 users"><br><br><button class="btn bp" onclick="tc('mockdata',{schema:g('mk-in')},this.id,'mk-out')" id="mb">Tạo</button></div><div class="card"><div class="out" id="mk-out">...</div></div></div></div>
    </div>
    <div class="panel" id="panel-terminal">
      <div class="twrap"><div class="tinner"><div class="th2">⌨️ Terminal</div><div class="card"><textarea class="inp" id="tm-in" rows="4"></textarea><br><br><button class="btn bp" onclick="tc('terminal',{input:g('tm-in')},this.id,'tm-out')" id="tmb">Phân tích</button></div><div class="card"><div class="out" id="tm-out">...</div></div></div></div>
    </div>

  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const g = id => ($(id)?.value||$(id)?.textContent||'').trim();
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const arz = el => { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,150)+'px'; };

let curSid=null, chatMsgs=[], hpVisible=true;

function toggleTheme(){
  const h=document.documentElement, dark=h.getAttribute('data-theme')==='dark';
  h.setAttribute('data-theme',dark?'light':'dark');
}
function toggleHP(){ hpVisible=!hpVisible; $('hpanel').classList.toggle('hidden',!hpVisible); }

function switchApp(name){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.anv').forEach(n=>n.classList.remove('active'));
  $('panel-'+name).classList.add('active');
  $('anav-'+name).classList.add('active');
  $('hpanel').style.visibility = name==='chat'?'':'hidden';
  $('hpanel').style.width      = name==='chat'&&hpVisible?'':'0';
}

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
      $('sdot').className='dot'; $('stxt').innerText='Kiểm tra API';
    }
  }catch(e){ $('stxt').innerText='Lỗi API'; }
}

async function loadHistory(){
  try{
    const ss=await(await fetch('/api/sessions')).json();
    const el=$('hlist');
    if(!ss.length){ el.innerHTML='<div class="h-empty">Chưa có trò chuyện</div>'; return; }
    el.innerHTML='';
    ss.forEach(s=>{
      const ac=s.id===curSid?'active':'';
      el.innerHTML+=`<div class="hi ${ac}" onclick="loadSess('${s.id}')">${esc(s.title)}</div>`;
    });
  }catch(e){}
}

async function newSession(){
  curSid=null; chatMsgs=[];
  $('chat-msgs').innerHTML='<div class="mrow sys"><div class="bbl">Cuộc trò chuyện mới ✨</div></div>';
  await loadHistory();
}

async function loadSess(id){
  try{
    const d=await(await fetch(`/api/sessions/${id}`)).json();
    curSid=d.session.id; chatMsgs=d.messages.map(m=>({role:m.role,content:m.content}));
    $('chat-msgs').innerHTML='';
    d.messages.forEach(m=>addBbl(m.role,m.content));
    await loadHistory();
  }catch(e){}
}

function addBbl(role,content){
  const m=$('chat-msgs');
  if(role==='sys'){ m.innerHTML+=`<div class="mrow sys"><div class="bbl">${esc(content)}</div></div>`; return; }
  const u=role==='user';
  const rnd = u ? esc(content) : (typeof marked!=='undefined' ? marked.parse(content) : esc(content));
  m.innerHTML+=`<div class="mrow ${u?'user':'ai'}"><div class="bbl ${u?'':'md-content'}">${rnd}</div></div>`;
  m.scrollTop=m.scrollHeight;
}

let _fileInfo = null;
function onFileSelect(input) {
  const file = input.files[0];
  if (!file) return;
  const prev = $('file-preview');
  prev.style.display = 'flex';
  prev.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;font-size:0.85rem;color:var(--tx2);">
      <span style="background:var(--bg3);padding:4px 8px;border-radius:6px;border:1px solid var(--bd);">
        📄 ${file.name}
      </span>
      <span id="up-stat">⏳ Đang xử lý...</span>
    </div>
    <button onclick="clearFile()" style="background:none;border:none;color:var(--rd);cursor:pointer;font-size:1.1rem">×</button>
  `;
  const fd = new FormData(); fd.append('file', file);
  fetch('/api/upload', {method:'POST', body:fd}).then(r=>r.json()).then(d=>{
    if(d.error) { $('up-stat').textContent='❌ '+d.error; _fileInfo=null; }
    else { $('up-stat').textContent='✅ Sẵn sàng'; $('up-stat').style.color='var(--gr)'; _fileInfo=d; }
  }).catch(e=>{ $('up-stat').textContent='❌ Lỗi tải lên'; _fileInfo=null; });
  input.value='';
}
function clearFile() { _fileInfo=null; $('file-preview').style.display='none'; }

async function sendChat(){
  const inp=$('chat-input'), msg=inp.value.trim();
  if(!msg && !_fileInfo) return;
  if(!curSid){
    const r=await fetch('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title:(msg||_fileInfo?.name||'New').slice(0,40),provider:$('gp').value,model:$('gm').value})});
    curSid=(await r.json()).id;
  }
  addBbl('user', msg + (_fileInfo?` 📎 ${_fileInfo.name}`:''));
  chatMsgs.push({role:'user', content: msg||'[File]'});
  inp.value=''; inp.style.height='auto';
  const fSend = _fileInfo; clearFile();
  $('sbtn').disabled=true;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg, history:chatMsgs, file:fSend,
        system:$('chat-system').value, provider:$('gp').value, model:$('gm').value, session_id:curSid})});
    const reply=(await r.json()).reply;
    addBbl('ai',reply); chatMsgs.push({role:'assistant',content:reply});
    await loadHistory();
  }catch(e){ addBbl('sys','Lỗi kết nối'); }
  $('sbtn').disabled=false;
}

async function tc(ep,payload,btnId,outId){
  if(Object.values(payload).some(v=>!v)) return;
  $(outId).textContent='⏳ Đang xử lý...';
  payload.provider=$('gp').value; payload.model=$('gm').value;
  try{
    const r=await fetch('/api/'+ep,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    $(outId).textContent=(await r.json()).result;
  }catch(e){ $(outId).textContent='❌ Lỗi'; }
}

window.onload=async()=>{ await initProviders(); await loadHistory(); const ss=await(await fetch('/api/sessions')).json(); if(ss.length) await loadSess(ss[0].id); };
</script>
</body>
</html>"""

# ===== ROUTES =====
@app.route('/')
def index(): return HTML

@app.route('/api/providers')
def api_providers(): return jsonify({k:{"name":v["name"],"default":v["default_model"]} for k,v in PROVIDERS.items()})

@app.route('/api/models')
def api_models():
    p = request.args.get('provider','lmstudio'); cfg = PROVIDERS.get(p, PROVIDERS["lmstudio"])
    try:
        ids = [m.id for m in get_client(p).models.list().data]; d = cfg["default_model"]
        if d in ids: ids.insert(0, ids.pop(ids.index(d)))
        return jsonify({"models":ids,"default":d})
    except: return jsonify({"models":[],"default":cfg["default_model"]})

@app.route('/api/sessions', methods=['GET','POST'])
def handle_sessions():
    if request.method == 'GET':
        return jsonify(db_fetchall("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 100"))
    d=request.json; sid=str(uuid.uuid4()); ts=now_str()
    db_execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)", (sid, d.get('title','New'), d.get('system_prompt',''), d.get('provider','lmstudio'), d.get('model',''), ts, ts))
    return jsonify({"id":sid})

@app.route('/api/sessions/<sid>', methods=['GET'])
def get_session(sid):
    s = db_fetchone("SELECT * FROM sessions WHERE id=?", (sid,))
    if not s: return jsonify({"error":"Not found"}), 404
    return jsonify({"session": s, "messages": db_fetchall("SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,))})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    c_data, c_type, fname = _read_file_content(request.files['file'])
    if c_type == "too_large": return jsonify({"error": "File > 5MB"}), 400
    if c_data is None: return jsonify({"error": "Định dạng không hỗ trợ"}), 400
    return jsonify({"content": c_data, "type": c_type, "name": fname, "is_image": c_type not in ("text","too_large")})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    d = request.json; sid = d.get('session_id'); u_msg = d.get('message', '')
    f_info = d.get('file'); msgs = [{"role": "system", "content": d.get("system", "Bạn là trợ lý AI.")}]
    msgs.extend(d.get("history", []))

    if f_info:
        if f_info.get("is_image"): msgs.append({"role": "user", "content": [{"type": "text", "text": u_msg or "Ảnh:"}, {"type": "image_url", "image_url": {"url": f"data:{f_info['type']};base64,{f_info['content']}"}}]})
        else: msgs.append({"role": "user", "content": f"[File: {f_info['name']}]\n```\n{f_info['content']}\n```\n\n{u_msg}"})
    else: msgs.append({"role": "user", "content": u_msg})

    reply = llm_call(msgs, d)
    if sid:
        ts = now_str()
        db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)", (sid, 'user', u_msg + (f" [+{f_info['name']}]" if f_info else ""), ts))
        db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)", (sid, 'assistant', reply, ts))
        db_execute("UPDATE sessions SET updated_at=? WHERE id=?", (ts, sid))
    return jsonify({"reply": reply})

# Các tính năng phụ trợ
for path in ['optimize_prompt', 'translate', 'review', 'summary', 'mockdata', 'terminal']:
    @app.route(f'/api/{path}', methods=['POST'], endpoint=path)
    def handle_tool(p=path):
        d=request.json; content = d.get('text') or d.get('code') or d.get('schema') or d.get('input') or str(d)
        return jsonify({"result": llm_call([{"role":"system","content":"Thực thi yêu cầu chuyên môn."},{"role":"user","content":content}], d, max_tokens=1024, temperature=0.3)})

if __name__=='__main__':
    import socket
    ip=socket.gethostbyname(socket.gethostname())
    print(f"\n{'='*46}\n  AI Apps v2.2 — Giao diện BO CONG cực xịn\n{'='*46}")
    print(f"  Local : http://localhost:5000\n  LAN   : http://{ip}:5000")
    print(f"{'='*46}\n")
    app.run(host='0.0.0.0',port=5000,debug=False)
