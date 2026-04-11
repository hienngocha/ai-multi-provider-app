#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmstudio_webapp.py - v3.0
==========================
Web app AI voi giao dien moi + luu lich su chat.

Database:
  - Local / LAN : SQLite (mac dinh, khong can cai them)
  - Render cloud: Turso (libSQL cloud, mien phi 5GB)

Cai dat:
  pip install flask openai
  pip install requests         # Da co san trong flask, khong can cai them

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
from google import genai
from ollama import chat
# ===== CONFIG =====
try:
    from config import LM_STUDIO_URL, REQUEST_TIMEOUT, GROQ_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, XAI_API_KEY, OLLAMA_API_KEY
except ImportError:
    LM_STUDIO_URL      = "http://127.0.0.1:1234"
    REQUEST_TIMEOUT    = 120
    GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

# Thêm dòng này cho Ollama (Mặc định Ollama chạy port 11434)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()
app = Flask(__name__)
PROVIDERS = {
    "lmstudio":   {"name": "LM Studio (Local)",       "base_url": f"{LM_STUDIO_URL}/v1",             "api_key": "lm-studio",        "default_model": "local-model"},
#    "ollama":     {"name": "Ollama (Local)",          "base_url": f"{OLLAMA_URL}/v1",                "api_key": "ollama",           "default_model": ""},
    "ollama_cloud": {"name": "Ollama (Cloud)",        "base_url": "https://ollama.com/v1",       "api_key": OLLAMA_API_KEY,     "default_model": "glm-5.1"},
    "groq":       {"name": "Groq (Cloud - Fast)",      "base_url": "https://api.groq.com/openai/v1",  "api_key": GROQ_API_KEY,       "default_model": "llama3-70b-8192"},
    "openrouter": {"name": "OpenRouter (Multi-Model)", "base_url": "https://openrouter.ai/api/v1",    "api_key": OPENROUTER_API_KEY, "default_model": "google/gemini-pro-1.5"},
    "gemini":     {"name": "Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",    "api_key": GEMINI_API_KEY, "default_model": "Gemma 4 31B"},
#    "deepSeek":   {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1",    "api_key": DEEPSEEK_API_KEY, "default_model": "deepseek-chat"},
#    "grok":       {"name": "Grok", "base_url": "https://api.x.ai/v1",    "api_key": XAI_API_KEY, "default_model": "grok-4.20-reasoning-latest"},
}

# ===== DATABASE =====
# Tu dong chon: Turso HTTP API (cloud) hoac SQLite (local)
# Turso HTTP API: khong dung asyncio, tuong thich gunicorn sync worker
TURSO_URL   = os.environ.get("TURSO_URL", "").strip()
TURSO_TOKEN = os.environ.get("TURSO_TOKEN", "").strip()
USE_TURSO   = bool(TURSO_URL and TURSO_TOKEN)

_CREATE_SESSIONS = """CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, title TEXT NOT NULL,
    system_prompt TEXT DEFAULT '', provider TEXT DEFAULT 'lmstudio',
    model TEXT DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""

_CREATE_MESSAGES = """CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"""

_CREATE_TEMPLATES = """CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, icon TEXT DEFAULT '🤖',
    system_prompt TEXT NOT NULL, description TEXT DEFAULT '',
    created_at TEXT NOT NULL)"""

_DEFAULT_TEMPLATES = [
    ("tpl-translator", "Dịch thuật chuyên nghiệp", "🌐",
     "Bạn là phiên dịch viên chuyên nghiệp. Dịch chính xác, tự nhiên, giữ nguyên văn phong và sắc thái gốc. Không thêm giải thích trừ khi được yêu cầu.",
     "Dịch đa ngôn ngữ, chuyên nghiệp"),
    ("tpl-dev",        "Lập trình viên Senior",   "💻",
     "Bạn là senior software engineer với 10+ năm kinh nghiệm. Viết code sạch, có comment rõ ràng, giải thích ngắn gọn bằng tiếng Việt. Ưu tiên best practices và bảo mật.",
     "Hỗ trợ code, review, debug"),
    ("tpl-content",    "Content Creator",          "✍️",
     "Bạn là chuyên gia Content Marketing. Viết nội dung sáng tạo, hấp dẫn, SEO-friendly. Phong cách trẻ trung, gần gũi, phù hợp mạng xã hội Việt Nam.",
     "Tạo nội dung marketing"),
    ("tpl-teacher",    "Gia sư thân thiện",        "🎓",
     "Bạn là gia sư kiên nhẫn và thân thiện. Giải thích đơn giản từ cơ bản đến nâng cao, dùng ví dụ thực tế. Khuyến khích người học và trả lời bằng tiếng Việt.",
     "Giải thích, hướng dẫn học tập"),
    ("tpl-analyst",    "Chuyên gia phân tích",     "📊",
     "Bạn là chuyên gia phân tích dữ liệu và chiến lược kinh doanh. Phân tích logic, đưa ra insight rõ ràng, trình bày có cấu trúc với bullet points và bảng biểu khi cần.",
     "Phân tích dữ liệu, chiến lược"),
]


# ── Turso HTTP API (dung requests, khong asyncio) ──
import requests as _req

def _turso_http(sql, params=()):
    """
    Goi Turso qua HTTP API.
    Tra ve list of dict, hoac [] neu khong co ket qua.
    Turso HTTP endpoint: POST /v2/pipeline
    """
    # Chuyen params thanh dung dinh dang Turso mong doi
    args = []
    for p in params:
        if p is None:
            args.append({"type": "null"})
        elif isinstance(p, int):
            args.append({"type": "integer", "value": str(p)})
        elif isinstance(p, float):
            args.append({"type": "float", "value": str(p)})
        else:
            args.append({"type": "text", "value": str(p)})

    # Turso URL co the dang https:// hoac libsql://
    # HTTP API can dung https://
    base_url = TURSO_URL.replace("libsql://", "https://")
    endpoint = f"{base_url}/v2/pipeline"
    headers  = {
        "Authorization": f"Bearer {TURSO_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": args}},
            {"type": "close"}
        ]
    }
    resp = _req.post(endpoint, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # Parse ket qua
    result = data.get("results", [{}])[0]
    if result.get("type") == "error":
        raise Exception(f"Turso error: {result.get('error', {}).get('message', 'unknown')}")

    rows_data = result.get("response", {}).get("result", {})
    cols  = [c.get("name") for c in rows_data.get("cols", [])]
    rows  = rows_data.get("rows", [])
    out   = []
    for row in rows:
        vals = []
        for cell in row:
            t = cell.get("type", "null")
            v = cell.get("value")
            if t == "null" or v is None:
                vals.append(None)
            elif t == "integer":
                vals.append(int(v))
            elif t == "float":
                vals.append(float(v))
            else:
                vals.append(v)
        out.append(dict(zip(cols, vals)))
    return out


# ── SQLite local ──
_db_path = Path(os.environ.get("DB_PATH", "") or (Path(__file__).parent / "chat_history.db"))

def _sqlite_conn():
    c = sqlite3.connect(str(_db_path))
    c.row_factory = sqlite3.Row
    return c

def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Interface thong nhat ──
def init_db():
    if USE_TURSO:
        _turso_http(_CREATE_SESSIONS)
        _turso_http(_CREATE_MESSAGES)
        _turso_http(_CREATE_TEMPLATES)
        print(f"  DB : Turso HTTP API ({TURSO_URL})")
    else:
        _db_path.parent.mkdir(parents=True, exist_ok=True)
        with _sqlite_conn() as c:
            c.execute(_CREATE_SESSIONS)
            c.execute(_CREATE_MESSAGES.replace(
                "INTEGER PRIMARY KEY",
                "INTEGER PRIMARY KEY AUTOINCREMENT"))
            c.execute(_CREATE_TEMPLATES)
            c.commit()
        print(f"  DB : SQLite ({_db_path})")
    # Them default templates neu chua co
    ts = now_str()
    for tid, name, icon, prompt, desc in _DEFAULT_TEMPLATES:
        existing = db_fetchone("SELECT id FROM templates WHERE id=?", (tid,))
        if not existing:
            db_execute("INSERT INTO templates VALUES (?,?,?,?,?,?)",
                       (tid, name, icon, prompt, desc, ts))

def db_execute(sql, params=()):
    if USE_TURSO:
        _turso_http(sql, params)
    else:
        with _sqlite_conn() as c:
            c.execute(sql, params)
            c.commit()

def db_fetchall(sql, params=()):
    if USE_TURSO:
        return _turso_http(sql, params)
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

def llm_call_stream(messages, d, max_tokens=4096, temperature=0.7):
    """Generator: yield từng chunk text từ streaming API (SSE)."""
    try:
        provider = d.get('provider', 'lmstudio')
        model    = d.get('model') or PROVIDERS[provider]["default_model"]
        if provider == 'groq':
            _groq_wait(_estimate_tokens(messages) + max_tokens)
        stream = get_client(provider).chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=True)
        # Bộ lọc token đặc biệt (think tags, stop tokens)
        _STOP_TOKS = ["<|END_RESPONSE|>","<|end_of_turn|>","<|eot_id|>",
                      "<|endoftext|>","<|im_end|>"]
        buf = ""
        in_think = False
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            buf += delta
            # Loc <think>...</think> streaming
            while True:
                if in_think:
                    end = buf.find("</think>")
                    if end == -1:
                        buf = ""  # Xoa phan trong think block
                        break
                    else:
                        buf = buf[end + 8:]
                        in_think = False
                else:
                    start = buf.find("<think>")
                    if start == -1:
                        break
                    yield buf[:start]
                    buf = buf[start + 7:]
                    in_think = True
            if not in_think:
                # Loc stop tokens
                clean = buf
                for tok in _STOP_TOKS:
                    clean = clean.replace(tok, "")
                if clean:
                    yield clean
                    buf = ""
        # Flush phan con lai
        if buf and not in_think:
            for tok in _STOP_TOKS:
                buf = buf.replace(tok, "")
            if buf:
                yield buf
    except Exception as e:
        yield f"\n\n❌ Lỗi streaming: {str(e)}"

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

# ===== HTML =====
HTML = r"""<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Apps</title>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked@9/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
/* Markdown rendering trong bubble chat */
.bbl.md-content h1,.bbl.md-content h2,.bbl.md-content h3{font-weight:600;margin:0 0 .2em;line-height:1.3}
.bbl.md-content h1{font-size:1.05em}.bbl.md-content h2{font-size:.98em}.bbl.md-content h3{font-size:.92em}
.bbl.md-content p{margin:0 0 .3em;line-height:1.5}
.bbl.md-content ul,.bbl.md-content ol{padding-left:1.2em;margin:0 0 .2em}
.bbl.md-content li{margin:.1em 0;line-height:1.5}
.bbl.md-content table{border-collapse:collapse;width:100%;margin:.4em 0;font-size:.85em}
.bbl.md-content th,.bbl.md-content td{border:1px solid var(--bd2);padding:5px 8px;text-align:left}
.bbl.md-content th{background:var(--bg3);font-weight:600;color:var(--tx)}
.bbl.md-content td{background:var(--bg2);color:var(--tx2)}
.bbl.md-content tr:hover td{background:var(--bg3)}
.bbl.md-content code{background:var(--bg3);border:1px solid var(--bd);border-radius:4px;padding:1px 5px;font-family:var(--f1);font-size:.85em;color:var(--ac2)}
.bbl.md-content pre{background:var(--bg0);border:1px solid var(--bd);border-radius:8px;padding:8px 10px;overflow-x:auto;margin:.3em 0}
.bbl.md-content pre code{background:none;border:none;padding:0;font-size:.82em;color:var(--tx)}
.bbl.md-content blockquote{border-left:3px solid var(--ac);padding:.2em .6em;margin:.3em 0;color:var(--tx2);background:var(--bg2);border-radius:0 4px 4px 0}
.bbl.md-content strong{font-weight:600;color:var(--tx)}
.bbl.md-content em{font-style:italic;color:var(--tx2)}
.bbl.md-content hr{border:none;border-top:1px solid var(--bd);margin:.4em 0}
.bbl.md-content a{color:var(--ac2);text-decoration:underline}
.bbl.md-content br{display:block;margin:0;line-height:1.2}
.stream-cursor{display:inline-block;color:var(--ac2);font-weight:400;animation:blink-cur .7s step-end infinite;margin-left:1px}
@keyframes blink-cur{0%,100%{opacity:1}50%{opacity:0}}
/* Code block với nút Copy */
.bbl.md-content pre{position:relative}
.bbl.md-content pre .copy-code-btn{position:absolute;top:6px;right:6px;background:var(--bg3);border:1px solid var(--bd2);color:var(--tx2);border-radius:5px;padding:2px 8px;font-size:.72rem;cursor:pointer;font-family:var(--f0);opacity:0;transition:opacity .15s}
.bbl.md-content pre:hover .copy-code-btn{opacity:1}
.bbl.md-content pre .copy-code-btn:hover{background:var(--ac);color:#fff;border-color:var(--ac)}
/* highlight.js override */
.bbl.md-content pre code.hljs{background:transparent;padding:0;font-size:.82em}
/* Modal overlay */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:200;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(4px)}
.modal-overlay.hidden{display:none}
.modal{background:var(--bg1);border:1px solid var(--bd2);border-radius:var(--r2);padding:20px;width:min(520px,92vw);max-height:80vh;overflow-y:auto;display:flex;flex-direction:column;gap:12px}
.modal-head{display:flex;align-items:center;justify-content:space-between}
.modal-title{font-size:.95rem;font-weight:600;color:var(--tx)}
.modal-close{cursor:pointer;color:var(--tx3);font-size:1.1rem;background:none;border:none;padding:2px 6px;border-radius:5px}
.modal-close:hover{background:var(--bg3);color:var(--rd)}
/* Template cards */
.tpl-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.tpl-card{background:var(--bg2);border:1px solid var(--bd);border-radius:10px;padding:10px 12px;cursor:pointer;transition:all .15s;position:relative}
.tpl-card:hover{border-color:var(--ac);background:var(--bg3)}
.tpl-card.active-tpl{border-color:var(--ac);box-shadow:0 0 0 2px rgba(139,116,255,.2)}
.tpl-icon{font-size:1.3rem;margin-bottom:4px}
.tpl-name{font-size:.82rem;font-weight:600;color:var(--tx);margin-bottom:2px}
.tpl-desc{font-size:.73rem;color:var(--tx3);line-height:1.3}
.tpl-del{position:absolute;top:6px;right:6px;opacity:0;font-size:.7rem;color:var(--tx3);background:none;border:none;cursor:pointer;padding:2px 5px;border-radius:4px}
.tpl-card:hover .tpl-del{opacity:1}
.tpl-del:hover{background:var(--rd);color:#fff}
/* Search bar in history */
.hp-search{padding:6px 8px;border-bottom:1px solid var(--bd);flex-shrink:0}
.hp-search input{width:100%;background:var(--bg2);border:1px solid var(--bd);color:var(--tx);border-radius:7px;padding:5px 9px;font-size:.8rem;font-family:var(--f0);outline:none}
.hp-search input:focus{border-color:var(--ac)}
/* TTS button */
.tts-btn{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;border:1px solid var(--bd);background:var(--bg2);color:var(--tx2);font-size:.75rem;cursor:pointer;transition:all .15s;font-family:var(--f0)}
.tts-btn:hover{border-color:var(--ac);color:var(--ac2)}
.tts-btn.playing{border-color:var(--gr);color:var(--gr);background:rgba(52,211,153,.08)}
/* Chat toolbar nút template */
.sys-wrap{display:flex;gap:6px;align-items:center;flex:1}
.tpl-btn{height:30px;padding:0 10px;background:var(--bg3);border:1px solid var(--bd);color:var(--tx2);border-radius:7px;font-size:.77rem;cursor:pointer;white-space:nowrap;font-family:var(--f0);transition:all .15s}
.tpl-btn:hover{border-color:var(--ac);color:var(--ac2)}
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
.hi{display:flex;align-items:center;justify-content:space-between}
.hi-content{flex:1;overflow:hidden}
.hi-del{opacity:0;width:20px;height:20px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:.7rem;color:var(--tx3);cursor:pointer;transition:all .15s;flex-shrink:0;margin-left:4px}
.hi:hover .hi-del{opacity:1}
.hi-del:hover{background:var(--rd);color:#fff}
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
    <div class="anv-sep"></div>
    <div class="anv" id="anav-imgprompt" onclick="switchApp('imgprompt')">🎨<span class="tip">Prompt Ảnh</span></div>
  </nav>

  <div class="hpanel" id="hpanel">
    <div class="hp-head">
      <span class="hp-title">Lịch sử chat</span>
      <div style="display:flex;gap:4px">
        <div class="icon-btn" onclick="newSession()" title="Tạo mới">✏️</div>
        <div class="icon-btn" onclick="exportCurrent()" title="Xuất MD/JSON">⬇️</div>
      </div>
    </div>
    <div class="hp-search"><input id="hist-search" placeholder="🔍 Tìm kiếm..." oninput="filterHistory(this.value)"></div>
    <div class="hp-body" id="hlist"></div>
  </div>

  <div class="main">

    <div class="panel active" id="panel-chat">
      <div class="chat-wrap">
        <div class="chat-toolbar">
          <span>System:</span>
          <div class="sys-wrap">
            <input id="chat-system" placeholder="Bạn là trợ lý AI hữu ích, trả lời bằng tiếng Việt.">
            <button class="tpl-btn" onclick="openTplModal()" title="Chọn nhân cách AI">🧩 Template</button>
          </div>
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
        <div class="tsub">Dịch văn bản đa ngôn ngữ — hỗ trợ văn học & game</div>
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
          <label class="lbl">Văn phong</label>
          <div class="chips" style="margin-bottom:10px">
            <span class="chip" id="style-standard" onclick="setTrStyle('standard')" style="border-color:var(--ac);color:var(--ac2)">📝 Tiêu chuẩn</span>
            <span class="chip" id="style-game" onclick="setTrStyle('game')">🎮 Game / Visual Novel</span>
            <span class="chip" id="style-literary" onclick="setTrStyle('literary')">📖 Văn học / Bay bổng</span>
          </div>
          <label class="lbl">Văn bản</label>
          <textarea class="inp" id="tr-in" rows="5" placeholder="Nhập văn bản cần dịch..."></textarea>
          <div class="row">
            <button class="btn bp" id="tr-btn" onclick="doTranslate()">🌐 Dịch</button>
            <button class="btn bg" onclick="cp('tr-out')">📋 Sao chép</button>
            <button class="tts-btn" id="tr-tts" onclick="ttsToggle('tr-out','tr-tts')">🔊 Đọc</button>
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
            <button class="tts-btn" id="sm-tts" onclick="ttsToggle('sm-out','sm-tts')">🔊 Đọc</button>
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

    <div class="panel" id="panel-imgprompt">
      <div class="twrap"><div class="tinner">
        <div class="th2">🎨 Tạo Prompt Ảnh AI</div>
        <div class="tsub">Tối ưu cho Stable Diffusion, SDXL, FLUX, ComfyUI</div>
        <div class="card">
          <label class="lbl">Phong cách</label>
          <div class="chips" style="margin-bottom:10px">
            <span class="chip" id="img-style-realistic" onclick="setImgStyle('realistic')" style="border-color:var(--ac);color:var(--ac2)">📷 Realistic</span>
            <span class="chip" id="img-style-anime" onclick="setImgStyle('anime')">🌸 Anime</span>
            <span class="chip" id="img-style-art" onclick="setImgStyle('art')">🖼️ Digital Art</span>
            <span class="chip" id="img-style-sdxl" onclick="setImgStyle('sdxl')">⚡ SDXL/FLUX</span>
          </div>
          <label class="lbl">Ý tưởng của bạn</label>
          <textarea class="inp" id="img-in" rows="4" placeholder="VD: Một cô gái anime đứng dưới mưa ban đêm, ánh đèn đường, phong cách Studio Ghibli..."></textarea>
          <div class="row">
            <button class="btn bp" id="img-btn" onclick="doImgPrompt()">🎨 Tạo Prompt</button>
            <button class="btn bg" onclick="cp('img-pos')">📋 Copy Positive</button>
            <button class="btn bg" onclick="cp('img-neg')">📋 Copy Negative</button>
            <button class="tts-btn" id="img-tts" onclick="ttsToggle('img-pos','img-tts')">🔊 Đọc</button>
          </div>
        </div>
        <div class="card">
          <label class="lbl">✅ Positive Prompt</label>
          <div class="out ph" id="img-pos" style="font-family:var(--f1);font-size:.8rem;min-height:70px">Prompt positive sẽ hiện ở đây...</div>
        </div>
        <div class="card">
          <label class="lbl">❌ Negative Prompt</label>
          <div class="out ph" id="img-neg" style="font-family:var(--f1);font-size:.8rem;min-height:50px;color:var(--rd)">Prompt negative sẽ hiện ở đây...</div>
        </div>
      </div></div>
    </div>

  </div><!-- end .main -->
</div><!-- end .layout -->

<!-- ===== TEMPLATE MODAL ===== -->
<div class="modal-overlay hidden" id="tpl-modal" onclick="if(event.target===this)closeTplModal()">
  <div class="modal">
    <div class="modal-head">
      <span class="modal-title">🧩 Chọn nhân cách AI</span>
      <button class="modal-close" onclick="closeTplModal()">✕</button>
    </div>
    <div id="tpl-grid" class="tpl-grid"></div>
    <div style="border-top:1px solid var(--bd);padding-top:12px">
      <div class="lbl" style="margin-bottom:8px">➕ Thêm template mới</div>
      <div style="display:flex;gap:8px;margin-bottom:8px">
        <input class="inp" id="new-tpl-icon" placeholder="🤖" style="width:52px;flex-shrink:0;text-align:center">
        <input class="inp" id="new-tpl-name" placeholder="Tên template...">
      </div>
      <input class="inp" id="new-tpl-desc" placeholder="Mô tả ngắn..." style="margin-bottom:8px">
      <textarea class="inp" id="new-tpl-prompt" rows="3" placeholder="Viết system prompt tại đây..."></textarea>
      <div class="row" style="margin-top:8px">
        <button class="btn bp" onclick="saveTpl()">💾 Lưu Template</button>
      </div>
    </div>
  </div>
</div>

<!-- ===== EXPORT MODAL ===== -->
<div class="modal-overlay hidden" id="export-modal" onclick="if(event.target===this)closeExportModal()">
  <div class="modal" style="max-width:320px">
    <div class="modal-head">
      <span class="modal-title">⬇️ Xuất hội thoại</span>
      <button class="modal-close" onclick="closeExportModal()">✕</button>
    </div>
    <p style="font-size:.83rem;color:var(--tx2)">Chọn định dạng xuất file:</p>
    <div style="display:flex;gap:10px">
      <button class="btn bp" style="flex:1" onclick="doExport('md')">📝 Markdown (.md)</button>
      <button class="btn bg" style="flex:1" onclick="doExport('json')">📦 JSON (.json)</button>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const g = id => ($(id)?.value||$(id)?.textContent||'').trim();
const si = (id,v) => { $(id).value=v; };
const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const arz = el => { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,130)+'px'; };

let curSid=null, chatMsgs=[], hpVisible=true, _allSessions=[];
let _trStyle='standard', _imgStyle='realistic';

// ===== THEME =====
function toggleTheme(){
  const h=document.documentElement, dark=h.getAttribute('data-theme')==='dark';
  h.setAttribute('data-theme',dark?'light':'dark');
}
function toggleHP(){ hpVisible=!hpVisible; $('hpanel').classList.toggle('hidden',!hpVisible); }
function swapL(){ const s=$('tr-src'),d=$('tr-dst'); [s.value,d.value]=[d.value,s.value]; }

// ===== SWITCH APP =====
function switchApp(name){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.anv').forEach(n=>n.classList.remove('active'));
  $('panel-'+name).classList.add('active');
  $('anav-'+name).classList.add('active');
  $('hpanel').style.visibility = name==='chat'?'':'hidden';
  $('hpanel').style.width      = name==='chat'&&hpVisible?'':'0';
}

// ===== CLIPBOARD =====
async function cp(id){
  const t=$(id).innerText;
  try{ await navigator.clipboard.writeText(t); }
  catch(e){ const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();document.execCommand('copy');document.body.removeChild(a); }
}

// ===== MARKED + HIGHLIGHT.JS setup =====
(function setupMarked(){
  if(typeof marked==='undefined') return;
  const renderer = new marked.Renderer();
  renderer.code = function(code, lang){
    // Normalize: marked v9+ passes object or string
    let codeStr = typeof code === 'object' ? (code.text||'') : code;
    let langStr = typeof code === 'object' ? (code.lang||'') : (lang||'');
    let highlighted = codeStr;
    if(typeof hljs!=='undefined'){
      try{
        highlighted = langStr && hljs.getLanguage(langStr)
          ? hljs.highlight(codeStr,{language:langStr}).value
          : hljs.highlightAuto(codeStr).value;
      }catch(e){ highlighted = esc(codeStr); }
    } else {
      highlighted = esc(codeStr);
    }
    const langLabel = langStr ? `<span class="code-lang">${esc(langStr)}</span>` : '';
    return `<pre><button class="copy-code-btn" onclick="copyCb(this)">📋 Copy</button>${langLabel}<code class="hljs${langStr?' language-'+langStr:''}">${highlighted}</code></pre>`;
  };
  marked.setOptions({renderer, breaks:true, gfm:true});
})();

function copyCb(btn){
  const code = btn.closest('pre').querySelector('code').innerText;
  navigator.clipboard.writeText(code).catch(()=>{});
  const orig = btn.textContent;
  btn.textContent='✅ Copied!'; btn.style.background='var(--gr)'; btn.style.color='#fff';
  setTimeout(()=>{ btn.textContent=orig; btn.style.background=''; btn.style.color=''; }, 1500);
}

function renderMdFull(text){
  if(typeof marked==='undefined') return esc(text);
  let clean = text.replace(/\n{3,}/g,'\n\n').split('\n').map(l=>l.trimEnd()).join('\n');
  return marked.parse(clean);
}

// ===== TTS (Web Speech API) =====
let _ttsUtter=null, _ttsPlaying=null;
function ttsToggle(contentId, btnId){
  const btn=$(btnId);
  if(_ttsPlaying===btnId){
    speechSynthesis.cancel();
    _ttsPlaying=null;
    btn.classList.remove('playing'); btn.textContent='🔊 Đọc';
    return;
  }
  if(_ttsPlaying){ speechSynthesis.cancel(); $(_ttsPlaying)?.classList.remove('playing'); }
  const text=$(contentId)?.innerText?.trim();
  if(!text){ return; }
  _ttsUtter=new SpeechSynthesisUtterance(text);
  _ttsUtter.lang='vi-VN'; _ttsUtter.rate=0.95;
  _ttsUtter.onend=()=>{ _ttsPlaying=null; btn.classList.remove('playing'); btn.textContent='🔊 Đọc'; };
  _ttsUtter.onerror=()=>{ _ttsPlaying=null; btn.classList.remove('playing'); btn.textContent='🔊 Đọc'; };
  speechSynthesis.speak(_ttsUtter);
  _ttsPlaying=btnId; btn.classList.add('playing'); btn.textContent='⏹ Dừng';
}

// ===== PROVIDERS & MODELS =====
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

// ===== HISTORY =====
async function loadHistory(){
  try{
    const ss=await(await fetch('/api/sessions')).json();
    _allSessions=ss;
    renderHistory(ss);
  }catch(e){}
}

function filterHistory(q){
  if(!q.trim()){ renderHistory(_allSessions); return; }
  const lq=q.toLowerCase();
  renderHistory(_allSessions.filter(s=>(s.title||'').toLowerCase().includes(lq)));
}

function renderHistory(ss){
  const el=$('hlist');
  if(!ss.length){ el.innerHTML='<div class="h-empty">Không có kết quả<br>Nhấn ✏️ để bắt đầu</div>'; return; }
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
        <div class="hi-content">${esc(s.title)}<div class="hi-time">${(s.updated_at||'').slice(11,16)}</div></div>
        <span class="hi-del" onclick="event.stopPropagation();delSess('${s.id}')" title="Xóa">✕</span>
      </div>`;
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

async function delSess(id){
  if(!confirm('Xóa cuộc trò chuyện này? Không thể hoàn tác.')) return;
  try{
    await fetch(`/api/sessions/${id}`,{method:'DELETE'});
    if(curSid===id){ curSid=null; chatMsgs=[]; $('chat-msgs').innerHTML='<div class="mrow sys"><div class="bbl">Đã xóa. Nhấn ✏️ để bắt đầu mới.</div></div>'; }
    await loadHistory();
  }catch(e){ console.error(e); }
}

// ===== EXPORT =====
function exportCurrent(){
  if(!curSid){ alert('Vui lòng mở một cuộc trò chuyện trước.'); return; }
  $('export-modal').classList.remove('hidden');
}
function closeExportModal(){ $('export-modal').classList.add('hidden'); }
function doExport(fmt){
  if(!curSid) return;
  window.open(`/api/sessions/${curSid}/export?format=${fmt}`,'_blank');
  closeExportModal();
}

// ===== BUBBLE =====
function addBbl(role,content,time=''){
  const m=$('chat-msgs');
  if(role==='sys'||role==='system'){
    m.innerHTML+=`<div class="mrow sys"><div class="bbl">${esc(content)}</div></div>`;
    return;
  }
  const u=role==='user';
  const av=u?`<div class="av user">👤</div>`:`<div class="av ai">🤖</div>`;
  const rendered = u ? esc(content) : renderMdFull(content);
  const mdClass = u ? '' : ' md-content';
  m.innerHTML+=`<div class="mrow ${u?'user':'ai'}">${u?'':av}
    <div><div class="bbl${mdClass}">${rendered}</div>${time?`<div class="msg-t">${time}</div>`:''}</div>
    ${u?av:''}</div>`;
  m.scrollTop=m.scrollHeight;
}

// ===== FILE ATTACH =====
let _fileInfo=null;
function onFileSelect(input){
  const file=input.files[0]; if(!file) return;
  const prev=$('file-preview');
  prev.style.display='flex';
  prev.innerHTML=`<span style="font-size:.8rem;background:var(--bg3);border:1px solid var(--bd);border-radius:6px;padding:3px 8px;color:var(--tx2);display:flex;align-items:center;gap:6px">
    📄 ${file.name} (${(file.size/1024).toFixed(1)}KB)
    <span onclick="clearFile()" style="cursor:pointer;color:var(--rd);font-weight:600">✕</span>
  </span><span id="upload-status" style="font-size:.75rem;color:var(--tx3)">Đang tải...</span>`;
  const fd=new FormData(); fd.append('file',file);
  fetch('/api/upload',{method:'POST',body:fd})
    .then(r=>r.json()).then(d=>{
      if(d.error){ $('upload-status').textContent='❌ '+d.error; $('upload-status').style.color='var(--rd)'; _fileInfo=null; }
      else { $('upload-status').textContent='✅ Sẵn sàng'; $('upload-status').style.color='var(--gr)'; _fileInfo=d; }
    }).catch(()=>{ $('upload-status').textContent='❌ Lỗi upload'; _fileInfo=null; });
  input.value='';
}
function clearFile(){ _fileInfo=null; const p=$('file-preview'); p.style.display='none'; p.innerHTML=''; }

// ===== SEND CHAT (Streaming) =====
async function sendChat(){
  const inp=$('chat-input'), msg=inp.value.trim();
  if(!msg && !_fileInfo) return;

  // Tạo session mới nếu chưa có
  if(!curSid){
    const title=msg||(_fileInfo?'File: '+_fileInfo.name:'Cuộc trò chuyện mới');
    const r=await fetch('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title:title.slice(0,60),system_prompt:$('chat-system').value.trim(),
        provider:$('gp').value,model:$('gm').value})});
    curSid=(await r.json()).id;
    // Auto-title sau 1 tin nhắn đầu
    if(msg) fetch(`/api/sessions/${curSid}/autotitle`,{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,provider:$('gp').value,model:$('gm').value})})
      .then(()=>loadHistory());
  }

  const now=new Date().toLocaleTimeString('vi',{hour:'2-digit',minute:'2-digit'});
  const userLabel=msg+(_fileInfo?'\n📎 '+_fileInfo.name:'');
  addBbl('user',userLabel,now);
  chatMsgs.push({role:'user',content:msg||`[File: ${_fileInfo?.name}]`});
  inp.value=''; inp.style.height='auto';
  const fileToSend=_fileInfo; clearFile();
  $('sbtn').disabled=true;

  // Bubble AI
  const m=$('chat-msgs');
  const rowId='trow_'+Date.now();
  m.innerHTML+=`<div class="mrow ai" id="${rowId}"><div class="av ai">🤖</div>
    <div><div class="bbl md-content" id="bbl_${rowId}"><div class="tyd"><span></span><span></span><span></span></div></div>
    <div class="msg-t" id="ts_${rowId}"></div></div></div>`;
  m.scrollTop=m.scrollHeight;

  let fullText='', renderTimer=null;
  const bbl=()=>$('bbl_'+rowId);

  function renderMdStream(){
    if(renderTimer) return;
    renderTimer=setTimeout(()=>{
      renderTimer=null;
      const el=bbl(); if(!el) return;
      el.innerHTML=renderMdFull(fullText)+'<span class="stream-cursor">▋</span>';
      m.scrollTop=m.scrollHeight;
    },60);
  }

  function finalRender(){
    if(renderTimer){ clearTimeout(renderTimer); renderTimer=null; }
    const el=bbl(); if(!el) return;
    el.innerHTML=renderMdFull(fullText);
    const tsEl=$('ts_'+rowId); if(tsEl) tsEl.textContent=new Date().toLocaleTimeString('vi',{hour:'2-digit',minute:'2-digit'});
  }

  try{
    const resp=await fetch('/api/chat/stream',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,history:chatMsgs,file:fileToSend,
        system:$('chat-system').value.trim()||'Bạn là trợ lý AI hữu ích.',
        provider:$('gp').value,model:$('gm').value,session_id:curSid})});
    if(!resp.ok||!resp.body) throw new Error('no stream');
    const reader=resp.body.getReader(), decoder=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read(); if(done) break;
      buf+=decoder.decode(value,{stream:true});
      const lines=buf.split('\n'); buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data:')) continue;
        try{
          const ev=JSON.parse(line.slice(5).trim());
          if(ev.chunk!==undefined){ fullText+=ev.chunk; renderMdStream(); }
          else if(ev.done){ finalRender(); chatMsgs.push({role:'assistant',content:fullText}); await loadHistory(); }
          else if(ev.error){ bbl().innerHTML='❌ '+esc(ev.error); }
        }catch(_){}
      }
    }
  }catch(e){
    // Fallback non-streaming
    bbl().innerHTML='<div class="tyd"><span></span><span></span><span></span></div>';
    try{
      const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:msg,history:chatMsgs,file:fileToSend,
          system:$('chat-system').value.trim()||'Bạn là trợ lý AI hữu ích.',
          provider:$('gp').value,model:$('gm').value,session_id:curSid})});
      const d=await r.json(); fullText=d.reply||'(Không có phản hồi)';
      finalRender(); chatMsgs.push({role:'assistant',content:fullText}); await loadHistory();
    }catch(e2){ bbl().textContent='❌ Lỗi: '+e2.message; }
  }
  $('sbtn').disabled=false;
}

// ===== TC (tool call) =====
async function tc(ep,payload,btnId,outId){
  if(Object.values(payload).some(v=>!v)) return;
  const btn=$(btnId), out=$(outId);
  btn.dataset.orig=btn.innerHTML;
  btn.innerHTML='<span class="sp"></span> Đang xử lý...'; btn.disabled=true;
  out.textContent='⏳ Đang xử lý...'; out.classList.remove('ph');
  payload.provider=$('gp').value; payload.model=$('gm').value;
  try{
    const r=await fetch('/api/'+ep,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    out.textContent=d.result||d.error||'(Không có kết quả)';
  }catch(e){ out.textContent='❌ Lỗi: '+e.message; }
  btn.innerHTML=btn.dataset.orig; btn.disabled=false;
}

// ===== TRANSLATE with style =====
function setTrStyle(s){
  _trStyle=s;
  ['standard','game','literary'].forEach(x=>{
    const el=$('style-'+x);
    if(el){ el.style.borderColor=x===s?'var(--ac)':''; el.style.color=x===s?'var(--ac2)':''; }
  });
}
async function doTranslate(){
  const text=g('tr-in'), src=g('tr-src'), dst=g('tr-dst');
  if(!text) return;
  const btn=$('tr-btn'), out=$('tr-out');
  btn.dataset.orig=btn.innerHTML;
  btn.innerHTML='<span class="sp"></span> Đang dịch...'; btn.disabled=true;
  out.textContent='⏳ Đang dịch...'; out.classList.remove('ph');
  try{
    const r=await fetch('/api/translate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text,src,dst,style:_trStyle,provider:$('gp').value,model:$('gm').value})});
    const d=await r.json();
    out.textContent=d.result||d.error||'';
  }catch(e){ out.textContent='❌ Lỗi: '+e.message; }
  btn.innerHTML=btn.dataset.orig; btn.disabled=false;
}

// ===== IMAGE PROMPT =====
function setImgStyle(s){
  _imgStyle=s;
  ['realistic','anime','art','sdxl'].forEach(x=>{
    const el=$('img-style-'+x);
    if(el){ el.style.borderColor=x===s?'var(--ac)':''; el.style.color=x===s?'var(--ac2)':''; }
  });
}
async function doImgPrompt(){
  const idea=g('img-in'); if(!idea) return;
  const btn=$('img-btn'), pos=$('img-pos'), neg=$('img-neg');
  btn.dataset.orig=btn.innerHTML;
  btn.innerHTML='<span class="sp"></span> Đang tạo...'; btn.disabled=true;
  pos.textContent='⏳ Đang xử lý...'; neg.textContent='';
  pos.classList.remove('ph'); neg.classList.remove('ph');
  try{
    const r=await fetch('/api/imgprompt',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({idea,style:_imgStyle,provider:$('gp').value,model:$('gm').value})});
    const d=await r.json();
    pos.textContent=d.positive||'(Không có kết quả)';
    neg.textContent=d.negative||'';
  }catch(e){ pos.textContent='❌ Lỗi: '+e.message; }
  btn.innerHTML=btn.dataset.orig; btn.disabled=false;
}

// ===== TEMPLATES =====
let _templates=[];
async function loadTemplates(){
  try{ _templates=await(await fetch('/api/templates')).json(); }catch(e){ _templates=[]; }
}
function openTplModal(){
  renderTplGrid();
  $('tpl-modal').classList.remove('hidden');
}
function closeTplModal(){ $('tpl-modal').classList.add('hidden'); }
function renderTplGrid(){
  const curPrompt=$('chat-system').value.trim();
  $('tpl-grid').innerHTML=_templates.map(t=>{
    const active=curPrompt===t.system_prompt?'active-tpl':'';
    const canDel=!t.id.startsWith('tpl-');
    return `<div class="tpl-card ${active}" onclick="applyTpl('${t.id}')">
      ${canDel?`<button class="tpl-del" onclick="event.stopPropagation();deleteTpl('${t.id}')" title="Xóa">✕</button>`:''}
      <div class="tpl-icon">${t.icon||'🤖'}</div>
      <div class="tpl-name">${esc(t.name)}</div>
      <div class="tpl-desc">${esc(t.description||'')}</div>
    </div>`;
  }).join('');
}
function applyTpl(id){
  const t=_templates.find(x=>x.id===id); if(!t) return;
  $('chat-system').value=t.system_prompt;
  closeTplModal();
}
async function deleteTpl(id){
  if(!confirm('Xóa template này?')) return;
  await fetch(`/api/templates/${id}`,{method:'DELETE'});
  await loadTemplates(); renderTplGrid();
}
async function saveTpl(){
  const name=g('new-tpl-name'), prompt=g('new-tpl-prompt');
  if(!name||!prompt){ alert('Vui lòng nhập tên và system prompt.'); return; }
  await fetch('/api/templates',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name,icon:$('new-tpl-icon').value||'🤖',system_prompt:prompt,description:g('new-tpl-desc')})});
  $('new-tpl-name').value=''; $('new-tpl-prompt').value=''; $('new-tpl-desc').value=''; $('new-tpl-icon').value='';
  await loadTemplates(); renderTplGrid();
}

// ===== INIT =====
window.onload=async()=>{
  await initProviders();
  await loadTemplates();
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

    reply = llm_call(msgs, d, max_tokens=4096, temperature=0.7)

    if sid:
        ts = now_str()
        label = user_msg + (f" [+{file_info['name']}]" if file_info else "")
        db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)",
                   (sid, 'user', label, ts))
        db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)",
                   (sid, 'assistant', reply, ts))
        db_execute("UPDATE sessions SET updated_at=? WHERE id=?", (ts, sid))
    return jsonify({"reply": reply})

@app.route('/api/chat/stream', methods=['POST'])
def api_chat_stream():
    """SSE endpoint: stream AI response từng chunk."""
    d         = request.json
    sid       = d.get('session_id')
    user_msg  = d.get('message', '')
    file_info = d.get('file')
    provider  = d.get('provider', 'lmstudio')
    system    = d.get("system", "Ban la tro ly AI huu ich.")

    msgs = [{"role": "system", "content": system}]
    msgs.extend(d.get("history", []))

    if file_info:
        fname = file_info.get("name", "file")
        if file_info.get("is_image"):
            mime = file_info["type"]
            b64  = file_info["content"]
            user_content = [
                {"type": "text",      "text": user_msg or f"Phan tich anh: {fname}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]
            msgs.append({"role": "user", "content": user_content})
        else:
            ftext  = file_info["content"]
            block  = f"[File: {fname}]\n```\n{ftext}\n```"
            prompt = block + "\n\n" + user_msg if user_msg else block + "\n\nHay phan tich noi dung file nay."
            msgs.append({"role": "user", "content": prompt})
    else:
        msgs.append({"role": "user", "content": user_msg})

    def generate():
        full_reply = []
        try:
            for chunk in llm_call_stream(msgs, d, max_tokens=4096, temperature=0.7):
                full_reply.append(chunk)
                # SSE format: data: <json>\n\n
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Luu vao DB sau khi stream xong
            reply_text = "".join(full_reply)
            if sid and reply_text:
                ts = now_str()
                label = user_msg + (f" [+{file_info['name']}]" if file_info else "")
                db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)",
                           (sid, 'user', label, ts))
                db_execute("INSERT INTO messages (session_id,role,content,created_at) VALUES (?,?,?,?)",
                           (sid, 'assistant', reply_text, ts))
                db_execute("UPDATE sessions SET updated_at=? WHERE id=?", (ts, sid))
            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/api/optimize_prompt', methods=['POST'])
def api_optimize():
    d=request.json
    r=llm_call([{"role":"system","content":"Bạn là chuyên gia Prompt Engineering. Viết lại thành prompt chuyên nghiệp, chi tiết. Chỉ trả về prompt, không giải thích."},
                {"role":"user","content":f"Yêu cầu: {d['text']}"}],d,max_tokens=1024,temperature=0.5)
    return jsonify({"result":r})

"""@app.route('/api/translate', methods=['POST'])
def api_translate():
    d=request.json
    r=llm_call([{"role":"system","content":f"Translate to {d['dst']}. Output ONLY the translation."},
                {"role":"user","content":f"Translate from {d['src']}:\n{d['text']}"}],d,max_tokens=1024,temperature=0.3)
    return jsonify({"result":r})"""

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

# ===== TEMPLATES =====
@app.route('/api/templates', methods=['GET'])
def get_templates():
    rows = db_fetchall("SELECT * FROM templates ORDER BY created_at ASC")
    return jsonify(rows)

@app.route('/api/templates', methods=['POST'])
def create_template():
    d = request.json
    tid = str(uuid.uuid4()); ts = now_str()
    db_execute("INSERT INTO templates VALUES (?,?,?,?,?,?)",
        (tid, d.get('name','Template mới'), d.get('icon','🤖'),
         d.get('system_prompt',''), d.get('description',''), ts))
    return jsonify({"id": tid, "name": d.get('name')})

@app.route('/api/templates/<tid>', methods=['DELETE'])
def delete_template(tid):
    if tid.startswith('tpl-'):
        return jsonify({"error": "Không thể xóa template mặc định"}), 400
    db_execute("DELETE FROM templates WHERE id=?", (tid,))
    return jsonify({"ok": True})

# ===== AUTO-TITLE =====
@app.route('/api/sessions/<sid>/autotitle', methods=['POST'])
def auto_title(sid):
    d = request.json
    first_msg = d.get('message', '')
    if not first_msg:
        return jsonify({"title": "Cuộc trò chuyện mới"})
    title = llm_call([
        {"role":"system","content":"Tóm tắt câu hỏi sau thành tiêu đề ngắn gọn tối đa 6 từ, không dùng dấu câu, không giải thích. Chỉ trả về tiêu đề."},
        {"role":"user","content": first_msg[:300]}
    ], d, max_tokens=30, temperature=0.3)
    title = title.strip().strip('"').strip("'")[:60] or "Cuộc trò chuyện mới"
    db_execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))
    return jsonify({"title": title})

# ===== EXPORT =====
@app.route('/api/sessions/<sid>/export', methods=['GET'])
def export_session(sid):
    fmt = request.args.get('format', 'md')
    s  = db_fetchone("SELECT * FROM sessions WHERE id=?", (sid,))
    ms = db_fetchall("SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,))
    if not s:
        return jsonify({"error": "Not found"}), 404
    if fmt == 'json':
        data = json.dumps({"session": s, "messages": ms}, ensure_ascii=False, indent=2)
        return Response(data, mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename="chat_{sid[:8]}.json"'})
    # Markdown
    lines = [f"# {s['title']}\n",
             f"> Provider: {s.get('provider','')} | Model: {s.get('model','')}",
             f"> Thời gian: {s.get('created_at','')}\n"]
    if s.get('system_prompt'):
        lines += [f"**System Prompt:** {s['system_prompt']}\n", "---\n"]
    for m in ms:
        role_label = "👤 **Bạn**" if m['role']=='user' else "🤖 **AI**"
        lines.append(f"{role_label}  \n{m['content']}\n")
        lines.append("---\n")
    md = "\n".join(lines)
    return Response(md, mimetype='text/markdown',
        headers={'Content-Disposition': f'attachment; filename="chat_{sid[:8]}.md"'})

# ===== IMAGE PROMPT =====
@app.route('/api/imgprompt', methods=['POST'])
def api_imgprompt():
    d = request.json
    style = d.get('style', 'realistic')
    idea  = d.get('idea', '')
    sys_map = {
        'realistic': 'photorealistic, hyperdetailed, 8k',
        'anime':     'anime style, cel shading, vibrant colors',
        'art':       'digital art, painterly, artistic masterpiece',
        'sdxl':      'SDXL optimized, cinematic lighting',
    }
    style_hint = sys_map.get(style, sys_map['realistic'])
    r = llm_call([
        {"role":"system","content":f"""You are an expert Stable Diffusion / FLUX prompt engineer.
Given a rough idea, output ONLY a JSON object with two keys:
"positive": detailed English prompt optimized for {style_hint}, comma-separated tags, max 120 words
"negative": negative prompt listing things to avoid, comma-separated, max 60 words
No markdown, no explanation, only valid JSON."""},
        {"role":"user","content": f"Idea: {idea}"}
    ], d, max_tokens=512, temperature=0.7)
    try:
        clean = r.strip().strip('`').removeprefix('json').strip()
        parsed = json.loads(clean)
        return jsonify({"positive": parsed.get("positive",""), "negative": parsed.get("negative","")})
    except:
        return jsonify({"positive": r, "negative": ""})

# ===== TRANSLATE (nâng cấp hỗ trợ style) =====
@app.route('/api/translate', methods=['POST'])
def api_translate():
    d    = request.json
    style = d.get('style', 'standard')
    sys_prompts = {
        'standard': f"Translate to {d['dst']}. Output ONLY the translation, preserve formatting.",
        'game':     f"Translate to {d['dst']} for a video game/visual novel. KEEP ALL CODE TAGS intact (e.g. {{player}}, [name], renpy syntax). Output ONLY the translated text.",
        'literary': f"Translate to {d['dst']} with literary flair. Use poetic, expressive language. Capture the emotional nuance and cultural context. Output ONLY the translation.",
    }
    sys = sys_prompts.get(style, sys_prompts['standard'])
    r = llm_call([{"role":"system","content":sys},
                  {"role":"user","content":f"Translate from {d['src']}:\n{d['text']}"}],
                 d, max_tokens=2048, temperature=0.4)
    return jsonify({"result": r})

if __name__=='__main__':
    # init_db() đã chạy ở module level — không cần gọi lại
    import socket
    ip=socket.gethostbyname(socket.gethostname())
    print(f"\n{'='*46}\n  AI Apps v3.0 — SQLite Chat History\n{'='*46}")
    print(f"  Local : http://localhost:5000")
    print(f"  LAN   : http://{ip}:5000")
    print(f"  DB    : {_db_path if not USE_TURSO else TURSO_URL}")
    print(f"\n  Deploy Render: Thêm biến môi trường:")
    print(f"    GROQ_API_KEY, OPENROUTER_API_KEY, APP_PASSWORD")
    print(f"    DB_PATH=/var/data/chat.db  (nếu dùng Render Disk)")
    print(f"{'='*46}\n")
    app.run(host='0.0.0.0',port=5000,debug=False)
