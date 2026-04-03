#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lmstudio_webapp.py
==================
Giao diện web cho LM Studio & Các API khác (Groq, OpenRouter).
"""

from flask import Flask, request, jsonify
from openai import OpenAI
import json, os, re, time

try:
    from config import LM_STUDIO_URL, REQUEST_TIMEOUT, GROQ_API_KEY, OPENROUTER_API_KEY
except ImportError:
    LM_STUDIO_URL   = "http://127.0.0.1:1234"
    REQUEST_TIMEOUT = 120
    # ĐIỀN API KEY CỦA BẠN VÀO ĐÂY NẾU KHÔNG DÙNG FILE config.py
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY") 
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

app = Flask(__name__)

# Cấu hình đa nhà cung cấp (Multi-Provider)
PROVIDERS = {
    "lmstudio": {
        "name": "LM Studio (Local)",
        "base_url": f"{LM_STUDIO_URL}/v1",
        "api_key": "lm-studio",
        "default_model": "local-model"
    },
    "groq": {
        "name": "Groq (Cloud - Fast)",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "default_model": "llama3-70b-8192"
    },
    "openrouter": {
        "name": "OpenRouter (Cloud - Multi)",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENROUTER_API_KEY,
        "default_model": "google/gemini-pro-1.5"
    }
}

def get_client(provider_key="lmstudio"):
    cfg = PROVIDERS.get(provider_key, PROVIDERS["lmstudio"])
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

def parse_reply(content):
    if not content: return ""
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    for tok in ["<|END_RESPONSE|>","<|end_of_turn|>","<|eot_id|>","<|endoftext|>","<|im_end|>"]:
        text = text.replace(tok, "")
    return text.strip()

# ==========================================
# ===== GIAO DIỆN HTML/CSS/JS (Cập nhật) =====
# ==========================================
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Multi-Provider Apps</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0f0f13; var(--surface): #17171e; var(--surface2): #1e1e28;
  --border: #2a2a38; var(--accent): #7c6af7; var(--accent2): #a78bfa;
  --green: #4ade80; var(--red): #f87171; var(--amber): #fbbf24;
  --text: #e8e6f0; var(--muted): #6b6880;
  --mono: 'DM Mono', monospace; var(--sans): 'DM Sans', sans-serif;
  --surface: #17171e; --surface2: #1e1e28; --accent: #7c6af7; --accent2: #a78bfa;
  --red: #f87171; --amber: #fbbf24; --muted: #6b6880;
  --sans: 'DM Sans', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; display: flex; flex-direction: column; }

/* HEADER */
header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 2rem; height: 60px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
.logo { font-family: var(--mono); font-size: 1.1rem; color: var(--accent2); letter-spacing: -0.5px; font-weight: 600;}
.logo span { color: var(--text); }
.header-controls { display: flex; gap: 12px; align-items: center; }
select.header-select { background: var(--surface2); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 4px 8px; font-size: 0.85rem; height: 32px; outline: none;}

#status-badge { display: flex; align-items: center; gap: 8px; font-size: 0.78rem; font-family: var(--mono); color: var(--muted); background: var(--surface2); border: 1px solid var(--border); padding: 4px 12px; border-radius: 20px; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.dot.online { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

/* LAYOUT & SIDEBAR */
.layout { display: flex; flex: 1; height: calc(100vh - 60px); overflow: hidden; }
.sidebar { width: 220px; min-width: 220px; background: var(--surface); border-right: 1px solid var(--border); padding: 1.5rem 0; overflow-y: auto; }
.sidebar-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); padding: 0 1.2rem; margin-bottom: 0.5rem; }
.nav-item { display: flex; align-items: center; gap: 10px; padding: 0.6rem 1.2rem; cursor: pointer; color: var(--muted); font-size: 0.88rem; font-weight: 400; border-left: 3px solid transparent; transition: all 0.15s; }
.nav-item:hover { background: var(--surface2); color: var(--text); }
.nav-item.active { background: var(--surface2); color: var(--accent2); border-left-color: var(--accent); }
.nav-icon { font-size: 1rem; width: 20px; text-align: center; }

/* MAIN & CARDS */
.main { flex: 1; overflow-y: auto; padding: 2rem; }
.panel { display: none; max-width: 860px; margin: 0 auto; animation: fadeIn 0.2s ease; }
.panel.active { display: block; }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
h2 { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.3rem; }
.subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem; }
.card-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin-bottom: 0.6rem; display:block; }

/* INPUTS & BUTTONS */
textarea, input[type=text], select { width: 100%; background: var(--surface2); border: 1px solid var(--border); color: var(--text); border-radius: 8px; padding: 0.7rem 0.9rem; font-family: var(--sans); font-size: 0.9rem; resize: vertical; outline: none; transition: border-color 0.15s; }
textarea:focus, input[type=text]:focus, select:focus { border-color: var(--accent); }
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 7px; padding: 0.6rem 1.2rem; border-radius: 8px; border: none; font-family: var(--sans); font-size: 0.88rem; font-weight: 500; cursor: pointer; transition: all 0.15s; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent2); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-ghost { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-ghost:hover { border-color: var(--accent); color: var(--accent2); }
.btn-row { display: flex; gap: 8px; margin-top: 0.8rem; flex-wrap: wrap; }

/* OUTPUT & CHAT */
.output { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; font-family: var(--mono); font-size: 0.84rem; line-height: 1.7; color: var(--text); white-space: pre-wrap; word-break: break-word; min-height: 80px; max-height: 520px; overflow-y: auto; }
.output.empty { color: var(--muted); font-style: italic; }
.chat-history { background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; height: 360px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; margin-bottom: 1rem; }
.msg { display: flex; gap: 10px; animation: fadeIn 0.15s ease; }
.msg-bubble { padding: 0.6rem 0.9rem; border-radius: 10px; font-size: 0.88rem; line-height: 1.6; max-width: 85%; white-space: pre-wrap; word-break: break-word; }
.msg.user { justify-content: flex-end; }
.msg.user .msg-bubble { background: var(--accent); color: #fff; border-radius: 10px 10px 2px 10px; }
.msg.ai .msg-bubble { background: var(--surface); border: 1px solid var(--border); border-radius: 10px 10px 10px 2px; }
.msg.system .msg-bubble { background: transparent; color: var(--muted); font-size: 0.8rem; font-style: italic; text-align: center; width: 100%; border: none; padding: 0; }
.chat-input-row { display: flex; gap: 8px; }
.chat-input-row textarea { min-height: 44px; max-height: 120px; resize: none; flex: 1; }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.3); border-top-color: #fff; border-radius: 50%; animation: spin .6s linear infinite; }
@keyframes spin { to{transform:rotate(360deg)} }
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<header>
  <div class="logo">AI<span>.apps</span></div>
  <div class="header-controls">
    <select id="global-provider" class="header-select" onchange="fetchModels()">
      </select>
    <select id="global-model" class="header-select" style="max-width: 150px;">
      <option value="">Loading...</option>
    </select>
    <div id="status-badge">
      <div class="dot" id="status-dot"></div>
      <span id="status-text">Đang kết nối...</span>
    </div>
  </div>
</header>

<div class="layout">
  <nav class="sidebar">
    <div class="sidebar-label">Ứng dụng</div>
    <div class="nav-item active" onclick="showPanel('chat')" id="nav-chat"><span class="nav-icon">💬</span> Chatbot</div>
    <div class="nav-item" onclick="showPanel('optimizer')" id="nav-optimizer"><span class="nav-icon">✨</span> Tối ưu Prompt</div>
    <div class="nav-item" onclick="showPanel('translate')" id="nav-translate"><span class="nav-icon">🌐</span> Dịch thuật</div>
    <div class="nav-item" onclick="showPanel('review')" id="nav-review"><span class="nav-icon">🔍</span> Code Review</div>
    <div class="nav-item" onclick="showPanel('summary')" id="nav-summary"><span class="nav-icon">📄</span> Tóm tắt</div>
    <div class="nav-item" onclick="showPanel('mockdata')" id="nav-mockdata"><span class="nav-icon">🗄️</span> Mock Data</div>
    <div class="nav-item" onclick="showPanel('terminal')" id="nav-terminal"><span class="nav-icon">⌨️</span> Terminal</div>
  </nav>

  <main class="main">

    <div class="panel active" id="panel-chat">
      <h2>💬 Chatbot</h2>
      <p class="subtitle">Hội thoại đa lượt với AI — nhớ lịch sử trong phiên làm việc</p>
      <div class="card">
        <label class="card-label">System Prompt</label>
        <textarea id="chat-system" rows="2" placeholder="Bạn là trợ lý AI hữu ích..."></textarea>
      </div>
      <div class="chat-history" id="chat-history">
        <div class="msg system"><div class="msg-bubble">Bắt đầu cuộc trò chuyện...</div></div>
      </div>
      <div class="chat-input-row">
        <textarea id="chat-input" placeholder="Nhập tin nhắn..." rows="2" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat()}"></textarea>
        <button class="btn btn-primary" onclick="sendChat()" id="chat-btn">Gửi</button>
      </div>
      <button class="btn btn-ghost" style="margin-top:10px" onclick="clearChat()">🗑 Xóa lịch sử</button>
    </div>

    <div class="panel" id="panel-optimizer">
      <h2>✨ Tối ưu Prompt (Prompt Engineering)</h2>
      <p class="subtitle">Biến yêu cầu sơ sài của bạn thành một Prompt chuyên nghiệp và chi tiết.</p>
      <div class="card">
        <label class="card-label">Ý tưởng ban đầu của bạn</label>
        <textarea id="opt-input" rows="4" placeholder="VD: Viết cho tôi một bài post facebook bán áo thun."></textarea>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="doOptimize()" id="opt-btn">Tối ưu hóa</button>
        </div>
      </div>
      <div class="card">
        <label class="card-label">Prompt đã được tối ưu</label>
        <div class="output empty" id="opt-output">Prompt chuyên nghiệp sẽ hiển thị ở đây...</div>
        <div class="btn-row">
          <button class="btn btn-ghost" onclick="copyOutput('opt-output')">📋 Sao chép</button>
        </div>
      </div>
    </div>

    <div class="panel" id="panel-translate">
      <h2>🌐 Dịch thuật</h2>
      <p class="subtitle">Dịch văn bản chuyên sâu</p>
      <div class="card">
        <div style="display:flex;gap:10px;margin-bottom:1rem">
          <select id="tr-src"><option value="English">English</option><option value="Vietnamese">Tiếng Việt</option></select>
          <button class="btn btn-ghost" onclick="let s=document.getElementById('tr-src'),d=document.getElementById('tr-dst');[s.value,d.value]=[d.value,s.value]">⇄</button>
          <select id="tr-dst"><option value="Vietnamese">Tiếng Việt</option><option value="English">English</option></select>
        </div>
        <textarea id="tr-input" rows="5" placeholder="Nhập văn bản cần dịch..."></textarea>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="callApi('translate', {text: val('tr-input'), src: val('tr-src'), dst: val('tr-dst')}, 'tr-btn', 'tr-output')">Dịch</button>
        </div>
      </div>
      <div class="card"><div class="output empty" id="tr-output">Kết quả...</div></div>
    </div>

    <div class="panel" id="panel-review">
      <h2>🔍 Code Review</h2>
      <div class="card">
        <textarea id="review-code" rows="8" placeholder="Paste code vào đây..."></textarea>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="callApi('review', {code: val('review-code')}, 'review-btn', 'review-output')" id="review-btn">Review</button>
        </div>
      </div>
      <div class="card"><div class="output empty" id="review-output">Kết quả...</div></div>
    </div>

    <div class="panel" id="panel-summary">
      <h2>📄 Tóm tắt văn bản</h2>
      <div class="card">
        <textarea id="summary-input" rows="8" placeholder="Paste văn bản..."></textarea>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="callApi('summary', {text: val('summary-input')}, 'summary-btn', 'summary-output')" id="summary-btn">Tóm tắt</button>
        </div>
      </div>
      <div class="card"><div class="output empty" id="summary-output">Kết quả...</div></div>
    </div>

    <div class="panel" id="panel-mockdata">
      <h2>🗄️ Sinh Mock Data JSON</h2>
      <div class="card">
        <input type="text" id="mock-schema" placeholder="VD: 5 users với email, phone, address">
        <div class="btn-row">
          <button class="btn btn-primary" onclick="callApi('mockdata', {schema: val('mock-schema')}, 'mock-btn', 'mock-output')" id="mock-btn">Tạo dữ liệu</button>
        </div>
      </div>
      <div class="card"><div class="output empty" id="mock-output">Kết quả...</div></div>
    </div>

    <div class="panel" id="panel-terminal">
      <h2>⌨️ Trợ lý Terminal</h2>
      <div class="card">
        <textarea id="term-input" rows="4" placeholder="Nhập lỗi hoặc yêu cầu lệnh..."></textarea>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="callApi('terminal', {input: val('term-input')}, 'term-btn', 'term-output')" id="term-btn">Phân tích</button>
        </div>
      </div>
      <div class="card"><div class="output empty" id="term-output">Kết quả...</div></div>
    </div>

  </main>
</div>

<script>
// ===== UTILS =====
const val = id => document.getElementById(id).value.trim();
function showPanel(name) {
  document.querySelectorAll('.panel, .nav-item').forEach(e => e.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
}
function setLoading(btnId, isLoading) {
  const btn = document.getElementById(btnId);
  if (isLoading) { btn.dataset.orig = btn.innerHTML; btn.innerHTML = '<span class="spinner"></span>...'; btn.disabled = true; }
  else { btn.innerHTML = btn.dataset.orig; btn.disabled = false; }
}
async function copyOutput(id) { navigator.clipboard.writeText(document.getElementById(id).innerText); }

// ===== PROVIDERS & MODELS =====
async function initProviders() {
  try {
    const res = await fetch('/api/providers');
    const data = await res.json();
    const sel = document.getElementById('global-provider');
    sel.innerHTML = '';
    for (const [k, v] of Object.entries(data)) {
      sel.innerHTML += `<option value="${k}">${v.name}</option>`;
    }
    fetchModels();
  } catch(e) {}
}

async function fetchModels() {
  const provider = val('global-provider');
  const selModel = document.getElementById('global-model');
  selModel.innerHTML = '<option value="">Loading...</option>';
  document.getElementById('status-dot').className = 'dot';
  document.getElementById('status-text').innerText = 'Đang tải model...';
  
  try {
    const res = await fetch(`/api/models?provider=${provider}`);
    const data = await res.json();
    selModel.innerHTML = '';
    
    if(data.models && data.models.length > 0) {
      data.models.forEach(m => {
        const isSelected = m === data.default ? 'selected' : '';
        selModel.innerHTML += `<option value="${m}" ${isSelected}>${m}</option>`;
      });
      document.getElementById('status-dot').className = 'dot online';
      document.getElementById('status-text').innerText = 'Sẵn sàng';
    } else {
      selModel.innerHTML = `<option value="${data.default}">${data.default} (Manual)</option>`;
      document.getElementById('status-dot').className = 'dot';
      document.getElementById('status-text').innerText = 'Kiểm tra API Key!';
    }
  } catch(e) {
    document.getElementById('status-text').innerText = 'Lỗi kết nối';
  }
}

// ===== CORE API CALLER =====
async function callApi(endpoint, payload, btnId, outId) {
  if (Object.values(payload).some(v => !v)) return;
  setLoading(btnId, true);
  document.getElementById(outId).textContent = '⏳ Đang xử lý...';
  document.getElementById(outId).classList.remove('empty');
  
  // Gắn thêm Provider và Model vào request
  payload.provider = val('global-provider');
  payload.model = val('global-model');

  try {
    const r = await fetch('/api/' + endpoint, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    document.getElementById(outId).textContent = d.result || d.error || '(Không có kết quả)';
  } catch(e) { document.getElementById(outId).textContent = '❌ Lỗi: ' + e.message; }
  setLoading(btnId, false);
}

// ===== CHAT APP =====
const chatHistory = [];
function addMsg(role, content) {
  const h = document.getElementById('chat-history');
  h.innerHTML += `<div class="msg ${role}"><div class="msg-bubble">${content.replace(/</g,'&lt;')}</div></div>`;
  h.scrollTop = h.scrollHeight;
}
async function sendChat() {
  const msg = val('chat-input'); if (!msg) return;
  addMsg('user', msg); chatHistory.push({role: 'user', content: msg});
  document.getElementById('chat-input').value = '';
  setLoading('chat-btn', true);
  
  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        message: msg, history: chatHistory, system: val('chat-system'),
        provider: val('global-provider'), model: val('global-model')
      })
    });
    const d = await r.json();
    addMsg('ai', d.reply); chatHistory.push({role: 'assistant', content: d.reply});
  } catch(e) { addMsg('system', 'Lỗi: ' + e.message); }
  setLoading('chat-btn', false);
}
function clearChat() { chatHistory.length = 0; document.getElementById('chat-history').innerHTML = ''; }

// ===== NEW APP: OPTIMIZER =====
function doOptimize() {
  callApi('optimize_prompt', { text: val('opt-input') }, 'opt-btn', 'opt-output');
}

// INIT
window.onload = initProviders;
</script>
</body>
</html>
"""

# ==========================================
# ===== BACKEND ROUTES (Cập nhật) ==========
# ==========================================

@app.route('/')
def index():
    return HTML

@app.route('/api/providers')
def api_providers():
    # Gửi cấu hình providers xuống frontend (ẩn API Key)
    safe_providers = {k: {"name": v["name"], "default": v["default_model"]} for k, v in PROVIDERS.items()}
    return jsonify(safe_providers)

@app.route('/api/models')
def api_models():
    provider = request.args.get('provider', 'lmstudio')
    cfg = PROVIDERS.get(provider, PROVIDERS["lmstudio"])
    default_mod = cfg["default_model"]
    
    try:
        client = get_client(provider)
        models_data = client.models.list().data
        model_ids = [m.id for m in models_data]
        # Ưu tiên xếp model mặc định lên đầu nếu có
        if default_mod in model_ids:
            model_ids.insert(0, model_ids.pop(model_ids.index(default_mod)))
        return jsonify({"models": model_ids, "default": default_mod})
    except Exception as e:
        print(f"Lỗi fetch models từ {provider}:", e)
        return jsonify({"models": [], "default": default_mod})

def llm_call(messages, d, max_tokens=1024, temperature=0.7):
    try:
        provider = d.get('provider', 'lmstudio')
        model = d.get('model') or PROVIDERS[provider]["default_model"]
        client = get_client(provider)
        
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature
        )
        return parse_reply(resp.choices[0].message.content or "")
    except Exception as e:
        return f"Lỗi API ({provider}): {str(e)}"

@app.route('/api/chat', methods=['POST'])
def api_chat():
    d = request.json
    messages = [{"role": "system", "content": d.get("system", "Bạn là trợ lý AI hữu ích.")}]
    messages.extend(d.get("history", []))
    reply = llm_call(messages, d, max_tokens=1024, temperature=0.7)
    return jsonify({"reply": reply})

@app.route('/api/optimize_prompt', methods=['POST'])
def api_optimize_prompt():
    d = request.json
    system = """Bạn là một Chuyên gia Prompt Engineering. Nhiệm vụ của bạn là nhận một yêu cầu cơ bản từ người dùng và viết lại thành một prompt hoàn chỉnh, chi tiết, đóng vai xuất sắc để AI khác đọc có thể hiểu và làm tốt nhất. 
    Chỉ trả về Prompt đã tối ưu (được viết bằng ngôn ngữ của yêu cầu gốc), không giải thích thêm."""
    result = llm_call([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Yêu cầu gốc: {d['text']}"}
    ], d, max_tokens=1024, temperature=0.5)
    return jsonify({"result": result})

@app.route('/api/translate', methods=['POST'])
def api_translate():
    d = request.json
    result = llm_call([
        {"role": "system", "content": f"Translate to {d['dst']}. Output ONLY the translation."},
        {"role": "user", "content": f"Translate this from {d['src']}:\n{d['text']}"}
    ], d, max_tokens=1024, temperature=0.3)
    return jsonify({"result": result})

@app.route('/api/review', methods=['POST'])
def api_review():
    d = request.json
    result = llm_call([
        {"role": "system", "content": "You are a senior developer. Review code, find bugs and suggest improvements in Vietnamese."},
        {"role": "user", "content": f"Review this code:\n\n```\n{d['code']}\n```"}
    ], d, max_tokens=2048, temperature=0.2)
    return jsonify({"result": result})

@app.route('/api/summary', methods=['POST'])
def api_summary():
    d = request.json
    result = llm_call([
        {"role": "system", "content": "Bạn là chuyên gia tóm tắt. Tóm tắt súc tích bằng tiếng Việt."},
        {"role": "user", "content": f"Tóm tắt văn bản sau:\n{d['text'][:8000]}"}
    ], d, max_tokens=1024, temperature=0.3)
    return jsonify({"result": result})

@app.route('/api/mockdata', methods=['POST'])
def api_mockdata():
    d = request.json
    result = llm_call([
        {"role": "system", "content": "Output ONLY a valid JSON array, no markdown backticks."},
        {"role": "user", "content": f"Generate JSON for: {d['schema']}"}
    ], d, max_tokens=2048, temperature=0.8)
    return jsonify({"result": result})

@app.route('/api/terminal', methods=['POST'])
def api_terminal():
    d = request.json
    result = llm_call([
        {"role": "system", "content": "Bạn là chuyên gia DevOps. Giải thích lỗi hoặc đưa ra lệnh terminal ngắn gọn, chính xác bằng tiếng Việt."},
        {"role": "user", "content": d['input']}
    ], d, max_tokens=1024, temperature=0.2)
    return jsonify({"result": result})

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}\n  AI Multi-Provider Apps\n{'='*50}")
    print(f"  Local   : http://localhost:5000\n  LAN     : http://{local_ip}:5000\n{'='*50}\n")
    app.run(host='0.0.0.0', port=5000, debug=True)