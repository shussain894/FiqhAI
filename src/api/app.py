"""
app.py

Simple review frontend for FiqhAI eval results and live queries.

Run with:
    python -m src.api.app
    then open http://localhost:8000
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

RESULTS_FILE = Path("data/eval/eval_results.jsonl")

# Pipeline loaded once at startup
_pipeline: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading pipeline...")
    from src.retrieval.retrieve import load_retriever
    from src.retrieval.rerank import load_reranker
    model, collection = load_retriever()
    reranker = load_reranker()
    _pipeline["model"] = model
    _pipeline["collection"] = collection
    _pipeline["reranker"] = reranker
    print(f"Ready. {collection.count()} chunks indexed.")
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/results")
def get_results():
    if not RESULTS_FILE.exists():
        return JSONResponse({"error": "No eval results found. Run src/evaluation/run_eval.py first."}, status_code=404)
    results = []
    for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
def live_query(req: QueryRequest):
    from src.generation.generate import run_rag_query
    t0 = time.time()
    result = run_rag_query(
        req.question,
        _pipeline["model"],
        _pipeline["collection"],
        reranker=_pipeline["reranker"],
    )
    sa = result.get("structured_answer")
    sources = result.get("sources", [])
    return {
        "question":        req.question,
        "rewritten_query": result.get("rewritten_query"),
        "topic_detected":  result.get("topic_detected"),
        "high_risk":       result.get("high_risk", False),
        "latency_s":       round(time.time() - t0, 2),
        "short_answer":    sa.short_answer if sa else None,
        "ruling":          sa.ruling if sa else None,
        "conditions":      sa.conditions if sa else None,
        "explanation":     sa.explanation if sa else None,
        "citations":       sa.citations if sa else [],
        "confidence":      sa.confidence if sa else None,
        "note":            sa.note if sa else None,
        "raw_answer":      result.get("answer"),
        "sources": [
            {
                "source_title": s.get("source_title"),
                "page":         s.get("page"),
                "topic":        s.get("topic"),
                "score":        s.get("score"),
                "rerank_score": s.get("rerank_score"),
            }
            for s in sources
        ],
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FiqhAI Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #222; }
  header { background: #1a1a2e; color: #fff; padding: 16px 24px;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.25rem; font-weight: 600; }
  header span { font-size: 0.8rem; color: #aaa; }
  .container { max-width: 1100px; margin: 0 auto; padding: 20px; }

  /* Stats bar */
  .stats { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat-card { background: #fff; border-radius: 8px; padding: 12px 18px;
               flex: 1; min-width: 130px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .stat-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase;
                      letter-spacing: .05em; }
  .stat-card .value { font-size: 1.5rem; font-weight: 700; margin-top: 2px; }
  .stat-card .value.good { color: #16a34a; }
  .stat-card .value.warn { color: #d97706; }
  .stat-card .value.bad  { color: #dc2626; }

  /* Live query */
  .query-box { background: #fff; border-radius: 8px; padding: 16px;
               margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .query-box h2 { font-size: 0.9rem; font-weight: 600; margin-bottom: 10px; color: #444; }
  .query-row { display: flex; gap: 8px; }
  .query-row input { flex: 1; padding: 8px 12px; border: 1px solid #ddd;
                     border-radius: 6px; font-size: 0.9rem; }
  .query-row button { padding: 8px 18px; background: #1a1a2e; color: #fff;
                      border: none; border-radius: 6px; cursor: pointer;
                      font-size: 0.9rem; }
  .query-row button:hover { background: #2d2d5e; }
  #live-result { margin-top: 12px; display: none; }

  /* Filters */
  .filters { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap;
             align-items: center; }
  .filters label { font-size: 0.82rem; color: #555; }
  .filters select, .filters input[type=text] {
    padding: 5px 10px; border: 1px solid #ddd; border-radius: 5px;
    font-size: 0.82rem; background: #fff; }
  #result-count { font-size: 0.82rem; color: #888; margin-left: auto; }

  /* Table */
  .table-wrap { background: #fff; border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,.1); overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { background: #f0f0f0; padding: 10px 12px; text-align: left;
       font-weight: 600; color: #444; border-bottom: 1px solid #e0e0e0; }
  td { padding: 10px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr.expandable:hover { background: #fafafa; cursor: pointer; }
  tr.detail-row td { background: #f9f9f9; padding: 0; }
  tr.detail-row { display: none; }
  tr.detail-row.open { display: table-row; }

  .detail-inner { padding: 16px; border-top: 2px solid #1a1a2e; }
  .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .detail-section h4 { font-size: 0.75rem; text-transform: uppercase;
                       letter-spacing: .06em; color: #888; margin-bottom: 6px; }
  .detail-section p, .detail-section ul { font-size: 0.85rem; color: #333;
                                          line-height: 1.55; }
  .detail-section ul { padding-left: 16px; }
  .detail-section ul li { margin-bottom: 3px; }
  .sources-list { margin-top: 12px; }
  .source-chip { display: inline-block; background: #eef; border: 1px solid #ccf;
                 border-radius: 4px; padding: 2px 7px; font-size: 0.78rem;
                 margin: 2px; color: #336; }
  .meta-row { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
  .meta-item { font-size: 0.78rem; color: #666; }
  .meta-item strong { color: #333; }

  /* Badges */
  .badge { display: inline-block; border-radius: 4px; padding: 2px 7px;
           font-size: 0.75rem; font-weight: 600; }
  .badge-hit   { background: #dcfce7; color: #166534; }
  .badge-miss  { background: #fee2e2; color: #991b1b; }
  .badge-block { background: #fef9c3; color: #854d0e; }
  .badge-high  { background: #dcfce7; color: #166534; }
  .badge-med   { background: #fef9c3; color: #854d0e; }
  .badge-low   { background: #fee2e2; color: #991b1b; }
  .topic-tag { display: inline-block; background: #e0e7ff; color: #3730a3;
               border-radius: 4px; padding: 2px 7px; font-size: 0.75rem; }

  .expand-icon { float: right; color: #aaa; font-size: 0.9rem; }
  .loading { color: #888; font-size: 0.85rem; padding: 8px 0; }
</style>
</head>
<body>
<header>
  <h1>FiqhAI — Eval Review</h1>
  <span>Phase 2 evaluation dashboard</span>
</header>

<div class="container">

  <!-- Live query -->
  <div class="query-box">
    <h2>Live Query</h2>
    <div class="query-row">
      <input id="q-input" type="text" placeholder="Ask a Hanafi fiqh question..." />
      <button onclick="runQuery()">Ask</button>
    </div>
    <div id="live-result"></div>
  </div>

  <!-- Stats -->
  <div id="stats" class="stats"></div>

  <!-- Filters -->
  <div class="filters">
    <label>Topic</label>
    <select id="f-topic" onchange="applyFilters()">
      <option value="">All</option>
      <option>Taharah</option><option>Salah</option><option>Sawm</option>
      <option>Zakah</option><option>Usul</option>
    </select>
    <label>Source hit</label>
    <select id="f-hit" onchange="applyFilters()">
      <option value="">All</option>
      <option value="true">Hit</option>
      <option value="false">Miss</option>
    </select>
    <label>Confidence</label>
    <select id="f-conf" onchange="applyFilters()">
      <option value="">All</option>
      <option>High</option><option>Medium</option><option>Low</option>
    </select>
    <label>Search</label>
    <input type="text" id="f-search" placeholder="filter questions..." oninput="applyFilters()" style="width:200px"/>
    <span id="result-count"></span>
  </div>

  <!-- Table -->
  <div class="table-wrap">
    <table id="results-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Topic</th>
          <th>Question</th>
          <th>Source</th>
          <th>Hit</th>
          <th>Conf</th>
          <th>Latency</th>
        </tr>
      </thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>

</div>

<script>
let allResults = [];

async function loadResults() {
  const r = await fetch('/api/results');
  if (!r.ok) { document.getElementById('table-body').innerHTML = '<tr><td colspan=7>No results yet — run the eval first.</td></tr>'; return; }
  allResults = await r.json();
  renderStats(allResults);
  applyFilters();
}

function renderStats(data) {
  const answered = data.filter(r => !r.high_risk_blocked);
  const n = answered.length;
  const hits = answered.filter(r => r.source_hit).length;
  const hitPct = n ? Math.round(100*hits/n) : 0;
  const parsed = answered.filter(r => r.parse_success).length;
  const blocked = data.filter(r => r.high_risk_blocked).length;
  const avgLat = data.reduce((s,r) => s + (r.latency_s||0), 0) / data.length;
  const confCounts = {High:0,Medium:0,Low:0};
  answered.forEach(r => { if(r.confidence) confCounts[r.confidence]++; });

  document.getElementById('stats').innerHTML = `
    <div class="stat-card"><div class="label">Questions</div><div class="value">${data.length}</div></div>
    <div class="stat-card"><div class="label">Source Hit Rate</div><div class="value ${hitPct>=75?'good':hitPct>=60?'warn':'bad'}">${hitPct}%</div></div>
    <div class="stat-card"><div class="label">Parse Success</div><div class="value good">${n?Math.round(100*parsed/n):0}%</div></div>
    <div class="stat-card"><div class="label">High Confidence</div><div class="value ${confCounts.High/n>=0.3?'good':'warn'}">${n?Math.round(100*confCounts.High/n):0}%</div></div>
    <div class="stat-card"><div class="label">Blocked</div><div class="value ${blocked>0?'warn':'good'}">${blocked}</div></div>
    <div class="stat-card"><div class="label">Avg Latency</div><div class="value">${avgLat.toFixed(1)}s</div></div>
  `;
}

function applyFilters() {
  const topic = document.getElementById('f-topic').value;
  const hit   = document.getElementById('f-hit').value;
  const conf  = document.getElementById('f-conf').value;
  const search= document.getElementById('f-search').value.toLowerCase();

  const filtered = allResults.filter(r => {
    if (topic && r.topic !== topic) return false;
    if (hit === 'true'  && !r.source_hit) return false;
    if (hit === 'false' &&  r.source_hit) return false;
    if (conf && r.confidence !== conf) return false;
    if (search && !r.question.toLowerCase().includes(search)) return false;
    return true;
  });

  document.getElementById('result-count').textContent = `${filtered.length} of ${allResults.length}`;
  renderTable(filtered);
}

function renderTable(data) {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';

  data.forEach((r, i) => {
    const hitBadge = r.high_risk_blocked
      ? '<span class="badge badge-block">Blocked</span>'
      : r.source_hit
        ? '<span class="badge badge-hit">Hit</span>'
        : '<span class="badge badge-miss">Miss</span>';

    const confBadge = r.confidence
      ? `<span class="badge badge-${r.confidence==='High'?'high':r.confidence==='Medium'?'med':'low'}">${r.confidence}</span>`
      : '—';

    const row = document.createElement('tr');
    row.className = 'expandable';
    row.dataset.idx = i;
    row.innerHTML = `
      <td style="font-size:.78rem;color:#888">${r.id}</td>
      <td><span class="topic-tag">${r.topic}</span></td>
      <td>${escHtml(r.question)}<span class="expand-icon">▸</span></td>
      <td style="font-size:.78rem;color:#666">${r.source_hint}</td>
      <td>${hitBadge}</td>
      <td>${confBadge}</td>
      <td style="font-size:.78rem;color:#888">${r.latency_s ? r.latency_s+'s' : '—'}</td>
    `;

    const detail = document.createElement('tr');
    detail.className = 'detail-row';
    detail.dataset.idx = i;
    detail.innerHTML = `<td colspan="7"><div class="detail-inner">${renderDetail(r)}</div></td>`;

    row.addEventListener('click', () => {
      const isOpen = detail.classList.contains('open');
      document.querySelectorAll('.detail-row.open').forEach(el => {
        el.classList.remove('open');
        el.previousSibling.querySelector('.expand-icon').textContent = '▸';
      });
      if (!isOpen) {
        detail.classList.add('open');
        row.querySelector('.expand-icon').textContent = '▾';
      }
    });

    tbody.appendChild(row);
    tbody.appendChild(detail);
  });
}

function renderDetail(r) {
  if (r.high_risk_blocked) {
    return `<p style="color:#854d0e">This query was blocked by the high-risk safety filter.</p>`;
  }

  const citations = (r.citations||[]).map(c => `<li>${escHtml(c)}</li>`).join('');
  const sources = (r.sources||[]).map(s =>
    `<span class="source-chip">${escHtml(s.source_title)} p.${s.page}${s.rerank_score!=null?' ('+s.rerank_score.toFixed(2)+')':''}</span>`
  ).join('');

  return `
    <div class="detail-grid">
      <div>
        <div class="detail-section">
          <h4>Short Answer</h4>
          <p>${escHtml(r.short_answer||'—')}</p>
        </div>
        <div class="detail-section" style="margin-top:12px">
          <h4>Hanafi Ruling</h4>
          <p>${escHtml(r.ruling||'—')}</p>
        </div>
        <div class="detail-section" style="margin-top:12px">
          <h4>Conditions / Exceptions</h4>
          <p>${escHtml(r.conditions||'—')}</p>
        </div>
      </div>
      <div>
        <div class="detail-section">
          <h4>Source-Based Explanation</h4>
          <p>${escHtml(r.explanation||'—')}</p>
        </div>
        <div class="detail-section" style="margin-top:12px">
          <h4>Citations</h4>
          <ul>${citations||'<li>—</li>'}</ul>
        </div>
      </div>
    </div>
    <div class="sources-list">
      <div class="detail-section"><h4>Sources Retrieved</h4></div>
      <div style="margin-top:4px">${sources||'—'}</div>
    </div>
    <div class="meta-row">
      <div class="meta-item"><strong>Rewritten query:</strong> ${escHtml(r.rewritten_query||'—')}</div>
      <div class="meta-item"><strong>Topic detected:</strong> ${r.topic_detected||'none'}</div>
      <div class="meta-item"><strong>Source hint:</strong> ${r.source_hint}</div>
    </div>
  `;
}

function renderLiveResult(r) {
  if (r.high_risk) {
    return `<div style="background:#fef9c3;padding:12px;border-radius:6px;font-size:.85rem;color:#854d0e">
      <strong>High-risk topic</strong> — please consult a qualified Hanafi scholar.
    </div>`;
  }
  const citations = (r.citations||[]).map(c => `<li>${escHtml(c)}</li>`).join('');
  const sources = (r.sources||[]).map(s =>
    `<span class="source-chip">${escHtml(s.source_title)} p.${s.page}${s.rerank_score!=null?' ('+s.rerank_score.toFixed(2)+')':''}</span>`
  ).join('');
  return `
    <div style="border-top:1px solid #eee;padding-top:12px">
      <div style="display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap">
        <span class="meta-item"><strong>Rewritten:</strong> ${escHtml(r.rewritten_query||'—')}</span>
        <span class="meta-item"><strong>Topic:</strong> ${r.topic_detected||'none'}</span>
        <span class="meta-item"><strong>Confidence:</strong> ${r.confidence||'—'}</span>
        <span class="meta-item"><strong>Latency:</strong> ${r.latency_s}s</span>
      </div>
      <div class="detail-grid">
        <div>
          <div class="detail-section"><h4>Short Answer</h4><p>${escHtml(r.short_answer||'—')}</p></div>
          <div class="detail-section" style="margin-top:10px"><h4>Hanafi Ruling</h4><p>${escHtml(r.ruling||'—')}</p></div>
          <div class="detail-section" style="margin-top:10px"><h4>Conditions / Exceptions</h4><p>${escHtml(r.conditions||'—')}</p></div>
        </div>
        <div>
          <div class="detail-section"><h4>Explanation</h4><p>${escHtml(r.explanation||'—')}</p></div>
          <div class="detail-section" style="margin-top:10px"><h4>Citations</h4><ul>${citations||'<li>—</li>'}</ul></div>
        </div>
      </div>
      <div class="sources-list" style="margin-top:10px">
        <div class="detail-section"><h4>Sources Retrieved</h4></div>
        <div style="margin-top:4px">${sources}</div>
      </div>
    </div>
  `;
}

async function runQuery() {
  const q = document.getElementById('q-input').value.trim();
  if (!q) return;
  const el = document.getElementById('live-result');
  el.style.display = 'block';
  el.innerHTML = '<div class="loading">Searching and generating answer...</div>';
  try {
    const r = await fetch('/api/query', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await r.json();
    el.innerHTML = renderLiveResult(data);
  } catch(e) {
    el.innerHTML = '<div class="loading">Error — is the server running?</div>';
  }
}

document.getElementById('q-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') runQuery();
});

function escHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadResults();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
