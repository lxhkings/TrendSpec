"""轻量实时面板：只读 state.json / ledger.jsonl，与研究进程解耦。"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from trendspec.research.ledger import read_ledger, read_state

_INDEX_HTML = """<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>因子研究监控</title>
<style>
 body{font-family:system-ui,monospace;margin:2rem;background:#0f1115;color:#e6e6e6}
 h1{font-size:1.2rem} table{border-collapse:collapse;width:100%;margin-top:1rem}
 td,th{border:1px solid #333;padding:.4rem .6rem;text-align:left}
 .bar{height:8px;background:#2a2a2a;border-radius:4px;overflow:hidden}
 .bar>i{display:block;height:100%;background:#4ade80}
</style></head>
<body>
<h1>因子研究监控</h1>
<div id="state"></div>
<h2>研究记录</h2>
<table id="ledger"><thead><tr><th>轮</th><th>逻辑</th><th>最佳OOS Sharpe</th><th>赢家</th></tr></thead>
<tbody></tbody></table>
<script>
async function tick(){
 const s=await (await fetch('/api/state')).json();
 const pct=s.sweep_total?Math.round(100*s.sweep_done/s.sweep_total):0;
 document.getElementById('state').innerHTML=
  `<p>状态: <b>${s.phase||'-'}</b> | 轮 ${s.round||'-'}/${s.max_rounds||'-'} | 赢家 ${s.winners||0}</p>`+
  (s.sweep_total?`<div class="bar"><i style="width:${pct}%"></i></div><small>扫参 ${s.sweep_done}/${s.sweep_total}</small>`:'');
 const rows=await (await fetch('/api/ledger')).json();
 const tb=document.querySelector('#ledger tbody'); tb.innerHTML='';
 rows.slice().reverse().forEach(r=>{
  const best=(r.top_candidates&&r.top_candidates[0])||{};
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${r.round||''}</td><td>${(r.hypothesis&&r.hypothesis.rationale)||r.error||''}</td>`+
   `<td>${best.oos_sharpe!=null?Number(best.oos_sharpe).toFixed(2):''}</td><td>${r.winners||0}</td>`;
  tb.appendChild(tr);
 });
}
tick(); setInterval(tick,2000);
</script>
</body></html>"""


def create_app(out_dir: str) -> FastAPI:
    out = Path(out_dir)
    app = FastAPI(title="TrendSpec 因子研究监控")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(read_state(out / "state.json"))

    @app.get("/api/ledger")
    def ledger() -> JSONResponse:
        return JSONResponse(read_ledger(out / "ledger.jsonl")[-50:])

    return app
