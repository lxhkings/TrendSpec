"""轻量实时面板：只读 state.json / ledger.jsonl，与研究进程解耦。"""

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

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
 .panel{background:#1a1d24;border:1px solid #2a2d34;border-radius:8px;padding:1rem 1.4rem;margin-bottom:1.4rem}
 label{display:inline-block;width:110px;color:#aaa;font-size:.85rem}
 input,select{background:#0f1115;border:1px solid #333;color:#e6e6e6;padding:.3rem .5rem;border-radius:4px;width:140px;font-size:.85rem}
 .row{margin:.4rem 0}
 button{margin-top:.8rem;padding:.45rem 1.2rem;background:#4ade80;color:#000;border:none;border-radius:4px;cursor:pointer;font-weight:bold}
 button:disabled{background:#2a2a2a;color:#666;cursor:not-allowed}
 #run-msg{margin-left:1rem;font-size:.85rem}
</style></head>
<body>
<h1>因子研究面板</h1>

<div class="panel">
 <div class="row"><label>市场</label>
  <select id="p-market"><option value="us">US</option><option value="cn">CN</option></select></div>
 <div class="row"><label>开始日期</label><input id="p-start" value="2015-01-01"></div>
 <div class="row"><label>结束日期</label><input id="p-end" value="2023-12-31"></div>
 <div class="row"><label>轮数</label><input id="p-rounds" type="number" value="10" min="1"></div>
 <div class="row"><label>每轮候选上限</label><input id="p-candidates" type="number" value="200" min="10"></div>
 <div class="row"><label>WF 窗口数</label><input id="p-windows" type="number" value="4" min="2"></div>
 <div class="row"><label>初始资金</label><input id="p-capital" type="number" value="100000"></div>
 <button id="btn-run" onclick="startRun()">开始研究</button>
 <span id="run-msg"></span>
</div>

<div id="state"></div>
<h2>研究记录</h2>
<table id="ledger"><thead><tr><th>轮</th><th>逻辑</th><th>最佳OOS Sharpe</th><th>赢家</th></tr></thead>
<tbody></tbody></table>

<script>
async function startRun(){
 const btn=document.getElementById('btn-run');
 const msg=document.getElementById('run-msg');
 btn.disabled=true; msg.textContent='提交中…';
 const body={
  market:document.getElementById('p-market').value,
  start:document.getElementById('p-start').value,
  end:document.getElementById('p-end').value,
  rounds:parseInt(document.getElementById('p-rounds').value),
  max_candidates:parseInt(document.getElementById('p-candidates').value),
  n_windows:parseInt(document.getElementById('p-windows').value),
  capital:parseFloat(document.getElementById('p-capital').value),
 };
 const res=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const data=await res.json();
 msg.textContent=data.detail||data.status||'';
 if(res.status!==200) btn.disabled=false;
}

async function tick(){
 const r=await fetch('/api/running');
 const running=(await r.json()).running;
 const btn=document.getElementById('btn-run');
 const msg=document.getElementById('run-msg');
 btn.disabled=running;
 if(running) msg.textContent='研究进行中…';
 else if(msg.textContent==='研究进行中…') msg.textContent='已完成';

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


class RunRequest(BaseModel):
    market: str = "us"
    start: str = "2015-01-01"
    end: str = "2023-12-31"
    rounds: int = 10
    max_candidates: int = 200
    n_windows: int = 4
    capital: float = 100000.0


def create_app(out_dir: str, settings=None) -> FastAPI:
    out = Path(out_dir)
    app = FastAPI(title="TrendSpec 因子研究监控")

    _state = {"running": False, "thread": None}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(read_state(out / "state.json"))

    @app.get("/api/ledger")
    def ledger() -> JSONResponse:
        return JSONResponse(read_ledger(out / "ledger.jsonl")[-50:])

    @app.get("/api/running")
    def running() -> JSONResponse:
        return JSONResponse({"running": _state["running"]})

    @app.post("/api/run")
    def run(req: RunRequest) -> JSONResponse:
        if _state["running"]:
            return JSONResponse({"detail": "研究正在运行中，请等待完成"}, status_code=409)

        from datetime import date

        import trendspec.factors  # noqa: F401
        import trendspec.strategy.factor_strategy  # noqa: F401
        from trendspec.research.agent import HypothesisAgent
        from trendspec.research.config import ResearchSettings
        from trendspec.research.llm_client import OpenAICompatClient
        from trendspec.research.orchestrator import ResearchOrchestrator
        from trendspec.research.fast_eval import ResearchEvaluator

        cfg = settings or ResearchSettings()
        client = OpenAICompatClient(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
        )
        agent = HypothesisAgent(client)
        start_date = date.fromisoformat(req.start)
        end_date = date.fromisoformat(req.end)
        evaluator = ResearchEvaluator(
            market=req.market, start=start_date, end=end_date,
            n_windows=req.n_windows, capital=req.capital, parallel=True)
        orch = ResearchOrchestrator(
            agent=agent,
            evaluate_fn=None,
            out_dir=str(out),
            max_rounds=req.rounds,
            max_candidates=req.max_candidates,
            batch_evaluator=evaluator,
        )

        def _run():
            _state["running"] = True
            try:
                orch.run()
            finally:
                _state["running"] = False

        t = threading.Thread(target=_run, daemon=True)
        _state["thread"] = t
        t.start()
        return JSONResponse({"status": "研究已启动"})

    return app
