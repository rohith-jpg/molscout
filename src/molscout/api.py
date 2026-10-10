"""Small local API and browser UI for the implemented offline analysis flow."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from molscout.analysis import analyze_batch

app = FastAPI(
    title="MolScout",
    description="Evidence-linked exploratory cheminformatics; not a clinical tool.",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=100)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "offline-heuristic"}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, object]:
    try:
        return analyze_batch(request.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Return a dependency-free, responsive form for the local demo."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MolScout — molecular triage</title>
  <style>
    :root{color-scheme:dark;--bg:#0b1220;--panel:#111c2d;--line:#24354b;--ink:#e5edf8;--muted:#9db0c8;--accent:#60d6b2}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 85% 0%,#17304a 0,transparent 42%),var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
    main{max-width:980px;margin:7vh auto;padding:24px}.eyebrow{color:var(--accent);font-size:.8rem;letter-spacing:.16em;text-transform:uppercase}
    h1{font-size:clamp(2rem,6vw,4rem);line-height:1.05;margin:.35rem 0}.sub{color:var(--muted);max-width:680px}
    section{background:#111c2de8;border:1px solid var(--line);border-radius:18px;padding:22px;margin-top:24px;box-shadow:0 18px 70px #0003}
    label{display:block;font-weight:650;margin-bottom:9px}textarea{width:100%;min-height:150px;background:#08111d;color:var(--ink);border:1px solid #324a64;border-radius:10px;padding:14px;font:14px/1.6 ui-monospace,monospace;resize:vertical}
    button{margin-top:13px;background:var(--accent);color:#072018;border:0;border-radius:9px;padding:11px 17px;font-weight:750;cursor:pointer}button:disabled{opacity:.55;cursor:wait}
    .note{color:var(--muted);font-size:.88rem}pre{white-space:pre-wrap;overflow:auto;max-height:60vh;background:#080f19;border-radius:10px;padding:16px;color:#cfe6df}
    .error{color:#ffb4ab}
  </style>
</head>
<body><main>
  <div class="eyebrow">Auditable cheminformatics demo</div>
  <h1>MolScout</h1>
  <p class="sub">Inspect SMILES, compare simple molecular descriptors, and trace each report statement to the deterministic tool result behind it.</p>
  <section>
    <form id="form"><label for="smiles">One molecule per line</label>
      <textarea id="smiles" spellcheck="false" aria-label="SMILES list">CCO
CC(=O)O
not-a-smiles</textarea>
      <button id="submit" type="submit">Analyze molecules</button>
      <p class="note">Ranking counts Lipinski threshold violations. It is not an ADMET prediction, efficacy estimate, or safety assessment.</p>
    </form>
  </section>
  <section><strong id="status">Ready for an offline analysis.</strong><pre id="result" aria-live="polite">Run the analysis to see ranked molecules and evidence IDs.</pre></section>
</main>
<script>
const form=document.querySelector('#form'),button=document.querySelector('#submit'),status=document.querySelector('#status'),result=document.querySelector('#result');
form.addEventListener('submit',async event=>{event.preventDefault();const smiles=document.querySelector('#smiles').value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);button.disabled=true;status.textContent='Analyzing locally…';status.className='';
 try{const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles})});const payload=await response.json();if(!response.ok)throw new Error(payload.detail||'Analysis failed');result.textContent=JSON.stringify(payload,null,2);status.textContent=`Analyzed ${payload.molecules.length} inputs; see the evidence IDs in each report.`;}
 catch(error){status.textContent=error.message;status.className='error';}finally{button.disabled=false;}});
</script></body></html>"""
