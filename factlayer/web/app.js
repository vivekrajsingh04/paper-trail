'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const api = async (path) => {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
};

const fmtNum = (v) => {
  if (v === null || v === undefined) return '—';
  if (!isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (a >= 1e6) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 1) return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  return v.toPrecision(4);
};

// ---------------------------------------------------------------- tabs
$$('nav button').forEach((b) => b.addEventListener('click', () => {
  $$('nav button').forEach((x) => x.classList.toggle('on', x === b));
  $$('main section').forEach((s) => s.classList.add('hidden'));
  $(`#tab-${b.dataset.tab}`).classList.remove('hidden');
  const load = { relations: loadRelations, facts: loadFacts, diagnostics: loadDiagnostics }[b.dataset.tab];
  if (load) load();
}));

// ---------------------------------------------------------------- header
async function loadSummary() {
  const s = await api('/api/summary');
  const rc = s.relations || {};
  const bv = rc.by_verdict || {};
  $('#stats').innerHTML = [
    ['Documents', s.documents], ['Pages', s.pages], ['Facts', s.facts],
    ['Corroborations', bv.corroborates || 0], ['Reconciled', bv.reconciled || 0],
    ['Contradictions', bv.contradicts || 0],
  ].map(([k, v]) => `<div class="stat"><b>${Number(v).toLocaleString()}</b><i>${k}</i></div>`).join('');
}

// ---------------------------------------------------------------- range chart
// The chart that carries the argument: each fact is a point estimate with the
// interval its digits imply, drawn on one axis so "do these overlap?" is
// something you see rather than something the prose asserts.

function niceTicks(lo, hi, count = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / (count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const first = Math.ceil(lo / step) * step;
  const out = [];
  for (let v = first; v <= hi + step * 1e-9; v += step) out.push(v);
  return out.length >= 2 ? out : [lo, hi];
}

function tickLabel(v, span) {
  const dp = span > 0 ? Math.max(0, Math.min(6, Math.ceil(-Math.log10(span / 4)) + 1)) : 0;
  return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

function rangeChart(v) {
  if (!v || v.mode !== 'interval') return '';
  const [a0, a1] = v.left_interval, [b0, b1] = v.right_interval;
  const av = v.left_value, bv = v.right_value;
  if (![a0, a1, b0, b1, av, bv].every(Number.isFinite)) return '';

  const W = 760, PADL = 10, PADR = 10;
  const ROW_A = 28, ROW_B = 60, BAR = 14, AXIS = 92, H = 116;
  const x0 = PADL, x1 = W - PADR;

  const lo = Math.min(a0, b0), hi = Math.max(a1, b1);
  const span = (hi - lo) || Math.max(Math.abs(hi), 1) * 1e-6;
  const pad = span * 0.14;
  const dLo = lo - pad, dHi = hi + pad, dSpan = dHi - dLo;
  const X = (val) => x0 + ((val - dLo) / dSpan) * (x1 - x0);

  // A range narrower than a couple of pixels still has to be visible.
  const bar = (v0, v1, y, cls) => {
    const px0 = X(v0);
    const w = Math.max(X(v1) - px0, 3);
    return `<rect class="${cls}" x="${px0.toFixed(1)}" y="${y}" width="${w.toFixed(1)}"
            height="${BAR}" rx="4" ry="4"/>`;
  };

  const iLo = Math.max(a0, b0), iHi = Math.min(a1, b1);
  const overlaps = iHi > iLo;
  const bandLo = overlaps ? iLo : iHi, bandHi = overlaps ? iHi : iLo;
  const bandW = Math.max(X(bandHi) - X(bandLo), 2);
  const band = `<rect class="${overlaps ? 'rc-band-ok' : 'rc-band-gap'}"
      x="${X(bandLo).toFixed(1)}" y="14" width="${bandW.toFixed(1)}" height="${AXIS - 14}" rx="2"/>`;

  const ticks = niceTicks(dLo, dHi, 5).map((t) => {
    const x = X(t);
    if (x < x0 - 1 || x > x1 + 1) return '';
    return `<line class="rc-grid" x1="${x.toFixed(1)}" y1="${AXIS}" x2="${x.toFixed(1)}" y2="${AXIS + 5}"/>
            <text class="rc-tick" x="${x.toFixed(1)}" y="${AXIS + 17}" text-anchor="middle">${tickLabel(t, dSpan)}</text>`;
  }).join('');

  return `<div class="rangechart">
    <svg viewBox="0 0 ${W} ${H}" role="img"
         aria-label="Implied precision ranges for both figures on a shared axis">
      ${band}
      ${bar(a0, a1, ROW_A, 'rc-bar-a')}
      ${bar(b0, b1, ROW_B, 'rc-bar-b')}
      <circle class="rc-dot rc-dot-a" cx="${X(av).toFixed(1)}" cy="${ROW_A + BAR / 2}" r="5"/>
      <circle class="rc-dot rc-dot-b" cx="${X(bv).toFixed(1)}" cy="${ROW_B + BAR / 2}" r="5"/>
      <text class="rc-label" x="${x0}" y="${ROW_A - 6}">A</text>
      <text class="rc-label" x="${x0}" y="${ROW_B - 6}">B</text>
      <line class="rc-grid" x1="${x0}" y1="${AXIS}" x2="${x1}" y2="${AXIS}"/>
      ${ticks}
    </svg>
    <div class="rc-legend">
      <span><i class="sw a"></i>A stated ${fmtNum(av)} <span class="rng">· implied ${fmtNum(a0)}–${fmtNum(a1)}</span></span>
      <span><i class="sw b"></i>B stated ${fmtNum(bv)} <span class="rng">· implied ${fmtNum(b0)}–${fmtNum(b1)}</span></span>
    </div>
    <div class="rc-caption">
      <span class="verdict ${overlaps ? 'ok' : 'no'}">${overlaps ? 'Ranges intersect' : 'Ranges are disjoint'}</span>
      <span class="muted">${overlaps
        ? '— the difference is within what the rounding of these figures allows.'
        : '— the difference is larger than rounding can account for.'}</span>
    </div>
  </div>`;
}

// ---------------------------------------------------------------- fact box
function dimChips(dims, conflict, agree) {
  return Object.entries(dims || {}).map(([k, v]) => {
    const cls = conflict.includes(k) ? 'conflict' : (agree.includes(k) ? 'agree' : '');
    return `<span class="dim ${cls}">${esc(k)}=${esc(v)}</span>`;
  }).join('');
}

function factBox(side, brief, dims, conflict, agree) {
  const page = brief.page_label ? `p.${esc(brief.page_label)}` : `page ${brief.page + 1}`;
  return `<div class="factbox side-${side.toLowerCase()}">
    <div class="src"><b>${side}</b> · ${esc(brief.doc_title || brief.doc_id)} · ${page}</div>
    <div class="val">${esc(brief.raw ?? fmtNum(brief.value))}</div>
    <div class="met">${esc(brief.metric)}${brief.period ? ` · <b>${esc(brief.period)}</b>` : ''}</div>
    <div class="quote">${esc(brief.quote || '')}</div>
    <div class="dims">${dimChips(dims, conflict, agree)}</div>
    <button class="btn tiny" style="margin-top:11px"
      onclick="showEvidence('${brief.id}')">Show on page →</button>
  </div>`;
}

function relationCard(rel, label) {
  const r = rel.reasoning || {};
  const dims = r.dimensions || {};
  const conflict = dims.conflict || [], agree = dims.agree || [];
  const left = r.left || {}, right = r.right || {};
  const v = rel.verdict;
  const mm = r.metric_match || {};

  return `<div class="card">
    <div class="card-head">
      ${label ? `<span class="case-label">${esc(label)}</span>` : ''}
      <span class="badge ${v}">${v}</span>
      ${rel.cross_document ? '<span class="badge plain">cross-document</span>' : '<span class="badge plain">same document</span>'}
      ${rel.explained_by ? `<span class="badge plain">explained by ${esc(rel.explained_by)}</span>` : ''}
      <span class="sub" style="margin-left:auto">confidence ${rel.confidence}</span>
    </div>
    <div class="card-body">
      <div class="pair">
        ${factBox('A', left, dims.left, conflict, agree)}
        ${factBox('B', right, dims.right, conflict, agree)}
      </div>
      <div class="reasoning">
        <h4>How the system reached this</h4>
        <div class="prose">${esc(rel.explanation)}</div>
        ${rangeChart(r.values)}
        <div class="provenance">
          Metric match: <code>${esc(mm.source || '?')}</code> — ${esc(mm.reason || '')}
        </div>
      </div>
    </div></div>`;
}

// ---------------------------------------------------------------- cases
const CASE_META = [
  ['corroboration', 'Case 1 — corroborated across documents, expressed differently'],
  ['contradiction', 'Case 2 — a genuine or likely contradiction'],
  ['reconciled', 'Case 3 — apparent contradiction explained by context'],
];

async function loadCases() {
  const c = await api('/api/cases');
  let html = '';
  for (const [key, label] of CASE_META) {
    html += c[key]
      ? relationCard(c[key], label)
      : `<div class="card"><div class="card-head"><span class="case-label">${esc(label)}</span></div>
         <div class="card-body muted">No example in the current knowledge layer yet.</div></div>`;
  }

  const f = c.failure;
  html += `<div class="card">
    <div class="card-head">
      <span class="case-label">Case 4 — an extraction or reasoning failure, and how it is handled</span>
    </div>
    <div class="card-body">
      ${f ? `<b>${esc(f.title || '')}</b>
        <p style="margin:8px 0">${esc(f.description || '')}</p>
        ${f.evidence ? `<div class="quote">${esc(f.evidence)}</div>` : ''}
        ${f.handling ? `<div class="reasoning" style="margin-top:12px">
          <h4>How the system handles it</h4>${esc(f.handling)}</div>` : ''}`
        : '<span class="muted">See the Diagnostics tab for live rejection and coverage figures.</span>'}
    </div></div>`;
  $('#cases').innerHTML = html;
}

// ---------------------------------------------------------------- relations
async function loadRelations() {
  const v = $('#rel-verdict').value, x = $('#rel-cross').value, cf = $('#rel-conf').value;
  let url = `/api/relations?limit=40&min_confidence=${cf}`;
  if (v) url += `&verdict=${v}`;
  if (x) url += `&cross_document=${x}`;
  $('#relations').innerHTML = '<div class="empty"><span class="spinner"></span> Loading…</div>';
  const d = await api(url);
  const c = d.counts || {}; const bv = c.by_verdict || {};
  $('#rel-count').textContent =
    `${d.count} shown · ${c.total || 0} total (${bv.corroborates || 0} corroborate, ` +
    `${bv.reconciled || 0} reconciled, ${bv.contradicts || 0} contradict)`;
  $('#relations').innerHTML = d.relations.length
    ? d.relations.map((r) => relationCard(r, null)).join('')
    : '<div class="empty">No relations match these filters.</div>';
}
['#rel-verdict', '#rel-cross', '#rel-conf'].forEach((s) =>
  document.addEventListener('DOMContentLoaded', () => $(s).addEventListener('change', loadRelations)));

// ---------------------------------------------------------------- facts
async function loadFacts() {
  const q = $('#fact-q').value.trim(), doc = $('#fact-doc').value;
  let url = '/api/facts?limit=150';
  if (q) url += `&q=${encodeURIComponent(q)}`;
  if (doc) url += `&doc_id=${encodeURIComponent(doc)}`;
  const d = await api(url);
  $('#fact-count').textContent = `${d.count} facts`;
  $('#facts-table tbody').innerHTML = d.facts.map((f) => {
    const qy = f.quantity || {};
    const entries = Object.entries(f.qualifiers || {});
    // Show a few qualifiers and count the rest: a fact can carry a dozen, and
    // rendering all of them turns every row into a paragraph.
    const shown = entries.slice(0, 3)
      .map(([k, v]) => `<span class="dim">${esc(k)}=${esc(String(v).slice(0, 22))}</span>`).join('');
    const more = entries.length > 3 ? `<span class="more">+${entries.length - 3}</span>` : '';
    const page = f.evidence.page_label ? 'p.' + esc(f.evidence.page_label)
                                       : 'pg ' + (f.evidence.page + 1);
    return `<tr class="clickable" onclick="showEvidence('${f.id}')"
              title="${esc(f.metric)}">
      <td><div class="clamp2">${esc(f.metric)}</div></td>
      <td class="num">${esc(qy.raw ?? f.state ?? '')}</td>
      <td class="small muted"><div class="clamp2">${esc(qy.unit_raw ?? qy.canonical_unit ?? '')}</div></td>
      <td class="small"><div class="clamp2">${esc(f.period?.label ?? '')}</div></td>
      <td><div class="chiprow">${shown}${more}</div></td>
      <td class="small muted"><div class="clamp2">${esc(f.evidence.doc_title || f.evidence.doc_id)} · ${page}</div></td>
      <td class="num">${f.confidence}</td></tr>`;
  }).join('');
}
document.addEventListener('DOMContentLoaded', () => {
  $('#fact-go').addEventListener('click', loadFacts);
  $('#fact-q').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadFacts(); });
  $('#fact-doc').addEventListener('change', loadFacts);
});

// ---------------------------------------------------------------- diagnostics
async function loadDiagnostics() {
  const d = await api('/api/diagnostics');
  let html = `<div class="card"><div class="card-head">
    <h3>Extraction accounting</h3>
    <span class="sub">Proposed vs kept is the hallucination guard; coverage is the recall estimate.</span>
  </div><div class="table-wrap"><table>
    <thead><tr><th>Document</th><th class="num">Pages</th><th class="num">Facts</th>
      <th class="num">Proposed</th><th class="num">Verified</th>
      <th class="num">Quantities seen</th><th class="num">Coverage</th></tr></thead><tbody>`;
  for (const doc of d.documents) {
    const vr = doc.verification_pass_rate;
    const cr = doc.coverage_ratio;
    html += `<tr><td>${esc(doc.title)}</td>
      <td class="num">${doc.pages}</td><td class="num">${doc.facts}</td>
      <td class="num">${doc.facts_proposed ?? '—'}</td>
      <td class="num">${vr != null ? (vr * 100).toFixed(1) + '%' : '—'}</td>
      <td class="num">${doc.quantities_detected ?? '—'}</td>
      <td class="num">${cr != null ? (cr * 100).toFixed(1) + '%' : '—'}</td></tr>`;
  }
  html += '</tbody></table></div></div>';

  const dims = d.dimensions || {};
  const rows = Object.entries(dims)
    .map(([k, v]) => [k, v.pairs || 0, v.value_differs || 0,
      (v.value_differs + 2) / (v.pairs + 4)])
    .sort((a, b) => b[3] - a[3]);
  html += `<div class="card"><div class="card-head">
      <h3>Learned dimension weights</h3>
      <span class="sub">How often each dimension accompanies a change in value, measured on this
        corpus — not declared in advance. This is what lets the schema grow.</span>
    </div><div class="card-body">`;
  html += rows.length ? rows.map(([k, pairs, diff, power]) => `
      <div class="meter-row">
        <div class="lab">
          <span class="k">${esc(k)}</span>
          <span class="v">${(power * 100).toFixed(0)}% of ${pairs} pair${pairs === 1 ? '' : 's'}</span>
        </div>
        <div class="meter"><div style="width:${(power * 100).toFixed(0)}%"></div></div>
      </div>`).join('')
    : '<span class="muted">No dimension statistics yet.</span>';
  html += '</div></div>';

  for (const doc of d.documents) {
    const worst = doc.lowest_coverage_pages || [];
    const rej = doc.rejections || [];
    if (!worst.length && !rej.length) continue;
    html += `<div class="card"><div class="card-head"><h3>${esc(doc.title)}</h3>
      <span class="sub">what was rejected or walked past</span></div><div class="card-body">`;
    if (rej.length) {
      html += '<h4 class="case-label">Rejected proposals</h4><div class="table-wrap"><table><thead><tr>'
        + '<th>Page</th><th>Reason</th><th>Detail</th></tr></thead><tbody>'
        + rej.map((r) => `<tr><td class="num">${r.page}</td><td class="mono small">${esc(r.why)}</td>
            <td class="small muted">${esc(r.quote || r.value || r.metric || '')}</td></tr>`).join('')
        + '</tbody></table></div>';
    }
    if (worst.length) {
      html += '<h4 class="case-label" style="margin-top:14px">Lowest-coverage pages</h4>'
        + worst.map((w) => `<div class="small" style="margin-bottom:7px">
            <b>page ${w.page + 1}</b> — ${w.covered}/${w.detected} quantities became facts
            <div class="quote">${esc((w.examples || []).join(' · '))}</div></div>`).join('');
    }
    html += '</div></div>';
  }
  $('#diagnostics').innerHTML = html;
}

// ---------------------------------------------------------------- evidence
async function showEvidence(factId) {
  $('#modal').classList.remove('hidden');
  $('#modal-body').innerHTML = '<div class="empty"><span class="spinner"></span> Rendering page…</div>';
  let e;
  try { e = await api(`/api/evidence/${factId}`); }
  catch { $('#modal-body').innerHTML = '<div class="empty">Evidence unavailable.</div>'; return; }

  $('#modal-title').textContent = e.doc_title || e.doc_id;
  $('#modal-sub').textContent =
    (e.page_label ? `printed page ${e.page_label}` : `page ${e.page + 1}`) + ` · “${e.quote}”`;

  const img = new Image();
  img.onload = () => {
    // Bounding boxes are stored in PDF points; scale them onto the rendered raster.
    const resp = img.dataset;
    const sx = img.naturalWidth / parseFloat(resp.pw);
    const sy = img.naturalHeight / parseFloat(resp.ph);
    const boxes = (e.bboxes || []).map((b) => {
      const pad = 2.5;
      return `<div class="hl" style="left:${(b.x0 * sx - pad) / img.naturalWidth * 100}%;
        top:${(b.y0 * sy - pad) / img.naturalHeight * 100}%;
        width:${((b.x1 - b.x0) * sx + pad * 2) / img.naturalWidth * 100}%;
        height:${((b.y1 - b.y0) * sy + pad * 2) / img.naturalHeight * 100}%"></div>`;
    }).join('');
    $('#modal-body').innerHTML =
      `<div class="page-canvas">${img.outerHTML}${boxes}</div>
       ${boxes ? '' : '<p class="muted small">No coordinates stored for this span.</p>'}`;
  };
  img.onerror = () => {
    $('#modal-body').innerHTML =
      `<p class="muted small">Source page could not be rendered. Extracted context:</p>
       <div class="quote">${esc(e.snippet)}</div>`;
  };

  const r = await fetch(e.image_url);
  if (!r.ok) { img.onerror(); return; }
  const blob = await r.blob();
  img.dataset.pw = r.headers.get('X-Page-Width') || '595';
  img.dataset.ph = r.headers.get('X-Page-Height') || '842';
  img.src = URL.createObjectURL(blob);
}

function closeModal() { $('#modal').classList.add('hidden'); }
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
document.addEventListener('DOMContentLoaded', () =>
  $('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); }));

// ---------------------------------------------------------------- upload
document.addEventListener('DOMContentLoaded', () => {
  $('#upload-go').addEventListener('click', async () => {
    const f = $('#file').files[0];
    if (!f) { $('#upload-status').textContent = 'Choose a PDF first.'; return; }
    const fd = new FormData();
    fd.append('file', f);
    const corpus = $('#corpus').value.trim();
    const r = await fetch('/api/documents' + (corpus ? `?corpus=${encodeURIComponent(corpus)}` : ''),
      { method: 'POST', body: fd });
    const j = await r.json();
    if (!j.job_id) { $('#upload-status').textContent = 'Upload failed: ' + JSON.stringify(j); return; }
    poll(j.job_id);
  });
});

async function poll(jobId) {
  const el = $('#upload-status');
  for (;;) {
    const j = await api(`/api/jobs/${jobId}`);
    if (j.stage === 'done') {
      const res = j.result || {};
      el.innerHTML = res.status === 'already_ingested'
        ? `<b>Already in the layer</b> — ${esc(res.title || '')}`
        : `<b>Done.</b> ${res.facts} facts, ${res.relations} new relations in ${res.elapsed_s}s.
           <br>Verification pass rate
           ${((res.extraction?.verification_pass_rate ?? 0) * 100).toFixed(1)}%.`;
      loadSummary(); loadCases(); loadDocsFilter();
      return;
    }
    if (j.stage === 'error') { el.innerHTML = `<b>Failed:</b> ${esc(j.error)}`; return; }
    el.innerHTML = `<span class="spinner"></span> ${esc(j.stage)}` +
      (j.done != null ? ` — page ${j.done}/${j.total}` : '…');
    await new Promise((r) => setTimeout(r, 1200));
  }
}

async function loadDocsFilter() {
  const docs = await api('/api/documents');
  $('#fact-doc').innerHTML = '<option value="">All documents</option>' +
    docs.map((d) => `<option value="${esc(d.doc_id)}">${esc(d.title || d.filename)}</option>`).join('');
}

// ---------------------------------------------------------------- boot
(async function boot() {
  try { await loadSummary(); await loadCases(); await loadDocsFilter(); }
  catch (e) {
    $('#cases').innerHTML =
      `<div class="empty">Could not reach the API (${esc(e.message)}).<br>
       Is the server running, and has anything been ingested yet?</div>`;
  }
})();
