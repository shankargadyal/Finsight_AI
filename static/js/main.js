/* =============================================================
   FinSight AI v2 — main.js
   Full frontend logic: auth, charts, risk, chat, watchlist
   ============================================================= */
'use strict';

/* ── Config ────────────────────────────────────────────────── */
const API_BASE   = '';
const TIMEOUT_MS = 120_000;
const QUICK_PICKS = ['AAPL','TSLA','MSFT','NVDA','AMZN','META','GOOGL','BTC-USD'];
const PERIODS     = ['1mo','3mo','6mo','1y','2y'];
const C = { cyan:'#00D4FF', green:'#00FF9D', red:'#FF3860', amber:'#FFB800', purple:'#B44DFF', white:'#E8F4FD' };

/* ── State ─────────────────────────────────────────────────── */
let mainChart    = null;
let currentData  = null;
let allPriceData = null;
let period       = '1y';
let authToken    = localStorage.getItem('fsToken') || null;
let currentUser  = null;
let chatSessionId = 'session_' + Date.now();
let chatContext  = null;

/* ── Utilities ─────────────────────────────────────────────── */
const $    = id  => document.getElementById(id);
const qs   = sel => document.querySelector(sel);
const fmt  = (n,d=2) => typeof n==='number' ? n.toFixed(d) : '–';
const fmtD = (n,d=2) => typeof n==='number' ? `$${n.toFixed(d)}` : '–';

async function apiFetch(url, opts = {}) {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  const headers = { 'Content-Type': 'application/json', ...(opts.headers||{}) };
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
  try {
    const res  = await fetch(API_BASE + url, { signal: ctrl.signal, ...opts, headers });
    clearTimeout(timer);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  } catch (e) { clearTimeout(timer); throw e; }
}

/* ── Loading / Error ───────────────────────────────────────── */
function showLoading(msg, step) {
  $('loadingBar').classList.add('show');
  $('loadingMsg').textContent = msg;
  [0,1,2].forEach(i => $('dot'+i).className = 'step-dot' + (i<=step?' done':''));
}
function hideLoading()  { $('loadingBar').classList.remove('show'); }
function showError(msg) { $('errorBar').classList.add('show'); $('errorMsg').textContent = msg; }
function hideError()    { $('errorBar').classList.remove('show'); }
function setBusy(b)     { $('analyseBtn').disabled = b; $('btnSpinner').style.display = b ? 'inline-block' : 'none'; }
function showSection(id){ $(id).classList.add('show'); }
function hideSection(id){ $(id).classList.remove('show'); }

/* ══════════════════════════════════════════════════════════════
   BOOT
═══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  buildQuickPicks();
  buildPeriodBtns();
  buildTickerTape();
  checkHealth();
  restoreAuth();
  $('searchInput').addEventListener('keydown', e => { if(e.key==='Enter') runAnalysis(); });
});

async function checkHealth() {
  try { const d = await apiFetch('/api/health'); console.log('[health]', d.message); }
  catch { console.warn('[health] Backend not reachable'); }
}

/* ══════════════════════════════════════════════════════════════
   AUTH
═══════════════════════════════════════════════════════════════ */
function openAuth()   { $('authOverlay').classList.add('show'); switchTab('login'); }
function closeAuth()  { $('authOverlay').classList.remove('show'); }

function switchTab(tab) {
  $('loginForm').style.display    = tab==='login'    ? '' : 'none';
  $('registerForm').style.display = tab==='register' ? '' : 'none';
  $('tabLogin').classList.toggle('active',    tab==='login');
  $('tabRegister').classList.toggle('active', tab==='register');
  $('loginError').textContent    = '';
  $('registerError').textContent = '';
}

async function doLogin() {
  const email = $('loginEmail').value.trim();
  const pw    = $('loginPassword').value.trim();
  $('loginError').textContent = '';
  try {
    const d = await apiFetch('/api/auth/login', {
      method: 'POST', body: JSON.stringify({ email, password: pw })
    });
    authToken = d.token;
    localStorage.setItem('fsToken', authToken);
    currentUser = d.user;
    closeAuth();
    onLoginSuccess();
  } catch(e) { $('loginError').textContent = e.message; }
}

async function doRegister() {
  const name  = $('regName').value.trim();
  const email = $('regEmail').value.trim();
  const pw    = $('regPassword').value.trim();
  $('registerError').textContent = '';
  try {
    const d = await apiFetch('/api/auth/register', {
      method: 'POST', body: JSON.stringify({ name, email, password: pw })
    });
    authToken = d.token;
    localStorage.setItem('fsToken', authToken);
    currentUser = d.user;
    closeAuth();
    onLoginSuccess();
  } catch(e) { $('registerError').textContent = e.message; }
}

async function doLogout() {
  try { await apiFetch('/api/auth/logout', { method: 'POST' }); } catch {}
  authToken = null; currentUser = null;
  localStorage.removeItem('fsToken');
  onLogoutSuccess();
}

async function restoreAuth() {
  if (!authToken) return;
  try {
    const d = await apiFetch('/api/auth/me');
    currentUser = d;
    onLoginSuccess();
  } catch { authToken = null; localStorage.removeItem('fsToken'); }
}

function onLoginSuccess() {
  const name = currentUser?.name || 'User';
  $('navUserName').textContent   = name;
  $('navAuthLink').style.display = 'none';
  $('navLogoutBtn').style.display = '';
  $('navUserSection').style.display = '';
  $('topAuthLabel').textContent  = name;
  $('topAuthBtn').onclick        = doLogout;
}

function onLogoutSuccess() {
  $('navUserName').textContent   = 'Guest User';
  $('navAuthLink').style.display = '';
  $('navLogoutBtn').style.display = 'none';
  $('navUserSection').style.display = 'none';
  $('topAuthLabel').textContent  = 'Sign In';
  $('topAuthBtn').onclick        = openAuth;
  closePanel('userDashPanel');
  closePanel('watchlistPanel');
}

/* ══════════════════════════════════════════════════════════════
   QUICK PICKS + PERIOD BUTTONS
═══════════════════════════════════════════════════════════════ */
function buildQuickPicks() {
  const wrap = $('quickPicks');
  const nav  = $('navQuickPicks');
  QUICK_PICKS.forEach(sym => {
    // Chip
    const b = document.createElement('button');
    b.className = 'chip'; b.id = 'chip-'+sym; b.textContent = sym;
    b.onclick = () => { $('searchInput').value = sym; runAnalysis(); };
    wrap.appendChild(b);
    // Nav
    const n = document.createElement('button');
    n.className = 'nav-quick-btn'; n.textContent = sym;
    n.onclick = () => { $('searchInput').value = sym; runAnalysis(); };
    nav.appendChild(n);
  });
}

function buildPeriodBtns() {
  const wrap = $('periodBtns');
  PERIODS.forEach(p => {
    const b = document.createElement('button');
    b.className = 'ctrl-btn' + (p==='1y'?' active':'');
    b.textContent = p; b.id = 'period-'+p;
    b.onclick = () => changePeriod(p);
    wrap.appendChild(b);
  });
}

function buildTickerTape() {
  const tickers = [
    {sym:'AAPL', price:'213.42', chg:'+1.2%'},{sym:'TSLA', price:'177.60', chg:'-0.8%'},
    {sym:'MSFT', price:'422.15', chg:'+0.5%'},{sym:'NVDA', price:'131.20', chg:'+2.1%'},
    {sym:'AMZN', price:'197.85', chg:'+0.3%'},{sym:'META', price:'583.40', chg:'+1.7%'},
    {sym:'GOOGL',price:'178.50', chg:'-0.4%'},{sym:'BTC', price:'97,200', chg:'+3.2%'},
    {sym:'SPY',  price:'591.10', chg:'+0.6%'},{sym:'JPM',  price:'244.80', chg:'+0.9%'},
  ];
  const tape = $('tickerTape');
  const items = [...tickers, ...tickers].map(t => {
    const isPos = t.chg.startsWith('+');
    return `<div class="tape-item">
      <span style="color:var(--dim);font-weight:600">${t.sym}</span>
      <span>$${t.price}</span>
      <span style="color:${isPos?'var(--green)':'var(--red)'}">${t.chg}</span>
    </div>`;
  }).join('');
  tape.innerHTML = items;
}

/* ══════════════════════════════════════════════════════════════
   MAIN ANALYSIS
═══════════════════════════════════════════════════════════════ */
async function runAnalysis() {
  const sym = $('searchInput').value.trim().toUpperCase();
  if (!sym) { showError('Please enter a ticker symbol.'); return; }

  setBusy(true); hideError();
  hideSection('dashboard'); hideSection('tickerHeader');
  $('emptyState').style.display = 'none';

  const msgs = ['Fetching price data…','Training ML models…','Analysing sentiment & risk…'];
  for (let i=0; i<3; i++) { showLoading(msgs[i], i); await new Promise(r=>setTimeout(r, i===1?400:200)); }

  try {
    const data = await apiFetch(`/api/analyze/${sym}`);
    currentData = data;
    allPriceData = data.historical;
    hideLoading();

    renderTickerHeader(data);
    renderMainChart(data, period);
    renderModelTable(data);
    renderSentimentGauge(data.sentiment);
    renderIndicators(data);
    renderAgreementBars(data);
    renderRecommendation(data);
    renderRisk(data.risk);
    renderNewsAI(data.news_summary);
    renderNewsGrid(data.articles);
    renderStatsStrip(data);
    renderPriceRange(data);

    // Set chat context
    chatContext = {
      symbol:          data.symbol,
      last_close:      data.last_close,
      recommendation:  data.recommendation,
      confidence:      data.confidence,
      trend:           data.trend,
      risk_level:      data.risk?.level,
      risk_score:      data.risk?.score,
      sentiment_label: data.sentiment?.label,
      sentiment_score: data.sentiment?.score,
      ensemble_d7:     data.ensemble?.length ? data.ensemble[data.ensemble.length-1] : null,
    };

    // Update chip + watchlist btn
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    const chip = $('chip-'+sym);
    if (chip) chip.classList.add('active');
    $('btnAddWatch').style.display = authToken ? '' : 'none';

    showSection('tickerHeader');
    showSection('dashboard');
    QUICK_PICKS.includes(sym) || void 0;

  } catch(e) {
    hideLoading();
    showError(e.message || 'Analysis failed. Please try again.');
    $('emptyState').style.display = '';
  } finally {
    setBusy(false);
  }
}

async function changePeriod(p) {
  period = p;
  document.querySelectorAll('.ctrl-btn').forEach(b => b.classList.remove('active'));
  $('period-'+p)?.classList.add('active');
  if (!currentData) return;
  try {
    const sym = currentData.symbol;
    const d   = await apiFetch(`/api/stock/${sym}?period=${p}`);
    const hist = d.data.map(r => ({ date: r.date, close: r.close, sma20: r.sma20, sma50: r.sma50 }));
    allPriceData = hist;
    renderMainChart({ ...currentData, historical: hist }, p);
  } catch {}
}

/* ══════════════════════════════════════════════════════════════
   RENDER FUNCTIONS
═══════════════════════════════════════════════════════════════ */
function renderTickerHeader(data) {
  $('headerSymbol').textContent = data.symbol;
  $('headerName').textContent   = data.info?.name || '';
  $('headerPrice').textContent  = fmtD(data.last_close);
  const chg  = data.change, chgp = data.change_pct;
  const pos  = chg >= 0;
  $('headerChange').textContent = `${pos?'+':''}${fmt(chg)} (${pos?'+':''}${fmt(chgp)}%)`;
  $('headerChange').style.color = pos ? C.green : C.red;
  // Risk pill in header
  if (data.risk) {
    const rp = $('headerRisk');
    rp.textContent = data.risk.level;
    rp.style.background = data.risk.color + '20';
    rp.style.border = `1px solid ${data.risk.color}60`;
    rp.style.color  = data.risk.color;
  }
}

/* ── MAIN CHART ─────────────────────────────────────────────── */
function renderMainChart(data, p) {
  const hist    = data.historical || [];
  const futDates = data.future_dates || [];
  const lr = data.linear_reg?.predictions || [];
  const ar = data.arima?.predictions      || [];
  const ls = data.lstm?.predictions       || [];
  const en = data.ensemble                || [];
  const arLo = data.arima?.conf_int_lower || [];
  const arHi = data.arima?.conf_int_upper || [];

  const labels  = [...hist.map(r=>r.date), ...futDates];
  const closes  = hist.map(r=>r.close);
  const sma20   = hist.map(r=>r.sma20);
  const sma50   = hist.map(r=>r.sma50);
  const pad     = (arr) => [...Array(hist.length).fill(null), ...arr];

  const ctx = document.getElementById('mainChart').getContext('2d');
  if (mainChart) mainChart.destroy();

  mainChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'Close', data:[...closes,...Array(futDates.length).fill(null)], borderColor:C.cyan, borderWidth:2, pointRadius:0, tension:.3, fill:false },
        { label:'LinReg', data:pad(lr), borderColor:C.amber, borderWidth:1.5, borderDash:[5,3], pointRadius:0, tension:.3, fill:false },
        { label:'ARIMA',  data:pad(ar), borderColor:C.purple,borderWidth:1.5, borderDash:[5,3], pointRadius:0, tension:.3, fill:false },
        { label:'LSTM',   data:pad(ls), borderColor:C.green, borderWidth:1.5, borderDash:[5,3], pointRadius:0, tension:.3, fill:false },
        { label:'Ensemble',data:pad(en),borderColor:'#fff',  borderWidth:2.5, pointRadius:3,    pointBackgroundColor:'#fff', tension:.3, fill:false },
        { label:'ARIMA CI Low', data:pad(arLo), borderColor:'rgba(180,77,255,.25)', borderWidth:1, borderDash:[2,4], pointRadius:0, fill:false },
        { label:'ARIMA CI High',data:pad(arHi), borderColor:'rgba(180,77,255,.25)', borderWidth:1, borderDash:[2,4], pointRadius:0, fill:'-1', backgroundColor:'rgba(180,77,255,.05)' },
        { label:'SMA20', data:[...sma20,...Array(futDates.length).fill(null)], borderColor:'rgba(180,77,255,.5)', borderWidth:1, borderDash:[4,4], pointRadius:0, tension:.3, fill:false },
        { label:'SMA50', data:[...sma50,...Array(futDates.length).fill(null)], borderColor:'rgba(255,184,0,.5)',  borderWidth:1, borderDash:[4,4], pointRadius:0, tension:.3, fill:false },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode:'index', intersect:false },
      plugins: {
        legend: { display:false },
        tooltip: {
          backgroundColor:'rgba(10,22,40,.95)', borderColor:'rgba(0,212,255,.3)', borderWidth:1,
          titleColor:C.cyan, bodyColor:'#aac', titleFont:{family:"'JetBrains Mono',monospace",size:10},
          bodyFont:{family:"'JetBrains Mono',monospace",size:10},
          callbacks: { label: ctx => ctx.parsed.y != null ? `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}` : null }
        }
      },
      scales: {
        x: { ticks:{color:'#3D5A80',font:{size:9,family:"'JetBrains Mono',monospace"},maxTicksLimit:10}, grid:{color:'rgba(0,212,255,.04)'} },
        y: { ticks:{color:'#3D5A80',font:{size:9,family:"'JetBrains Mono',monospace"},callback:v=>`$${v.toFixed(0)}`}, grid:{color:'rgba(0,212,255,.06)'} }
      }
    }
  });

  // Trend badge
  const tb = $('trendBadge');
  const up = data.trend === 'up';
  tb.textContent = up ? '▲ UPTREND' : data.trend === 'down' ? '▼ DOWNTREND' : '— NEUTRAL';
  tb.style.background = up ? 'rgba(0,255,157,.1)' : 'rgba(255,56,96,.1)';
  tb.style.border     = `1px solid ${up ? 'rgba(0,255,157,.4)' : 'rgba(255,56,96,.4)'}`;
  tb.style.color      = up ? C.green : C.red;
  tb.classList.add('show');
}

/* ── MODEL TABLE ────────────────────────────────────────────── */
function renderModelTable(data) {
  const tbody = $('modelTableBody');
  const last  = data.last_close;
  const models = [
    { name:'Linear Reg.',  preds: data.linear_reg?.predictions||[], metric:`RMSE: ${fmt(data.linear_reg?.rmse)}`, color:C.amber },
    { name:'ARIMA (5,1,0)',preds: data.arima?.predictions||[],       metric:`AIC: ${fmt(data.arima?.aic,0)}`,     color:C.purple },
    { name:'LSTM',         preds: data.lstm?.predictions||[],        metric:`RMSE: ${fmt(data.lstm?.rmse)}`,      color:C.green  },
    { name:'Ensemble',     preds: data.ensemble||[],                 metric:'Weighted Avg',                      color:'#fff',  bold:true },
  ];
  tbody.innerHTML = models.map(m => {
    const d7  = m.preds[6] ?? null;
    const chg = d7 != null ? ((d7 - last) / last * 100) : null;
    const pos = chg != null && chg >= 0;
    return `<tr>
      <td><span class="model-name" style="color:${m.color}">${m.name}</span></td>
      <td>${fmtD(m.preds[0])}</td>
      <td>${fmtD(m.preds[2])}</td>
      <td>${fmtD(m.preds[4])}</td>
      <td style="font-weight:${m.bold?'700':'400'}">${fmtD(d7)}</td>
      <td style="color:${pos?C.green:C.red}">${chg!=null?`${pos?'+':''}${fmt(chg)}%`:'–'}</td>
      <td style="color:var(--muted);font-size:.68rem">${m.metric}</td>
    </tr>`;
  }).join('');
}

/* ── SENTIMENT GAUGE ────────────────────────────────────────── */
function renderSentimentGauge(s) {
  if (!s) return;
  const score = s.score ?? 0;
  // Arc: cx=100 cy=95 r=72, from 180° to 0°
  const cx=100, cy=95, r=72;
  const startA = Math.PI, endA = 0;
  const toXY = a => [cx+r*Math.cos(a), cy-r*Math.sin(a)];
  const arcPath = (a1,a2) => {
    const [x1,y1]=toXY(a1), [x2,y2]=toXY(a2);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  $('gaugeTrack').setAttribute('d', arcPath(startA, endA));
  const t     = (score + 1) / 2; // 0→1
  const angle = Math.PI - t * Math.PI;
  $('gaugeFill').setAttribute('d', arcPath(startA, angle));
  const [nx,ny] = toXY(angle);
  $('gaugeNeedle').setAttribute('x1', String(cx)); $('gaugeNeedle').setAttribute('y1', String(cy));
  $('gaugeNeedle').setAttribute('x2', String(cx+(nx-cx)*.85));
  $('gaugeNeedle').setAttribute('y2', String(cy+(ny-cy)*.85));
  $('gaugeScore').textContent = score.toFixed(3);
  const col = score > 0.1 ? 'var(--green)' : score < -0.1 ? 'var(--red)' : 'var(--amber)';
  $('gaugeFill').setAttribute('stroke', col);
  $('gaugeNeedle').setAttribute('stroke', col);
  $('gaugeHub').setAttribute('fill', col);
  $('gaugeScore').setAttribute('fill', col);
  $('sentLabel').textContent = s.label?.toUpperCase() || '–';
  $('sentLabel').style.color = col;
  $('bdownPos').style.width  = (s.positive_pct||0)+'%';
  $('bdownNeu').style.width  = (s.neutral_pct||0)+'%';
  $('bdownNeg').style.width  = (s.negative_pct||0)+'%';
  $('bdownPosLbl').textContent = `+${s.positive_pct?.toFixed(0)||0}%`;
  $('bdownNeuLbl').textContent = `${s.neutral_pct?.toFixed(0)||0}%`;
  $('bdownNegLbl').textContent = `${s.negative_pct?.toFixed(0)||0}%`;
}

/* ── INDICATORS ─────────────────────────────────────────────── */
function renderIndicators(data) {
  const hist = data.historical || [];
  const last = hist[hist.length-1] || {};
  const rsi  = last.rsi;
  const rsiSig = rsi == null ? '–' : rsi > 70 ? '🔴 Overbought' : rsi < 30 ? '🟢 Oversold' : '🟡 Neutral';
  const sma20 = last.sma20, sma50 = last.sma50, close = last.close;
  const maGolden = sma20 != null && sma50 != null ? (sma20 > sma50 ? '🟢 Golden Cross' : '🔴 Death Cross') : '–';
  const indicators = [
    { name:'RSI (14)', value: rsi!=null?fmt(rsi):'–', signal: rsiSig },
    { name:'SMA 20',   value: fmtD(sma20), signal: close&&sma20 ? (close>sma20?'🟢 Above':'🔴 Below') : '–' },
    { name:'SMA 50',   value: fmtD(sma50), signal: close&&sma50 ? (close>sma50?'🟢 Above':'🔴 Below') : '–' },
    { name:'MA Cross', value: sma20&&sma50?`${fmt(sma20-sma50,1)}`:'–', signal: maGolden },
  ];
  $('indicatorsGrid').innerHTML = indicators.map(i => `
    <div class="ind-item">
      <div class="ind-name">${i.name}</div>
      <div class="ind-value" style="color:var(--cyan)">${i.value}</div>
      <div class="ind-signal" style="color:var(--dim)">${i.signal}</div>
    </div>`).join('');
}

/* ── AGREEMENT BARS ─────────────────────────────────────────── */
function renderAgreementBars(data) {
  const last = data.last_close;
  const models = [
    { name:'Linear Reg.', preds:data.linear_reg?.predictions||[], color:C.amber },
    { name:'ARIMA',       preds:data.arima?.predictions||[],      color:C.purple },
    { name:'LSTM',        preds:data.lstm?.predictions||[],       color:C.green  },
  ];
  $('agreementBars').innerHTML = models.map(m => {
    const d7  = m.preds[6];
    const pct = d7 != null ? Math.min(Math.abs((d7-last)/last*100)*10, 100) : 0;
    const up  = d7 != null && d7 > last;
    return `<div class="agree-row">
      <div class="agree-top">
        <span style="color:${m.color}">${m.name}</span>
        <span style="color:${up?C.green:C.red}">${d7!=null?`${up?'▲':'▼'} ${fmtD(d7)}`:'–'}</span>
      </div>
      <div class="agree-bar-bg">
        <div class="agree-bar-fill" style="width:${pct}%;background:${m.color}"></div>
      </div>
    </div>`;
  }).join('');
}

/* ── RECOMMENDATION ─────────────────────────────────────────── */
function renderRecommendation(data) {
  const rec  = data.recommendation || 'HOLD';
  const conf = data.confidence || 0;
  const ens  = data.ensemble || [];
  const d7   = ens[ens.length-1];
  const colMap = {
    'STRONG BUY':C.green,'BUY':C.green,'HOLD':C.amber,'SELL':C.red,'STRONG SELL':C.red
  };
  const col = colMap[rec] || C.amber;
  const rb = $('recBadge');
  rb.textContent    = rec;
  rb.style.background = col + '15';
  rb.style.border   = `2px solid ${col}60`;
  rb.style.color    = col;
  $('recPrice').textContent = fmtD(d7);
  if (d7 != null) {
    const chg = ((d7 - data.last_close) / data.last_close * 100);
    $('recDelta').textContent = `${chg>=0?'+':''}${fmt(chg)}% from current`;
    $('recDelta').style.color = chg>=0 ? C.green : C.red;
  }
  // Confidence ring
  const circ = 2 * Math.PI * 46;
  const filled = (conf/100) * circ;
  $('confRing').setAttribute('stroke-dasharray', `${filled} ${circ}`);
  $('confRing').setAttribute('stroke', conf>=70?C.green:conf>=50?C.amber:C.red);
  $('confPct').textContent = `${Math.round(conf)}%`;
  // Metrics
  const lr = data.linear_reg, ar = data.arima, ls = data.lstm;
  $('metricsGrid').innerHTML = [
    { label:'LR RMSE',    val:fmt(lr?.rmse) },
    { label:'ARIMA AIC',  val:fmt(ar?.aic,0) },
    { label:'LSTM RMSE',  val:fmt(ls?.rmse) },
    { label:'LSTM MAE',   val:fmt(ls?.mae) },
  ].map(m=>`<div class="metric-box">
    <div class="metric-label">${m.label}</div>
    <div class="metric-value">${m.val}</div>
  </div>`).join('');
  // Reasoning
  const rb2 = $('reasoningBox'), rt = $('reasoningText');
  rb2.style.display = '';
  const sent = data.sentiment?.label || 'neutral';
  const up = data.trend === 'up';
  rt.innerHTML = `
    Ensemble forecast projects <strong style="color:${up?C.green:C.red}">${up?'upward':'downward'}</strong>
    movement over 7 trading days. News sentiment is <strong>${sent}</strong>
    (score: ${fmt(data.sentiment?.score??0,3)}).
    ${conf>=70?'Strong':'Moderate'} model agreement at ${Math.round(conf)}% confidence.
    Risk level: <strong style="color:${data.risk?.color||C.amber}">${data.risk?.level||'–'}</strong>
    (score: ${Math.round(data.risk?.score||0)}/100).
    ${rec==='HOLD'?'Mixed signals suggest a wait-and-see approach.':'Signals support the ' + rec + ' recommendation.'}
  `;
}

/* ── RISK ANALYZER ──────────────────────────────────────────── */
function renderRisk(risk) {
  if (!risk) return;
  $('riskCard').style.display = '';
  $('riskMiniCard').style.display = '';
  // Full risk card
  drawRiskMeter(risk.score, risk.color);
  $('riskLevelBadge').textContent = risk.level;
  $('riskLevelBadge').style.color = risk.color;
  $('riskLevelBadge').style.borderColor = risk.color + '60';
  $('riskExplanation').textContent = risk.explanation;
  $('riskFactors').innerHTML = (risk.factors||[]).map(f => `
    <div class="risk-factor-row">
      <span class="rf-name">${f.name}</span>
      <span class="rf-value">${f.value}</span>
      <span style="font-family:var(--fm);font-size:.62rem;color:var(--muted)">${f.weight}</span>
      <div class="rf-bar-wrap">
        <div class="rf-bar" style="width:${f.score}%;background:${risk.color}"></div>
      </div>
    </div>`).join('');
  // Mini card
  $('riskMiniScore').textContent = `${Math.round(risk.score)}/100`;
  $('riskMiniScore').style.color = risk.color;
  $('riskMiniLevel').textContent = risk.level;
  $('riskMiniLevel').style.color = risk.color;
  $('riskMiniVol').textContent   = `Vol: ${risk.volatility}% ann.`;
  $('riskMiniBar').style.width      = risk.score + '%';
  $('riskMiniBar').style.background = risk.color;
}

function drawRiskMeter(score, color) {
  const cx=100, cy=100, r=76;
  const startA = Math.PI, endA = 0;
  const toXY = a => [cx+r*Math.cos(a), cy-r*Math.sin(a)];
  const arcPath = (a1,a2) => {
    const [x1,y1]=toXY(a1), [x2,y2]=toXY(a2);
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };
  $('riskTrack').setAttribute('d', arcPath(startA, endA));
  const t = score / 100;
  const angle = Math.PI - t * Math.PI;
  $('riskFill').setAttribute('d', arcPath(startA, angle));
  const [nx,ny] = toXY(angle);
  $('riskNeedle').setAttribute('x1','100'); $('riskNeedle').setAttribute('y1','100');
  $('riskNeedle').setAttribute('x2', String(100+(nx-cx)*.8));
  $('riskNeedle').setAttribute('y2', String(100+(ny-cy)*.8));
  $('riskScore').textContent = `${Math.round(score)}/100`;
}

/* ── AI NEWS SUMMARY ────────────────────────────────────────── */
function renderNewsAI(ns) {
  if (!ns) return;
  $('newsSummaryCard').style.display = '';
  const impBadge = $('newsImpactBadge');
  impBadge.textContent    = ns.impact_level;
  impBadge.style.background = ns.impact_color + '20';
  impBadge.style.border     = `1px solid ${ns.impact_color}50`;
  impBadge.style.color      = ns.impact_color;
  $('newsSummaryText').textContent = ns.summary;
  $('newsBullets').innerHTML = (ns.bullets||[]).map(b=>`<div class="news-bullet">${b}</div>`).join('');
  $('newsThemesList').innerHTML = (ns.key_themes||[]).map(t=>`<div class="news-theme-tag">${t}</div>`).join('');
  $('newsAnalystNote').textContent = ns.analyst_note || '';
  const aib = $('aiPoweredBadge');
  aib.textContent    = ns.ai_powered ? '🤖 AI Powered' : '📊 Rule-Based';
  aib.style.background = ns.ai_powered ? 'rgba(0,255,157,.1)' : 'rgba(255,184,0,.1)';
  aib.style.color      = ns.ai_powered ? C.green : C.amber;
  aib.style.border     = `1px solid ${ns.ai_powered?C.green:C.amber}40`;
  aib.style.padding    = '.12rem .45rem';
  aib.style.borderRadius = '3px';
  aib.style.fontFamily = 'var(--fm)';
  aib.style.fontSize   = '.6rem';
}

/* ── NEWS GRID ──────────────────────────────────────────────── */
function renderNewsGrid(articles) {
  if (!articles?.length) return;
  $('newsCard').style.display = '';
  const colMap = { positive:C.green, negative:C.red, neutral:C.amber };
  $('newsGrid').innerHTML = articles.slice(0,8).map(a => `
    <div class="news-item">
      <div class="news-item-title">${a.title}</div>
      <div class="news-item-foot">
        <span>${a.source||''}</span>
        <span>
          <span class="sent-dot" style="background:${colMap[a.label]||C.amber}"></span>
          ${a.label}  ${fmt(a.compound,3)}
        </span>
      </div>
    </div>`).join('');
}

/* ── STATS STRIP ────────────────────────────────────────────── */
function renderStatsStrip(data) {
  const sent = data.sentiment || {};
  const risk = data.risk || {};
  const stats = [
    { label:'ARTICLES',   val: sent.article_count ?? '–' },
    { label:'POS SENT',   val: sent.positive_pct != null ? sent.positive_pct+'%' : '–' },
    { label:'ANN VOL',    val: risk.volatility != null ? risk.volatility+'%' : '–' },
    { label:'MAX DD',     val: risk.max_drawdown != null ? risk.max_drawdown+'%' : '–' },
    { label:'RISK SCORE', val: risk.score != null ? Math.round(risk.score) : '–' },
    { label:'VOL TREND',  val: risk.volatility_trend || '–' },
  ];
  const ss = $('statsStrip');
  ss.innerHTML = stats.map(s=>`
    <div class="stat-box">
      <div class="stat-label">${s.label}</div>
      <div class="stat-val">${s.val}</div>
    </div>`).join('');
  ss.classList.add('show');
}

/* ── PRICE RANGE ────────────────────────────────────────────── */
function renderPriceRange(data) {
  const pr = data.price_range;
  if (!pr || !pr.target) return;
  $('priceRangeCard').style.display = '';
  $('priceRangeLow').textContent    = fmtD(pr.low);
  $('priceRangeTarget').textContent = fmtD(pr.target);
  $('priceRangeHigh').textContent   = fmtD(pr.high);
  // Position target dot on bar
  const pct = pr.low && pr.high ? ((pr.target - pr.low) / (pr.high - pr.low) * 100) : 50;
  $('priceRangeBar').style.width = '100%';
  $('priceRangeDot').style.left  = Math.round(Math.min(Math.max(pct,2),98)) + '%';
}

/* ══════════════════════════════════════════════════════════════
   CHAT
═══════════════════════════════════════════════════════════════ */
function toggleChat() {
  const sb  = $('chatSidebar');
  const fab = $('chatFab');
  const open = sb.classList.toggle('open');
  fab.classList.toggle('hidden', open);
}

function sendSuggestion(el) {
  $('chatInput').value = el.textContent;
  sendChat();
}

async function sendChat() {
  const input = $('chatInput');
  const msg   = input.value.trim();
  if (!msg) return;
  input.value = '';
  $('chatSuggestions').style.display = 'none';
  appendChatMsg('user', msg);
  showTyping();
  try {
    const d = await apiFetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message: msg, session_id: chatSessionId, context: chatContext })
    });
    hideTyping();
    appendChatMsg('assistant', d.reply);
  } catch(e) {
    hideTyping();
    appendChatMsg('assistant', 'Sorry, I could not reach the server. Please try again.');
  }
}

function appendChatMsg(role, text) {
  const msgs = $('chatMessages');
  const div  = document.createElement('div');
  div.className = `chat-msg ${role}`;
  // Convert **bold** markdown
  const html = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>');
  div.innerHTML = `<div class="chat-bubble">${html}</div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

let typingEl = null;
function showTyping() {
  const msgs = $('chatMessages');
  typingEl   = document.createElement('div');
  typingEl.className = 'chat-msg assistant';
  typingEl.innerHTML = `<div class="chat-bubble chat-typing">
    <div class="chat-typing-dots">
      <div class="chat-typing-dot"></div>
      <div class="chat-typing-dot"></div>
      <div class="chat-typing-dot"></div>
    </div></div>`;
  msgs.appendChild(typingEl);
  msgs.scrollTop = msgs.scrollHeight;
}
function hideTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

/* ══════════════════════════════════════════════════════════════
   WATCHLIST
═══════════════════════════════════════════════════════════════ */
async function addToWatchlist() {
  if (!authToken) { openAuth(); return; }
  const sym = currentData?.symbol;
  if (!sym) return;
  try {
    await apiFetch(`/api/watchlist/${sym}`, { method: 'POST' });
    $('btnAddWatch').textContent = '★ Saved';
    $('btnAddWatch').style.color  = C.amber;
  } catch(e) { console.warn(e.message); }
}

async function showWatchlistPanel() {
  if (!authToken) { openAuth(); return; }
  closePanel('userDashPanel');
  try {
    const d = await apiFetch('/api/watchlist');
    const wrap = $('watchlistItems');
    if (!d.watchlist?.length) {
      wrap.innerHTML = '<p style="color:var(--muted);font-size:.8rem;text-align:center;padding:1rem">No symbols in watchlist yet. Analyse a stock and click ★</p>';
    } else {
      wrap.innerHTML = d.watchlist.map(w => `
        <div class="watchlist-card" onclick="loadFromWatchlist('${w.symbol}')">
          <div class="wl-sym">${w.symbol}</div>
          <div class="wl-date">Added ${w.added_at?.split('T')[0]||''}</div>
          <button class="wl-remove" onclick="removeFromWL(event,'${w.symbol}')">Remove</button>
        </div>`).join('');
    }
  } catch {}
  $('watchlistPanel').style.display = '';
}

async function removeFromWL(e, sym) {
  e.stopPropagation();
  try { await apiFetch(`/api/watchlist/${sym}`, { method: 'DELETE' }); showWatchlistPanel(); }
  catch {}
}

function loadFromWatchlist(sym) {
  $('searchInput').value = sym;
  closePanel('watchlistPanel');
  runAnalysis();
}

/* ══════════════════════════════════════════════════════════════
   USER DASHBOARD
═══════════════════════════════════════════════════════════════ */
async function showDashboardPanel() {
  if (!authToken) { openAuth(); return; }
  closePanel('watchlistPanel');
  try {
    const d = await apiFetch('/api/dashboard');
    // Searches
    $('dashSearches').innerHTML = (d.recent_searches||[]).slice(0,6).map(s=>
      `<div class="upanel-item" onclick="quickLoad('${s.symbol}')">
         <span>${s.symbol}</span>
         <span style="color:var(--muted);font-size:.65rem">${s.last_searched?.split('T')[0]||''}</span>
       </div>`).join('') || '<div style="color:var(--muted);font-size:.75rem">No recent searches</div>';
    // Portfolio risk
    const pr = d.portfolio_risk;
    const col = pr.label==='Low'?C.green:pr.label==='High'?C.red:C.amber;
    $('dashRisk').innerHTML = `
      <div class="upanel-risk-score" style="color:${col}">${pr.avg_score||'–'}</div>
      <div class="upanel-risk-label" style="color:${col}">${pr.label} Risk</div>
      <div style="font-family:var(--fm);font-size:.6rem;color:var(--muted);text-align:center">Avg across ${d.prediction_history?.length||0} analyses</div>`;
    // Prediction history
    $('dashPredictions').innerHTML = (d.prediction_history||[]).length ?
      `<table class="upanel-table">
         <thead><tr><th>SYMBOL</th><th>PRICE</th><th>TARGET</th><th>REC</th><th>CONF</th><th>RISK</th><th>DATE</th></tr></thead>
         <tbody>${(d.prediction_history||[]).map(p=>{
           const col = {STRONG_BUY:C.green,BUY:C.green,HOLD:C.amber,SELL:C.red,STRONG_SELL:C.red}[p.recommendation] || C.amber;
           return `<tr>
             <td style="cursor:pointer;color:var(--cyan)" onclick="quickLoad('${p.symbol}')">${p.symbol}</td>
             <td>${p.last_close?'$'+p.last_close.toFixed(2):'–'}</td>
             <td>${p.ensemble_d7?'$'+p.ensemble_d7.toFixed(2):'–'}</td>
             <td style="color:${col};font-weight:700">${p.recommendation||'–'}</td>
             <td>${p.confidence?Math.round(p.confidence)+'%':'–'}</td>
             <td style="color:${p.risk_level==='High Risk'?C.red:p.risk_level==='Low Risk'?C.green:C.amber}">${p.risk_level||'–'}</td>
             <td style="color:var(--muted)">${p.predicted_at?.split('T')[0]||''}</td>
           </tr>`;}).join('')}
         </tbody>
       </table>` : '<div style="color:var(--muted);font-size:.75rem;text-align:center;padding:1rem">No predictions yet. Run an analysis to see history here.</div>';
  } catch {}
  $('userDashPanel').style.display = '';
}

function quickLoad(sym) {
  $('searchInput').value = sym;
  closePanel('userDashPanel');
  runAnalysis();
}

function closePanel(id) { $(id).style.display = 'none'; }

function setNavActive(el) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  el.classList.add('active');
}
