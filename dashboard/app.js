/* MarketBot dashboard: talks to the FastAPI service. */

const $ = (id) => document.getElementById(id);
let equityChart = null;

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return `${(v * 100).toFixed(1)}%`;
}
function fmtNum(v, d = 2) {
  if (v === null || v === undefined) return '—';
  return Number(v).toFixed(d);
}
function cls(v) {
  if (v === null || v === undefined) return '';
  return v > 0 ? 'pos' : v < 0 ? 'neg' : '';
}

async function loadStatus() {
  try {
    const s = await api('/api/live-status');
    const el = $('live-status');
    if (s.live_enabled) {
      el.className = 'status live';
      el.textContent = `LIVE TRADING ENABLED on ${s.exchange} (sandbox: ${s.sandbox})`;
    } else {
      el.className = 'status paper';
      el.textContent = `Paper mode — ${s.reason}`;
    }
  } catch (e) {
    $('live-status').textContent = `API unreachable: ${e.message}`;
  }
}

async function loadStrategies() {
  const data = await api('/api/strategies');
  const sel = $('strategy');
  sel.innerHTML = '';
  for (const s of data.strategies) {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = s.name;
    opt.title = s.description;
    sel.appendChild(opt);
  }
}

function renderMetrics(m) {
  if (!m) { $('metrics').innerHTML = '<p class="muted">No result yet.</p>'; return; }
  const cells = [
    ['Total return', fmtPct(m.total_return), cls(m.total_return)],
    ['Annualized', fmtPct(m.annualized_return), cls(m.annualized_return)],
    ['Sharpe', fmtNum(m.sharpe), ''],
    ['Sortino', fmtNum(m.sortino), ''],
    ['Max drawdown', fmtPct(m.max_drawdown), 'bad'],
    ['Calmar', fmtNum(m.calmar), ''],
    ['Trades', m.trade_stats.trades, ''],
    ['Win rate', m.trade_stats.win_rate === null ? '—' : fmtPct(m.trade_stats.win_rate), ''],
    ['Profit factor', fmtNum(m.trade_stats.profit_factor), ''],
    ['Expectancy', fmtNum(m.trade_stats.expectancy), cls(m.trade_stats.expectancy)],
    ['Directional acc.', m.accuracy.directional_accuracy === null ? '—' : fmtPct(m.accuracy.directional_accuracy), ''],
    ['Total fees', fmtNum(m.trade_stats.total_fees), 'muted'],
  ];
  $('metrics').innerHTML = cells
    .map(([l, v, c]) => `<div class="metric"><div class="label">${l}</div><div class="value ${c}">${v}</div></div>`)
    .join('');
}

function renderEquity(curve) {
  const ctx = $('equity');
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: curve.map((_, i) => i),
      datasets: [{
        label: 'Equity',
        data: curve,
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.08)',
        fill: true,
        pointRadius: 0,
        tension: 0.15,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid: { color: '#1f2a3a' }, ticks: { color: '#8a9bb0' } },
      },
    },
  });
}

function renderTrades(trades) {
  if (!trades.length) {
    $('trades').innerHTML = '<p class="muted">No trades. A strategy that never trades usually means a bug, not a good result.</p>';
    return;
  }
  const rows = trades.map((t) => `<tr>
    <td>${t.symbol}</td>
    <td>${t.side}</td>
    <td class="num">${fmtNum(t.entry_price, 4)}</td>
    <td class="num">${fmtNum(t.exit_price, 4)}</td>
    <td class="num ${cls(t.pnl)}">${fmtNum(t.pnl)}</td>
    <td class="num ${cls(t.pnl_pct)}">${fmtPct(t.pnl_pct)}</td>
    <td class="num">${t.bars_held}</td>
    <td><span class="muted">${t.exit_reason || ''}</span></td>
  </tr>`).join('');
  $('trades').innerHTML = `<table><thead><tr>
    <th>Symbol</th><th>Side</th><th class="num">Entry</th><th class="num">Exit</th>
    <th class="num">P&amp;L</th><th class="num">P&amp;L %</th><th class="num">Bars</th><th>Exit</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

async function runBacktest() {
  const body = {
    symbol: $('symbol').value,
    strategy: $('strategy').value,
    interval: $('interval').value,
    bars: Number($('bars').value),
    params: {},
  };
  try {
    const r = await api('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    renderMetrics(r.metrics);
    renderEquity(r.equity_curve);
    renderTrades(r.trades);
    if (r.halted) {
      $('metrics').insertAdjacentHTML('beforeend',
        `<p class="neg" style="margin-top:10px">Halted: ${r.halt_reason}</p>`);
    }
  } catch (e) {
    $('metrics').innerHTML = `<p class="neg">${e.message}</p>`;
  }
}

async function runCompare() {
  try {
    const r = await api(`/api/compare?symbol=${$('symbol').value}&interval=${$('interval').value}&bars=${$('bars').value}`);
    const rows = r.rows.map((x) => {
      if (x.error) return `<tr><td>${x.strategy}</td><td colspan="6" class="neg">${x.error}</td></tr>`;
      return `<tr>
        <td>${x.strategy}</td>
        <td class="num ${cls(x.total_return)}">${fmtPct(x.total_return)}</td>
        <td class="num">${fmtNum(x.sharpe)}</td>
        <td class="num">${fmtPct(x.max_drawdown)}</td>
        <td class="num">${x.trades}</td>
        <td class="num">${x.win_rate === null ? '—' : fmtPct(x.win_rate)}</td>
        <td class="num">${fmtNum(x.profit_factor)}</td>
      </tr>`;
    }).join('');
    $('comparison').innerHTML = `<table><thead><tr>
      <th>Strategy</th><th class="num">Return</th><th class="num">Sharpe</th>
      <th class="num">Max DD</th><th class="num">Trades</th><th class="num">Win</th><th class="num">PF</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
  } catch (e) {
    $('comparison').innerHTML = `<p class="neg">${e.message}</p>`;
  }
}

async function runFirm() {
  try {
    const r = await api('/api/firm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: $('symbol').value, bars: Number($('bars').value) }),
    });
    const analysts = r.analysts.map((a) => `
      <div class="analyst">
        <span class="role">${a.role}</span>
        <span>${a.direction} ${(a.confidence * 100).toFixed(0)}%</span>
      </div>
      <div class="muted" style="font-size:12px;margin:-4px 0 8px">${a.reasoning}</div>
    `).join('');
    $('firmout').innerHTML = `
      <div class="verdict">${r.verdict}</div>
      ${analysts}
      <div class="debate">
        Debate winner: <strong>${r.debate.winner}</strong>
        (score ${r.debate.reconciled_score.toFixed(2)}, confidence ${(r.debate.confidence * 100).toFixed(0)}%)
        <br>Decision: ${r.decision.action} via ${r.decision.strategy || '—'}
        at ${(r.decision.conviction * 100).toFixed(0)}% conviction
        <br>Risk approved: ${r.decision.risk_approved} · Portfolio approved: ${r.decision.portfolio_approved}
        ${r.decision.reject_reason ? `<br><span class="neg">Rejected: ${r.decision.reject_reason}</span>` : ''}
      </div>
    `;
  } catch (e) {
    $('firmout').innerHTML = `<p class="neg">${e.message}</p>`;
  }
}

async function showChart() {
  try {
    const r = await api(`/api/market/${$('symbol').value}?interval=${$('interval').value}&bars=${$('bars').value}`);
    const closes = r.candles.map((c) => c.close);
    const ctx = $('equity');
    if (equityChart) equityChart.destroy();
    equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: r.candles.map((c) => new Date(c.ts * 1000).toISOString().slice(0, 10)),
        datasets: [{
          label: `${r.symbol} close`,
          data: closes,
          borderColor: '#a78bfa',
          pointRadius: 0,
          tension: 0.15,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#8a9bb0', maxTicksLimit: 10 } },
          y: { grid: { color: '#1f2a3a' }, ticks: { color: '#8a9bb0' } },
        },
      },
    });
  } catch (e) {
    $('metrics').innerHTML = `<p class="neg">${e.message}</p>`;
  }
}

$('run').addEventListener('click', runBacktest);
$('compare').addEventListener('click', runCompare);
$('firm').addEventListener('click', runFirm);
$('chart').addEventListener('click', showChart);

(async function init() {
  await loadStatus();
  try {
    await loadStrategies();
  } catch (e) {
    $('live-status').textContent = `Could not load strategies: ${e.message}`;
  }
})();
