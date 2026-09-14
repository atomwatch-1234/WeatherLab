// weather+irrML Interactive Dashboard Client (5-Model Arena + 100% Open-Meteo AI Forecaster)

let curvesChart = null;
let metricsChart = null;
let eventChart = null;
let weather24hChart = null;
let futureForecastChart = null;
let validationChart = null;

let currentDate = "2026-05-21";
let currentTarget = "total";
let currentFutureTarget = "total";
let currentValDate = "2026-05-21";
let currentValTarget = "total";
let currentValRes = "5m";
let currentForecastRange = 168;
let futureForecastData = null;
let allDatesList = [];

document.addEventListener("DOMContentLoaded", () => {
  loadSummary();
  loadDates();
  loadCloudAnalysis();
  initNowcastSim();
  loadFutureForecast('total', false);
  loadValidationData('2026-05-21', 'total', '5m');
  startAutoRefreshTimer();
});

// ==========================================
// 1. TAB SWITCHING
// ==========================================
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
  
  document.querySelectorAll(".tab-btn").forEach(el => {
    el.classList.remove("active", "bg-sky-600", "text-white", "shadow-sm", "font-bold");
    el.classList.add("text-slate-500", "font-semibold");
  });

  const target = document.getElementById(tabId);
  if (target) target.classList.remove("hidden");

  // Highlight active button
  const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find(btn => 
    btn.getAttribute("onclick") && btn.getAttribute("onclick").includes(tabId)
  );
  if (activeBtn) {
    activeBtn.classList.add("active", "bg-sky-600", "text-white", "shadow-sm", "font-bold");
    activeBtn.classList.remove("text-slate-500", "font-semibold");
  }

  // Handle Chart resizing, lazy-loading, and date synchronization
  if (tabId === 'tab-arena' && metricsChart) {
    setTimeout(() => metricsChart.resize(), 50);
  } else if (tabId === 'tab-curves') {
    const sel = document.getElementById("dateSelect");
    if (sel && sel.value !== currentDate) sel.value = currentDate;
    loadCurvesForDate(currentDate);
    if (curvesChart) setTimeout(() => curvesChart.resize(), 50);
  } else if (tabId === 'tab-future') {
    const valSel = document.getElementById("valDateSelect");
    if (valSel && valSel.value !== currentDate) valSel.value = currentDate;
    loadValidationData(currentDate, currentValTarget, currentValRes);
    if (!futureForecastData) {
      loadFutureForecast(currentFutureTarget, false);
    } else if (futureForecastChart) {
      setTimeout(() => futureForecastChart.resize(), 50);
    }
  } else if (tabId === 'tab-weather-24h') {
    const wSel = document.getElementById("weatherDateSelect");
    const dateToLoad = (wSel && wSel.value) ? wSel.value : "today";
    loadWeather24h(dateToLoad);
    if (weather24hChart) setTimeout(() => weather24hChart.resize(), 50);
  } else if (tabId === 'tab-simulator') {
    const ncSel = document.getElementById("nowcastDateSelect");
    if (ncSel && ncSel.value !== currentDate) ncSel.value = currentDate;
    nowcastSimDate = currentDate;
    if (!nowcastSimData || nowcastSimData.date !== currentDate) {
      loadNowcastSim(currentDate, nowcastSimHour, nowcastSimTarget);
    }
    if (nowcastChart) setTimeout(() => nowcastChart.resize(), 50);
  }
}

// ==========================================
// 2. TAB 1: 5-MODEL ARENA SUMMARY
// ==========================================
async function loadSummary() {
  try {
    const res = await fetch("/api/models/summary");
    const data = await res.json();
    if (data.status !== "success") return;

    const m = data.models;
    if (m.model_1) {
      document.getElementById("m1-r2").textContent = m.model_1.r2.toFixed(4);
      document.getElementById("m1-rmse").textContent = m.model_1.rmse.toFixed(2) + " kW";
    }
    if (m.model_2) {
      document.getElementById("m2-r2").textContent = m.model_2.r2.toFixed(4);
      document.getElementById("m2-rmse").textContent = m.model_2.rmse.toFixed(2) + " kW";
    }
    if (m.model_3) {
      document.getElementById("m3-r2").textContent = m.model_3.r2.toFixed(4);
      document.getElementById("m3-rmse").textContent = m.model_3.rmse.toFixed(2) + " kW";
    }
    if (m.model_4) {
      document.getElementById("m4-r2").textContent = m.model_4.r2.toFixed(4);
      document.getElementById("m4-rmse").textContent = m.model_4.rmse.toFixed(2) + " kW";
    }
    if (m.model_5) {
      document.getElementById("m5-r2").textContent = m.model_5.r2.toFixed(4);
      document.getElementById("m5-rmse").textContent = m.model_5.rmse.toFixed(2) + " kW";
    }

    renderMetricsChart(m);
  } catch (err) {
    console.error("Failed to load summary:", err);
  }
}

// 5-Model Metrics Comparison Chart
function renderMetricsChart(models) {
  const ctx = document.getElementById("metricsComparisonChart");
  if (!ctx) return;

  const labels = [
    "M1: Linear (Baseline)",
    "M2: ANN (MLP)",
    "M3: LSTM (PyTorch)",
    "M4: Random Forest 👑",
    "M5: HistGBDT ⚡"
  ];

  const rmseData = [
    models.model_1 ? models.model_1.rmse : 1.36,
    models.model_2 ? models.model_2.rmse : 1.18,
    models.model_3 ? models.model_3.rmse : 1.51,
    models.model_4 ? models.model_4.rmse : 0.78,
    models.model_5 ? models.model_5.rmse : 0.88
  ];

  const maeData = [
    models.model_1 ? models.model_1.mae : 0.73,
    models.model_2 ? models.model_2.mae : 0.65,
    models.model_3 ? models.model_3.mae : 0.81,
    models.model_4 ? models.model_4.mae : 0.40,
    models.model_5 ? models.model_5.mae : 0.48
  ];

  if (metricsChart) metricsChart.destroy();

  metricsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "RMSE (kW) - ยิ่งต่ำยิ่งแม่นยำ",
          data: rmseData,
          backgroundColor: [
            "rgba(217, 119, 6, 0.85)",   // Amber Linear
            "rgba(147, 51, 234, 0.85)",  // Purple ANN
            "rgba(79, 70, 229, 0.85)",   // Indigo LSTM
            "rgba(2, 132, 199, 0.95)",   // Sky RF (Winner)
            "rgba(16, 185, 129, 0.85)"   // Emerald HistGBDT
          ],
          borderColor: ["#d97706", "#9333ea", "#4f46e5", "#0284c7", "#10b981"],
          borderWidth: 1.5,
          borderRadius: 8
        },
        {
          label: "MAE (kW) - ค่าเฉลี่ยความคลาดเคลื่อนสมบูรณ์",
          data: maeData,
          backgroundColor: [
            "rgba(217, 119, 6, 0.25)",
            "rgba(147, 51, 234, 0.25)",
            "rgba(79, 70, 229, 0.25)",
            "rgba(2, 132, 199, 0.35)",
            "rgba(16, 185, 129, 0.25)"
          ],
          borderColor: ["#d97706", "#9333ea", "#4f46e5", "#0284c7", "#10b981"],
          borderWidth: 1.5,
          borderRadius: 8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#475569", font: { size: 12, family: "'Plus Jakarta Sans'" }, boxWidth: 14 }
        },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#38bdf8",
          bodyColor: "#ffffff",
          cornerRadius: 8,
          padding: 10,
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label.split("-")[0].trim()}: ${ctx.raw.toFixed(3)} kW`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#334155", font: { size: 11, weight: "bold", family: "'Plus Jakarta Sans'" } },
          grid: { display: false }
        },
        y: {
          title: { display: true, text: "Error (kW)", color: "#64748b" },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(226, 232, 240, 0.8)" }
        }
      }
    }
  });
}

// ==========================================
// 3. TAB 2: 123 DAYS TOURNAMENT CURVES
// ==========================================
async function loadDates() {
  try {
    const res = await fetch("/api/available_dates");
    const dates = await res.json();
    const sel = document.getElementById("dateSelect");
    const wSel = document.getElementById("weatherDateSelect");
    const valSel = document.getElementById("valDateSelect");
    const ncSel = document.getElementById("nowcastDateSelect");
    if (!sel || !dates.length) return;

    allDatesList = dates;
    sel.innerHTML = "";
    if (wSel) wSel.innerHTML = "";
    if (valSel) valSel.innerHTML = "";
    if (ncSel) ncSel.innerHTML = "";

    let defaultDate = dates[0].date;
    const cloudyDay = dates.find(d => d.cloud_events >= 3);
    if (cloudyDay) defaultDate = cloudyDay.date;
    currentDate = defaultDate;
    currentValDate = defaultDate;
    nowcastSimDate = defaultDate;

    dates.forEach(d => {
      const cloudTag = d.cloud_events > 0 ? ` ⛅ (${d.cloud_events} จุดเมฆ)` : ` ☀️`;
      
      const opt = document.createElement("option");
      opt.value = d.date;
      opt.textContent = `${d.date} ${cloudTag} - ${d.total_kwh} kWh`;
      if (d.date === defaultDate) opt.selected = true;
      sel.appendChild(opt);

      if (valSel) {
        const vOpt = document.createElement("option");
        vOpt.value = d.date;
        vOpt.textContent = `${d.date} ${cloudTag} - ${d.total_kwh} kWh`;
        if (d.date === defaultDate) vOpt.selected = true;
        valSel.appendChild(vOpt);
      }

      if (ncSel) {
        const nOpt = document.createElement("option");
        nOpt.value = d.date;
        nOpt.textContent = `${d.date} ${cloudTag} - ${d.total_kwh} kWh`;
        if (d.date === defaultDate) nOpt.selected = true;
        ncSel.appendChild(nOpt);
      }
    });

    if (wSel) {
      wSel.innerHTML = "";
      // 1. Top option: Today's Live Weather
      const todayOpt = document.createElement("option");
      todayOpt.value = "today";
      todayOpt.textContent = `🔴 วันนี้ล่าสุด (Live Forecast) — ข้อมูลสด Open-Meteo`;
      todayOpt.selected = true;
      wSel.appendChild(todayOpt);

      // 2. Historical 123 Days
      const histGroup = document.createElement("optgroup");
      histGroup.label = "--- ข้อมูลย้อนหลัง 123 วัน (Historical Archive) ---";
      dates.forEach(d => {
        const cloudTag = d.cloud_events > 0 ? ` ⛅ (${d.cloud_events} จุดเมฆ)` : ` ☀️`;
        const wOpt = document.createElement("option");
        wOpt.value = d.date;
        wOpt.textContent = `${d.date} ${cloudTag} - ${d.total_kwh} kWh`;
        histGroup.appendChild(wOpt);
      });
      wSel.appendChild(histGroup);
    }

    loadCurvesForDate(defaultDate);
    loadValidationData(defaultDate, currentValTarget);
  } catch (err) {
    console.error("Failed to load dates:", err);
  }
}

function syncMasterDate(date) {
  if (!date) return;
  const isToday = (date === "today");
  if (!isToday) {
    currentDate = date;
    currentValDate = date;
    nowcastSimDate = date;
  }

  const sel = document.getElementById("dateSelect");
  const wSel = document.getElementById("weatherDateSelect");
  const valSel = document.getElementById("valDateSelect");
  const ncSel = document.getElementById("nowcastDateSelect");

  if (!isToday) {
    if (sel && sel.value !== date) sel.value = date;
    if (valSel && valSel.value !== date) valSel.value = date;
    if (ncSel && ncSel.value !== date) ncSel.value = date;
  }
  if (wSel && wSel.value !== date) wSel.value = date;

  // Determine active tab to reload its view immediately
  const activeTab = document.querySelector(".tab-content:not(.hidden)");
  const activeTabId = activeTab ? activeTab.id : "";

  if (activeTabId === "tab-curves") {
    loadCurvesForDate(currentDate);
  } else if (activeTabId === "tab-future") {
    loadValidationData(currentValDate, currentValTarget, currentValRes);
  } else if (activeTabId === "tab-simulator") {
    loadNowcastSim(nowcastSimDate, nowcastSimHour, nowcastSimTarget);
  } else if (activeTabId === "tab-weather-24h") {
    loadWeather24h(date);
  } else {
    loadCurvesForDate(currentDate);
  }
}

function onWeatherDateChange(val) {
  loadWeather24h(val);
  if (val !== "today") {
    currentDate = val;
  }
}

function syncWeatherDate(date) {
  onWeatherDateChange(date);
}

function changeNowcastDate(date) {
  syncMasterDate(date);
}

function selectTarget(target) {
  currentTarget = target;
  
  ["total", "pv1", "pv2", "pv3", "pv4"].forEach(t => {
    const btn = document.getElementById(`target-btn-${t}`);
    if (btn) {
      if (t === target) {
        btn.className = "target-btn active px-3 py-1.5 text-xs font-bold rounded-lg bg-sky-600 text-white shadow-sm transition-all duration-150";
      } else {
        btn.className = "target-btn px-3 py-1.5 text-xs font-medium rounded-lg text-slate-600 hover:text-slate-900 transition-all duration-150";
      }
    }
  });

  const badge = document.getElementById("targetBadge");
  if (badge) {
    if (target === 'total') badge.textContent = "มุมมองปัจจุบัน: รวมทั้งระบบ (18.72 kWp)";
    else badge.textContent = `มุมมองปัจจุบัน: สตริง ${target.toUpperCase()} (4.68 kWp)`;
  }

  loadCurvesForDate(currentDate, target);
}

async function loadCurvesForDate(date, target = currentTarget) {
  currentDate = date;
  currentTarget = target;
  const wSel = document.getElementById("weatherDateSelect");
  if (wSel && wSel.value !== date) wSel.value = date;

  const badge = document.getElementById("dateBadge");
  if (badge) badge.textContent = `กำลังประมวลผล ${date}...`;

  try {
    const res = await fetch(`/api/models/curves?date=${date}&target=${target}`);
    const data = await res.json();
    if (data.status !== "success") {
      if (badge) badge.textContent = "ไม่พบข้อมูล";
      return;
    }

    renderDailyWinnerCard(data.daily_evaluation, date);

    // Day stats
    const totalKwh = (data.actual_power.reduce((a, b) => a + b, 0) * (5 / 60)).toFixed(1);
    const peakIrr = data.irradiance.length ? Math.max(...data.irradiance).toFixed(0) : "0";
    const avgCloud = data.cloud_cover.length ? (data.cloud_cover.reduce((a, b) => a + b, 0) / data.cloud_cover.length).toFixed(0) : "0";
    const cloudPoints = data.cloud_cover ? data.cloud_cover.filter(c => c > 50).length : 0;

    document.getElementById("day-actual-kwh").textContent = `${totalKwh} kWh`;
    document.getElementById("day-peak-irr").textContent = `${peakIrr} W/m²`;
    document.getElementById("day-avg-cloud").textContent = `${avgCloud}%`;
    document.getElementById("day-cloud-events").textContent = cloudPoints > 0 ? `พบเมฆหนาแน่น ${cloudPoints} ช่วง` : "ท้องฟ้าปลอดโปร่งทั้งวัน";

    if (badge) {
      badge.textContent = cloudPoints > 0 ? `⛅ มีเมฆบัง (${cloudPoints} ช่วง)` : `☀️ แดดจัด ฟ้าเปิด`;
      badge.className = cloudPoints > 0 
        ? "text-xs px-2.5 py-1 rounded-full bg-sky-50 text-sky-700 border border-sky-200 font-mono-nums font-semibold"
        : "text-xs px-2.5 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-mono-nums font-semibold";
    }

    renderCurvesChart(data);
  } catch (err) {
    console.error("Failed to load curves:", err);
    if (badge) badge.textContent = "เกิดข้อผิดพลาด";
  }
}

function renderDailyWinnerCard(ev, date) {
  const title = document.getElementById("day-winner-title");
  const p1 = document.getElementById("day-m1-pill");
  const p2 = document.getElementById("day-m2-pill");
  const p3 = document.getElementById("day-m3-pill");
  const p4 = document.getElementById("day-m4-pill");
  const p5 = document.getElementById("day-m5-pill");

  if (!ev) return;

  const winName = ev.winner_name || "Random Forest";
  const imp = ev.improvement_pct || 0;
  const bestRmse = ev.best_rmse ? ev.best_rmse.toFixed(3) : "-";
  const targetText = currentTarget === 'total' ? 'ทั้งระบบ' : currentTarget.toUpperCase();

  if (title) {
    title.innerHTML = `
      <span class="font-extrabold text-slate-900">${winName}</span>
      <span class="text-xs font-semibold text-slate-500">(${targetText})</span>
      <span class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 font-extrabold ml-1">
        RMSE ${bestRmse} kW ${imp > 0 ? `(+${imp}% เหนือกว่า Linear)` : 'ชนะเลิศ'}
      </span>
    `;
  }

  const setPill = (el, id, label, colorClass, data) => {
    if (!el || !data) return;
    const isWinner = ev.winner_id === id;
    const winnerBorder = isWinner ? "ring-2 ring-emerald-500 font-bold bg-white shadow-sm" : "";
    const crown = isWinner ? " 👑" : "";
    el.className = `px-2.5 py-1.5 rounded-xl border text-[11px] font-mono-nums ${colorClass} ${winnerBorder}`;
    el.innerHTML = `<span class="font-sans font-semibold mr-1">${label}${crown}:</span> RMSE <b>${data.rmse.toFixed(3)}</b> | R² ${(data.r2 * 100).toFixed(1)}%`;
  };

  setPill(p1, "m1", "M1 Linear", "bg-amber-50 border-amber-200 text-amber-900", ev.m1);
  setPill(p2, "m2", "M2 ANN", "bg-purple-50 border-purple-200 text-purple-900", ev.m2);
  setPill(p3, "m3", "M3 LSTM", "bg-indigo-50 border-indigo-200 text-indigo-900", ev.m3);
  setPill(p4, "m4", "M4 RF", "bg-sky-50 border-sky-300 text-sky-900", ev.m4);
  setPill(p5, "m5", "M5 GBDT", "bg-emerald-50 border-emerald-300 text-emerald-900", ev.m5);
}

function renderCurvesChart(data) {
  const ctx = document.getElementById("curvesComparisonChart");
  if (!ctx) return;

  if (curvesChart) curvesChart.destroy();

  curvesChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: `Actual Power (${data.target ? data.target.toUpperCase() : 'จริง'} kW)`,
          data: data.actual_power,
          borderColor: "#0f172a",
          backgroundColor: "rgba(15, 23, 42, 0.05)",
          borderWidth: 3,
          pointRadius: 1,
          pointHoverRadius: 6,
          tension: 0.2,
          yAxisID: "y"
        },
        {
          label: "Model 1: Baseline (Linear)",
          data: data.pred_m1,
          borderColor: "#d97706",
          borderWidth: 1.8,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y"
        },
        {
          label: "Model 2: ANN (MLP)",
          data: data.pred_m2,
          borderColor: "#9333ea",
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y"
        },
        {
          label: "Model 3: LSTM",
          data: data.pred_m3,
          borderColor: "#4f46e5",
          borderWidth: 1.8,
          borderDash: [2, 2],
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y"
        },
        {
          label: "Model 4: Random Forest 👑",
          data: data.pred_m4,
          borderColor: "#0284c7",
          backgroundColor: "rgba(2, 132, 199, 0.06)",
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y"
        },
        {
          label: "Model 5: HistGBDT ⚡",
          data: data.pred_m5,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.06)",
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y"
        },
        {
          label: "Solar Irradiance (W/m²)",
          data: data.irradiance,
          borderColor: "rgba(245, 158, 11, 0.4)",
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointRadius: 0,
          tension: 0.2,
          yAxisID: "y1",
          hidden: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#38bdf8",
          bodyColor: "#ffffff",
          cornerRadius: 8,
          padding: 10,
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.yAxisID === "y") {
                return ` ${ctx.dataset.label.split("(")[0].trim()}: ${ctx.raw} kW`;
              } else {
                return ` Irradiance: ${ctx.raw} W/m²`;
              }
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#64748b",
            maxTicksLimit: window.innerWidth < 640 ? 8 : 16,
            font: { size: 10, family: "'Plus Jakarta Sans'" }
          },
          grid: { color: "rgba(226, 232, 240, 0.7)" }
        },
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Active Power (kW)", color: "#64748b", font: { size: 11, family: "'Plus Jakarta Sans'" } },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(226, 232, 240, 0.8)" },
          min: 0
        },
        y1: {
          type: "linear",
          display: false,
          position: "right",
          grid: { drawOnChartArea: false },
          min: 0,
          max: 1200
        }
      }
    }
  });
}

function toggleDataset(index, btn) {
  if (!curvesChart) return;
  const isVisible = curvesChart.isDatasetVisible(index);
  curvesChart.setDatasetVisibility(index, !isVisible);
  curvesChart.update();

  if (!isVisible) {
    btn.classList.add("active");
  } else {
    btn.classList.remove("active");
  }
}

// ==========================================
// 4. TAB 3: 100% OPEN-METEO FUTURE FORECAST
// ==========================================
async function loadFutureForecast(target = currentFutureTarget, forceReload = false) {
  currentFutureTarget = target;
  try {
    const res = await fetch(`/api/forecast/future?target=${target}&reload=${forceReload}`);
    const data = await res.json();
    if (data.status !== "success") return;

    futureForecastData = data;

    // A. 3 Hero KPI Cards
    const summaries = data.daily_summaries || [];
    if (summaries.length >= 1) {
      const today = summaries[0];
      document.getElementById("fc-today-condition").textContent = today.weather_condition;
      document.getElementById("fc-today-kwh").textContent = `${today.estimated_kwh_rf} kWh`;
      document.getElementById("fc-today-peak").textContent = `${today.peak_kw} kW`;
      document.getElementById("fc-today-time").textContent = `${today.peak_time} น.`;
      document.getElementById("fc-today-date").textContent = `วันที่ ${today.date}`;
    }

    if (summaries.length >= 2) {
      const tmrw = summaries[1];
      document.getElementById("fc-tmrw-condition").textContent = tmrw.weather_condition;
      document.getElementById("fc-tmrw-kwh").textContent = `${tmrw.estimated_kwh_rf} kWh`;
      document.getElementById("fc-tmrw-peak").textContent = `${tmrw.peak_kw} kW`;
      document.getElementById("fc-tmrw-time").textContent = `${tmrw.peak_time} น.`;
      document.getElementById("fc-tmrw-date").textContent = `วันที่ ${tmrw.date}`;
    }

    if (summaries.length > 0) {
      const total7d = summaries.reduce((acc, d) => acc + d.estimated_kwh_rf, 0);
      const avg7d = total7d / summaries.length;
      const savingBaht = Math.round(total7d * 4.40);

      document.getElementById("fc-7d-total-kwh").textContent = `${total7d.toFixed(1)} kWh`;
      document.getElementById("fc-7d-avg-kwh").textContent = `${avg7d.toFixed(1)} kWh/วัน`;
      document.getElementById("fc-7d-saving").textContent = `~ ${savingBaht.toLocaleString()} บาท`;
    }

    // B. Chart & Daily Cards
    renderFutureForecastChart(data, currentForecastRange);
    renderFutureDailyCards(summaries);

    // C. Last Updated Badges
    const badge = document.getElementById("fc-last-updated-badge");
    if (badge) badge.textContent = data.last_updated_time || data.last_updated || "--:-- น.";

    const cardUpdated = document.getElementById("fc-cards-updated-time");
    if (cardUpdated) cardUpdated.textContent = data.last_updated_th || data.last_updated || "--";

  } catch (err) {
    console.error("Failed to load future forecast:", err);
  }
}

function changeFutureTarget(target) {
  currentFutureTarget = target;
  ["total", "pv1", "pv2", "pv3", "pv4"].forEach(t => {
    const btn = document.getElementById(`future-tgt-${t}`);
    if (btn) {
      if (t === target) {
        btn.className = "px-2.5 py-1 rounded-lg bg-sky-600 text-white font-bold shadow-sm";
      } else {
        btn.className = "px-2.5 py-1 rounded-lg text-slate-600 font-medium hover:text-slate-900";
      }
    }
  });

  loadFutureForecast(target, false);
}

function reloadFutureForecast(force = true) {
  loadFutureForecast(currentFutureTarget, force);
}

function setForecastRange(hours) {
  currentForecastRange = hours;
  [24, 48, 168].forEach(h => {
    const btn = document.getElementById(`fc-range-${h}`);
    if (btn) {
      if (h === hours) {
        btn.className = "px-3 py-1 rounded-lg bg-white text-slate-900 font-bold shadow-sm";
      } else {
        btn.className = "px-3 py-1 rounded-lg text-slate-600 font-medium hover:text-slate-900";
      }
    }
  });

  if (futureForecastData) {
    renderFutureForecastChart(futureForecastData, hours);
  }
}

function renderFutureForecastChart(data, hours = 168) {
  const ctx = document.getElementById("futureForecastChart");
  if (!ctx || !data.hourly_curves) return;

  const hc = data.hourly_curves;
  const limit = Math.min(hours, hc.times.length);

  const labels = hc.times.slice(0, limit);
  const pRf = hc.pred_rf.slice(0, limit);
  const pGbdt = hc.pred_gbdt.slice(0, limit);
  const pLinear = hc.pred_linear.slice(0, limit);
  const ghi = hc.ghi.slice(0, limit);
  const rain = hc.rain_prob.slice(0, limit);
  const pNowcast = (hc.pred_nowcast || []).slice(0, limit);

  if (futureForecastChart) futureForecastChart.destroy();

  futureForecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: `Random Forest (แชมป์โมเดล - kW)`,
          data: pRf,
          borderColor: "#0284c7",
          backgroundColor: "rgba(2, 132, 199, 0.12)",
          borderWidth: 2.5,
          fill: true,
          pointRadius: limit <= 48 ? 2 : 0,
          pointHoverRadius: 6,
          tension: 0.25,
          yAxisID: "y"
        },
        {
          label: "HistGBDT (Weather-Physics - kW)",
          data: pGbdt,
          borderColor: "#10b981",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          yAxisID: "y"
        },
        {
          label: "Linear (Baseline - kW)",
          data: pLinear,
          borderColor: "#d97706",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.25,
          yAxisID: "y",
          hidden: true
        },
        {
          label: "Solar Irradiance GHI (W/m²)",
          data: ghi,
          borderColor: "rgba(245, 158, 11, 0.6)",
          borderWidth: 1.5,
          borderDash: [3, 3],
          pointRadius: 0,
          tension: 0.25,
          yAxisID: "y1"
        },
        {
          label: "โอกาสฝนตก Rain (%)",
          data: rain,
          borderColor: "rgba(148, 163, 184, 0.7)",
          backgroundColor: "rgba(148, 163, 184, 0.15)",
          borderWidth: 1,
          pointRadius: 0,
          fill: true,
          tension: 0.25,
          yAxisID: "y2",
          hidden: true
        },
        {
          label: "1-Hour Nowcast (Sun-Angle - kW)",
          data: pNowcast,
          borderColor: "#6366f1",
          backgroundColor: "rgba(99, 102, 241, 0.08)",
          borderWidth: 2.2,
          borderDash: [5, 3],
          pointRadius: limit <= 48 ? 2 : 0,
          pointHoverRadius: 6,
          tension: 0.25,
          yAxisID: "y"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#38bdf8",
          bodyColor: "#ffffff",
          cornerRadius: 8,
          padding: 10,
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.yAxisID === "y") {
                return ` ${ctx.dataset.label.split("(")[0].trim()}: ${ctx.raw} kW`;
              } else if (ctx.dataset.yAxisID === "y1") {
                return ` GHI แดด: ${ctx.raw} W/m²`;
              } else {
                return ` โอกาสฝน: ${ctx.raw}%`;
              }
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#64748b",
            maxTicksLimit: limit <= 24 ? 12 : (limit <= 48 ? 16 : 24),
            font: { size: 10, family: "'Plus Jakarta Sans'" }
          },
          grid: { color: "rgba(226, 232, 240, 0.7)" }
        },
        y: {
          type: "linear",
          position: "left",
          title: { display: true, text: "กำลังไฟฟ้าคาดการณ์ (kW)", color: "#0284c7", font: { size: 11 } },
          ticks: { color: "#0284c7" },
          grid: { color: "rgba(226, 232, 240, 0.8)" },
          min: 0
        },
        y1: {
          type: "linear",
          position: "right",
          title: { display: true, text: "GHI (W/m²)", color: "#f59e0b", font: { size: 10 } },
          ticks: { color: "#f59e0b" },
          grid: { drawOnChartArea: false },
          min: 0,
          max: 1200
        },
        y2: {
          type: "linear",
          position: "right",
          display: false,
          min: 0,
          max: 100
        }
      }
    }
  });
}

function renderFutureDailyCards(summaries) {
  const container = document.getElementById("futureDailyCardsContainer");
  if (!container) return;

  const getEmoji = (icon, emoji) => {
    if (emoji) return emoji;
    if (icon === "thunderstorm" || icon === "thunderstorm-heavy") return "⛈️";
    if (icon === "drizzle") return "🌦️";
    if (icon === "rain" || icon === "rain-heavy") return "🌧️";
    if (icon === "sun") return "☀️";
    if (icon === "sun-small-cloud") return "🌤️";
    if (icon === "sun-cloud") return "⛅";
    if (icon === "cloud") return "☁️";
    if (icon === "fog") return "🌫️";
    if (icon === "snow") return "❄️";
    return "☀️";
  };

  const dayNames = ["อาทิตย์", "จันทร์", "อังคาร", "พุธ", "พฤหัสฯ", "ศุกร์", "เสาร์"];

  container.innerHTML = summaries.map((s, idx) => {
    const d = new Date(s.date);
    const dayName = dayNames[d.getDay()] || "";
    const isToday = idx === 0;

    return `
      <div class="flex flex-col justify-between p-3.5 rounded-2xl border ${isToday ? 'border-amber-300 bg-amber-50/50 ring-2 ring-amber-400/50' : 'border-slate-200 bg-white'} card-shadow transition-all hover:shadow-md">
        <div>
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-slate-800">${dayName}</span>
            <span class="text-[10px] font-mono-nums ${isToday ? 'text-amber-700 font-bold' : 'text-slate-400'}">${s.date.slice(5)}</span>
          </div>
          <div class="flex items-center gap-2 my-2">
            <span class="text-2xl">${getEmoji(s.weather_icon, s.weather_emoji)}</span>
            <div class="text-[11px] font-medium text-slate-600 line-clamp-1">${s.weather_condition}</div>
          </div>
          <div class="mt-2 pt-2 border-t border-slate-100">
            <div class="text-[10px] text-slate-400 font-medium">พลังงานคาดการณ์</div>
            <div class="text-lg font-extrabold font-mono-nums text-sky-600">${s.estimated_kwh_rf} <span class="text-xs font-normal">kWh</span></div>
          </div>
        </div>
        <div class="mt-3 pt-2 border-t border-slate-100 grid grid-cols-2 gap-1 text-[10px] text-slate-500 font-mono-nums">
          <div>Peak: <b class="text-slate-800">${s.peak_kw} kW</b></div>
          <div class="text-right">ฝน: <b class="${s.rain_prob_max > 40 ? 'text-rose-600' : 'text-slate-600'}">${s.rain_prob_max}%</b></div>
        </div>
      </div>
    `;
  }).join("");
}

// Future Forecast Dataset Visibility Toggle
function toggleFutureDataset(index, btn) {
  if (!futureForecastChart) return;
  const isVisible = futureForecastChart.isDatasetVisible(index);
  futureForecastChart.setDatasetVisibility(index, !isVisible);
  futureForecastChart.update();

  if (!isVisible) {
    btn.classList.add("active");
    if (index === 0) btn.className = "curve-toggle active px-2.5 py-1 rounded-lg border border-sky-400 bg-sky-600 text-white font-bold text-[11px] shadow-sm transition-all";
    else if (index === 1) btn.className = "curve-toggle active px-2.5 py-1 rounded-lg border border-emerald-300 bg-emerald-50 text-emerald-800 font-bold text-[11px] transition-all";
    else if (index === 2) btn.className = "curve-toggle active px-2.5 py-1 rounded-lg border border-amber-300 bg-amber-50 text-amber-800 font-semibold text-[11px] transition-all";
    else if (index === 3) btn.className = "curve-toggle active px-2.5 py-1 rounded-lg border border-amber-300 bg-amber-50 text-amber-800 font-semibold text-[11px] transition-all";
    else if (index === 4) btn.className = "curve-toggle active px-2.5 py-1 rounded-lg border border-slate-400 bg-slate-100 text-slate-800 font-bold text-[11px] transition-all";
    else if (index === 5) btn.className = "curve-toggle active px-2.5 py-1 rounded-lg border border-indigo-300 bg-indigo-50 text-indigo-800 font-bold text-[11px] transition-all";
  } else {
    btn.classList.remove("active");
    btn.className = "curve-toggle px-2.5 py-1 rounded-lg border border-slate-300 bg-white text-slate-500 font-medium text-[11px] transition-all";
  }
}

// Validation Dataset Visibility Toggle
function toggleValidationDataset(index, btn) {
  if (!validationChart) return;
  const isVisible = validationChart.isDatasetVisible(index);
  validationChart.setDatasetVisibility(index, !isVisible);
  validationChart.update();

  if (!isVisible) {
    btn.classList.add("active");
    if (index === 0) btn.className = "curve-toggle active px-2 py-1 rounded-lg border border-slate-400 bg-slate-900 text-white font-bold text-[11px] shadow-xs transition-all flex items-center gap-1.5 cursor-pointer";
    else if (index === 1) btn.className = "curve-toggle active px-2 py-1 rounded-lg border border-emerald-300 bg-emerald-50 text-emerald-800 font-bold text-[11px] shadow-xs transition-all flex items-center gap-1.5 cursor-pointer";
    else if (index === 2) btn.className = "curve-toggle active px-2 py-1 rounded-lg border border-amber-300 bg-amber-50 text-amber-800 font-bold text-[11px] shadow-xs transition-all flex items-center gap-1.5 cursor-pointer";
    else if (index === 3) btn.className = "curve-toggle active px-2 py-1 rounded-lg border border-indigo-300 bg-indigo-50 text-indigo-800 font-bold text-[11px] shadow-xs transition-all flex items-center gap-1.5 cursor-pointer";
  } else {
    btn.classList.remove("active");
    btn.className = "curve-toggle px-2 py-1 rounded-lg border border-slate-200 bg-white text-slate-400 font-medium text-[11px] line-through transition-all flex items-center gap-1.5 opacity-60 cursor-pointer";
  }
}

// ==========================================
// 4.1 HISTORICAL VALIDATION & CALIBRATION LAB
// ==========================================
async function loadValidationData(date = currentValDate, target = currentValTarget, res = currentValRes) {
  currentValDate = date;
  currentValTarget = target;
  currentValRes = res;
  try {
    const response = await fetch(`/api/forecast/validation?date=${date}&target=${target}&res=${res}`);
    const data = await response.json();
    if (data.status !== "success") return;

    const m = data.metrics;

    // Winner Banner
    const wEmoji = document.getElementById("val-winner-emoji");
    const wBadge = document.getElementById("val-winner-badge");
    const wReason = document.getElementById("val-winner-reason");
    const wStat = document.getElementById("val-winner-stat");

    if (wBadge) wBadge.textContent = m.winner_badge || "ผลการตัดสินความแม่นยำ";
    if (wReason) wReason.textContent = m.winner_reason || "--";
    if (wEmoji) {
      if (m.winner === 'sensor') wEmoji.textContent = "🏆";
      else if (m.winner === 'calibrated') wEmoji.textContent = "⭐";
      else wEmoji.textContent = "🌤️";
    }
    if (wStat) {
      if (m.winner === 'sensor') {
        wStat.textContent = `เซนเซอร์แม่นกว่า +${m.sensor_lead_pct}%`;
        wStat.className = "px-3 py-1 rounded-lg bg-emerald-100 text-emerald-800 border border-emerald-300 shadow-xs";
      } else {
        wStat.textContent = `Rolling แม่นขึ้น +${m.improvement_pct}%`;
        wStat.className = "px-3 py-1 rounded-lg bg-indigo-100 text-indigo-800 border border-indigo-300 shadow-xs";
      }
    }

    const sIrrEl = document.getElementById("val-s-irr");
    const omIrrEl = document.getElementById("val-om-irr");
    const biasEl = document.getElementById("val-irr-bias");
    if (sIrrEl) sIrrEl.textContent = `${m.mean_sensor_irr} W/m²`;
    if (omIrrEl) omIrrEl.textContent = `(OM: ${m.mean_om_irr})`;
    if (biasEl) biasEl.textContent = `ส่วนต่างแดดเฉลี่ย: ${m.irr_diff_pct > 0 ? '+' : ''}${m.irr_diff_pct}% (ดาวเทียมเทียบหัววัด)`;

    const sRmseEl = document.getElementById("val-sensor-rmse");
    const sR2El = document.getElementById("val-sensor-r2");
    if (sRmseEl) sRmseEl.textContent = `${m.sensor.rmse} kW`;
    if (sR2El) sR2El.textContent = `R²: ${(m.sensor.r2 * 100).toFixed(1)}% | หัวเซนเซอร์จริง`;

    const rawRmseEl = document.getElementById("val-raw-rmse");
    const rawR2El = document.getElementById("val-raw-r2");
    if (rawRmseEl) rawRmseEl.textContent = `${m.raw_openmeteo.rmse} kW`;
    if (rawR2El) rawR2El.textContent = `R²: ${(m.raw_openmeteo.r2 * 100).toFixed(1)}% | คลาดเคลื่อนจากเมฆเฉพาะจุด`;

    const calRmseEl = document.getElementById("val-cal-rmse");
    const calR2El = document.getElementById("val-cal-r2");
    const impBadge = document.getElementById("val-imp-badge");
    if (calRmseEl) calRmseEl.textContent = `${m.calibrated_rolling.rmse} kW`;
    if (calR2El) calR2El.textContent = `R²: ${(m.calibrated_rolling.r2 * 100).toFixed(1)}% | ดึงค่าความแม่นยำกลับมา`;
    if (impBadge) impBadge.textContent = `แม่นขึ้น +${m.improvement_pct}%`;

    renderValidationChart(data);
  } catch (err) {
    console.error("Failed to load validation data:", err);
  }
}

function setValidationResolution(res) {
  currentValRes = res;
  const btn5m = document.getElementById("val-res-5m");
  const btn1h = document.getElementById("val-res-1h");
  if (res === "5m") {
    if (btn5m) btn5m.className = "px-2.5 py-1 rounded-lg bg-indigo-600 text-white font-bold shadow-sm transition-all";
    if (btn1h) btn1h.className = "px-2.5 py-1 rounded-lg text-slate-600 font-medium hover:text-slate-900 transition-all";
  } else {
    if (btn1h) btn1h.className = "px-2.5 py-1 rounded-lg bg-indigo-600 text-white font-bold shadow-sm transition-all";
    if (btn5m) btn5m.className = "px-2.5 py-1 rounded-lg text-slate-600 font-medium hover:text-slate-900 transition-all";
  }
  loadValidationData(currentValDate, currentValTarget, res);
}

function changeValTarget(target) {
  currentValTarget = target;
  ["total", "pv1", "pv2", "pv3", "pv4"].forEach(t => {
    const btn = document.getElementById(`val-tgt-${t}`);
    if (btn) {
      if (t === target) {
        btn.className = "px-2.5 py-1 rounded-lg bg-indigo-600 text-white font-bold shadow-sm";
      } else {
        btn.className = "px-2.5 py-1 rounded-lg text-slate-600 font-medium hover:text-slate-900";
      }
    }
  });
  loadValidationData(currentValDate, target, currentValRes);
}

function renderValidationChart(data) {
  const ctx = document.getElementById("validationComparisonChart");
  if (!ctx) return;

  if (validationChart) validationChart.destroy();

  const is1h = data.resolution === "1h";
  const ptRadius = is1h ? 3.5 : 0;
  const ptHover = is1h ? 6 : 4;

  validationChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels || data.hours,
      datasets: [
        {
          label: "ค่าผลิตจริง Actual Power (kW)",
          data: data.actual_power,
          borderColor: "#0f172a",
          backgroundColor: "rgba(15, 23, 42, 0.05)",
          borderWidth: 2.5,
          pointRadius: is1h ? 4 : 1,
          pointHoverRadius: 6,
          tension: is1h ? 0.3 : 0.2,
          yAxisID: "y"
        },
        {
          label: "ทำนายจากเซนเซอร์หลังคา (kW)",
          data: data.pred_sensor,
          borderColor: "#10b981",
          borderWidth: 2,
          pointRadius: ptRadius,
          pointHoverRadius: ptHover,
          tension: is1h ? 0.3 : 0.2,
          yAxisID: "y"
        },
        {
          label: "ทำนายจาก Open-Meteo ดิบ (kW)",
          data: data.pred_openmeteo,
          borderColor: "#f59e0b",
          borderWidth: 1.8,
          borderDash: [4, 4],
          pointRadius: ptRadius,
          pointHoverRadius: ptHover,
          tension: is1h ? 0.3 : 0.25,
          yAxisID: "y"
        },
        {
          label: "ปรับจูน Rolling 1 ชม. ⭐ (kW)",
          data: data.pred_calibrated,
          borderColor: "#4f46e5",
          backgroundColor: "rgba(79, 70, 229, 0.08)",
          fill: true,
          borderWidth: 2,
          pointRadius: ptRadius,
          pointHoverRadius: ptHover,
          tension: is1h ? 0.3 : 0.25,
          yAxisID: "y"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#818cf8",
          bodyColor: "#ffffff",
          cornerRadius: 8,
          padding: 10,
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label.split("(")[0].trim()}: ${ctx.raw} kW`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#64748b", maxTicksLimit: 16, font: { size: 10, family: "'Plus Jakarta Sans'" } },
          grid: { color: "rgba(226, 232, 240, 0.7)" }
        },
        y: {
          title: { display: true, text: "Active Power (kW)", color: "#4f46e5", font: { size: 11 } },
          ticks: { color: "#4f46e5" },
          grid: { color: "rgba(226, 232, 240, 0.8)" },
          min: 0
        }
      }
    }
  });

  // Re-apply visibility state if user previously toggled off any dataset
  [0, 1, 2, 3].forEach(idx => {
    const btn = document.getElementById(`val-toggle-${idx}`);
    if (btn && !btn.classList.contains("active")) {
      validationChart.setDatasetVisibility(idx, false);
    }
  });
  validationChart.update();
}

// ==========================================
// 5. TAB 4: CLOUD ANALYSIS (FALSE ALARM FILTER)
// ==========================================
async function loadCloudAnalysis() {
  try {
    const res = await fetch("/api/models/cloud_analysis");
    const data = await res.json();
    renderEventPieChart(data);
  } catch (err) {
    console.error("Failed to load cloud analysis:", err);
  }
}

function renderEventPieChart(data) {
  const ctx = document.getElementById("eventPieChart");
  if (!ctx) return;

  if (eventChart) eventChart.destroy();

  eventChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: [
        `การทำงานปกติ (${data.normal_count?.toLocaleString()} จุด)`,
        `เมฆบังชั่วคราว (คัดกรอง ${data.cloud_shading_count?.toLocaleString()} จุด)`,
        `พบความผิดปกติจริง (${data.real_fault_count?.toLocaleString()} จุด)`
      ],
      datasets: [
        {
          data: [data.normal_count || 15000, data.cloud_shading_count || 1100, data.real_fault_count || 160],
          backgroundColor: ["#10b981", "#0284c7", "#f43f5e"],
          borderColor: "#ffffff",
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#475569", font: { size: 11, family: "'Plus Jakarta Sans'" }, boxWidth: 12, padding: 12 }
        }
      },
      cutout: "68%"
    }
  });
}

// ==========================================
// 6. TAB 5: WHAT-IF SIMULATOR
// ==========================================
let simDebounce = null;
async function runSimulation() {
  clearTimeout(simDebounce);
  simDebounce = setTimeout(async () => {
    const irrEl = document.getElementById("sim-irr");
    const cloudEl = document.getElementById("sim-cloud");
    const tempEl = document.getElementById("sim-temp");
    if (!irrEl || !cloudEl || !tempEl) return;

    const irr = parseFloat(irrEl.value);
    const cloud = parseFloat(cloudEl.value);
    const temp = parseFloat(tempEl.value);

    document.getElementById("sim-irr-val").textContent = `${irr} W/m²`;
    document.getElementById("sim-cloud-val").textContent = `${cloud}%`;
    document.getElementById("sim-temp-val").textContent = `${temp} °C`;

    try {
      const res = await fetch("/api/models/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ irradiance: irr, cloud_cover: cloud, ambient_temp: temp })
      });
      const data = await res.json();

      document.getElementById("sim-out-m1").textContent = `${data.predictions.model_1.toFixed(2)} kW`;
      document.getElementById("sim-out-m2").textContent = `${data.predictions.model_2.toFixed(2)} kW`;
      document.getElementById("sim-out-m3").textContent = `${data.predictions.model_3.toFixed(2)} kW`;

      const vBox = document.getElementById("sim-verdict-box");
      const vText = document.getElementById("sim-verdict-text");

      vBox.className = `p-4 rounded-xl border verdict-${data.verdict_badge}`;
      vText.textContent = data.verdict;
    } catch (err) {
      console.error("Simulation failed:", err);
    }
  }, 100);
}

// ==========================================
// 7. TAB 6: 24-HOUR WEATHER STATION
// ==========================================
async function loadWeather24h(date) {
  try {
    const res = await fetch(`/api/weather/hourly?date=${date}`);
    const data = await res.json();
    if (data.status !== "success") return;

    const s = data.summary;
    const elPeak = document.getElementById("w-peak-ghi");
    if (elPeak) elPeak.textContent = `${s.peak_ghi} W/m²`;
    const elEnergy = document.getElementById("w-total-energy");
    if (elEnergy) elEnergy.textContent = `${s.total_solar_kwh_m2} kWh/m²`;
    const elCloud = document.getElementById("w-avg-cloud");
    if (elCloud) elCloud.textContent = `${s.avg_cloud}%`;
    const elTemp = document.getElementById("w-temp-range");
    if (elTemp) elTemp.textContent = `${s.min_temp} - ${s.max_temp} °C`;

    const cloudDesc = document.getElementById("w-cloud-desc");
    if (cloudDesc) {
      if (s.avg_cloud > 70) cloudDesc.textContent = "☁️ มีเมฆมากเกือบทั้งวัน";
      else if (s.avg_cloud > 30) cloudDesc.textContent = "⛅ ท้องฟ้ามีเมฆบางส่วน";
      else cloudDesc.textContent = "☀️ ท้องฟ้าโปร่ง แดดจัด";
    }

    // Update badge with live/today indicator
    const badge = document.getElementById("weatherDateBadge");
    if (badge) {
      if (data.is_today) {
        badge.className = "text-xs px-2.5 py-1 rounded-full bg-rose-50 text-rose-700 border border-rose-200 font-semibold flex items-center gap-1.5 shadow-xs";
        badge.innerHTML = `<span class="size-2 rounded-full bg-rose-500 animate-pulse"></span> ข้อมูลสภาพอากาศสดวันนี้ (${data.date}) · ปัจจุบัน ~${data.current_hour}:00 น.`;
      } else {
        badge.className = "text-xs px-2.5 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-mono-nums font-semibold";
        badge.textContent = `24 ชั่วโมง (${data.date})`;
      }
    }

    const wSel = document.getElementById("weatherDateSelect");
    if (wSel) {
      if (data.is_today && (date === "today" || wSel.value === "today")) {
        wSel.value = "today";
      } else if (wSel.value !== data.date) {
        wSel.value = data.date;
      }
    }

    renderWeather24hChart(data);
    renderHourlyChips(data.hourly_cards || [], data);
    renderHourlyTable(data.hourly_cards || [], data);

  } catch (err) {
    console.error("Failed to load 24h weather:", err);
  }
}

function renderWeather24hChart(data) {
  const ctx = document.getElementById("weather24hChart");
  if (!ctx) return;

  if (weather24hChart) weather24hChart.destroy();

  weather24hChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.hours,
      datasets: [
        {
          label: "GHI (Global Horizontal Irradiance W/m²)",
          data: data.ghi,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.15)",
          borderWidth: 2.5,
          fill: true,
          tension: 0.3,
          yAxisID: "y"
        },
        {
          label: "DNI (Direct Normal Irradiance W/m²)",
          data: data.dni,
          borderColor: "#ea580c",
          borderDash: [4, 4],
          borderWidth: 1.8,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          yAxisID: "y"
        },
        {
          label: "DHI (Diffuse Horizontal Irradiance W/m²)",
          data: data.dhi,
          borderColor: "#eab308",
          borderDash: [2, 2],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          yAxisID: "y"
        },
        {
          label: "Ambient Temperature (°C)",
          data: data.temperature,
          borderColor: "#ef4444",
          borderWidth: 2,
          pointRadius: 2,
          fill: false,
          tension: 0.3,
          yAxisID: "y1"
        },
        {
          label: "Cloud Cover (%)",
          data: data.cloud_cover,
          borderColor: "#94a3b8",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          fill: false,
          tension: 0.3,
          yAxisID: "y2"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#475569", boxWidth: 12, font: { size: 11, family: "'Plus Jakarta Sans'" } }
        },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#38bdf8",
          bodyColor: "#ffffff",
          cornerRadius: 8,
          padding: 10
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#64748b",
            maxTicksLimit: window.innerWidth < 640 ? 8 : 14,
            font: { size: 10, family: "'Plus Jakarta Sans'" }
          },
          grid: { color: "rgba(226, 232, 240, 0.7)" }
        },
        y: {
          type: "linear",
          position: "left",
          title: { display: true, text: "Irradiance (W/m²)", color: "#f59e0b", font: { size: 11 } },
          ticks: { color: "#f59e0b" },
          grid: { color: "rgba(226, 232, 240, 0.8)" },
          min: 0
        },
        y1: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Temp (°C)", color: "#ef4444", font: { size: 11 } },
          ticks: { color: "#ef4444" },
          grid: { drawOnChartArea: false },
          min: 20,
          max: 42
        },
        y2: {
          type: "linear",
          position: "right",
          display: false,
          min: 0,
          max: 100
        }
      }
    }
  });
}

function renderHourlyChips(cards, weatherData) {
  const container = document.getElementById("hourlyCardsContainer");
  if (!container) return;

  const getEmoji = (icon) => {
    if (icon === 'thunderstorm' || icon === 'thunderstorm-heavy') return '⛈️';
    if (icon === 'drizzle') return '🌦️';
    if (icon === 'sun') return '☀️';
    if (icon === 'sun-small-cloud') return '🌤️';
    if (icon === 'cloud-sun' || icon === 'sun-cloud') return '⛅';
    if (icon === 'cloud') return '☁️';
    if (icon === 'cloud-rain' || icon === 'rain' || icon === 'rain-heavy') return '🌧️';
    if (icon === 'fog') return '🌫️';
    return '🌙';
  };

  container.innerHTML = cards.map(c => {
    const isCurrent = Boolean(c.is_current);
    return `
    <div class="flex flex-col items-center justify-between p-2.5 rounded-xl border ${isCurrent ? 'border-2 border-rose-500 bg-rose-50/80 shadow-md ring-2 ring-rose-300/40 relative' : c.ghi > 600 ? 'border-amber-300 bg-amber-50/50' : 'border-slate-200 bg-slate-50/50'} text-center transition-all hover:shadow-md hover:border-sky-300">
      ${isCurrent ? '<span class="absolute -top-2 px-1.5 py-0.2 rounded-full bg-rose-500 text-white text-[9px] font-black uppercase tracking-wider animate-pulse">ตอนนี้</span>' : ''}
      <span class="text-[11px] font-bold font-mono-nums ${isCurrent ? 'text-rose-700 font-black' : 'text-slate-700'}">${c.hour}</span>
      <span class="text-xl my-1">${c.emoji || getEmoji(c.icon)}</span>
      <span class="text-xs font-bold font-mono-nums ${c.ghi > 0 ? (isCurrent ? 'text-rose-600' : 'text-amber-600') : 'text-slate-400'}">${c.ghi} <span class="text-[9px] font-normal">W/m²</span></span>
      <div class="flex items-center justify-between w-full mt-1.5 pt-1.5 border-t border-slate-200/70 text-[10px] text-slate-500 font-mono-nums">
        <span>${c.temp}°C</span>
        <span>☁️${c.cloud}%</span>
      </div>
    </div>
    `;
  }).join('');
}

function renderHourlyTable(cards, weatherData) {
  const tbody = document.getElementById("hourlyWeatherTableBody");
  if (!tbody) return;

  const getEmoji = (icon) => {
    if (icon === 'thunderstorm' || icon === 'thunderstorm-heavy') return '⛈️';
    if (icon === 'drizzle') return '🌦️';
    if (icon === 'sun') return '☀️';
    if (icon === 'sun-small-cloud') return '🌤️';
    if (icon === 'cloud-sun' || icon === 'sun-cloud') return '⛅';
    if (icon === 'cloud') return '☁️';
    if (icon === 'cloud-rain' || icon === 'rain' || icon === 'rain-heavy') return '🌧️';
    if (icon === 'fog') return '🌫️';
    return '🌙';
  };

  tbody.innerHTML = cards.map(c => {
    const isCurrent = Boolean(c.is_current);
    return `
    <tr class="hover:bg-slate-50 transition-colors ${isCurrent ? 'bg-rose-50/70 font-semibold' : c.ghi > 700 ? 'bg-amber-50/40' : ''}">
      <td class="px-3 py-2.5 font-bold text-slate-900 flex items-center gap-1.5">
        ${isCurrent ? '<span class="size-2 rounded-full bg-rose-500 animate-pulse"></span>' : ''}
        <span>${c.hour}</span>
        ${isCurrent ? '<span class="px-1.5 py-0.2 rounded bg-rose-100 text-rose-800 text-[10px] font-bold">เวลาปัจจุบัน</span>' : ''}
      </td>
      <td class="px-3 py-2.5 text-center">
        <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold ${c.ghi > 0 ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-600'}">
          ${c.emoji || getEmoji(c.icon)} ${c.condition}
        </span>
      </td>
      <td class="px-3 py-2.5 text-center font-bold text-amber-600">${c.ghi}</td>
      <td class="px-3 py-2.5 text-center text-slate-600">${c.dni}</td>
      <td class="px-3 py-2.5 text-center text-slate-600">${c.dhi}</td>
      <td class="px-3 py-2.5 text-center text-sky-600 font-semibold">${c.cloud}%</td>
      <td class="px-3 py-2.5 text-center font-semibold text-rose-600">${c.temp} °C</td>
    </tr>
    `;
  }).join('');
}

// ==========================================
// 5. TAB 5: 1-HOUR AHEAD NOWCAST LAB (SIMULATOR)
// ==========================================
let nowcastSimDate = "2026-04-15";
let nowcastSimHour = 10;
let nowcastSimTarget = "total";
let nowcastSimData = null;
let nowcastSimRevealed = false;
let nowcastChart = null;

async function initNowcastSim() {
  const dateSel = document.getElementById("nowcastDateSelect");
  if (dateSel && (!dateSel.options || dateSel.options.length === 0)) {
    if (allDatesList && allDatesList.length > 0) {
      populateNowcastDateSelect(allDatesList);
    } else {
      try {
        const res = await fetch("/api/available_dates");
        const dates = await res.json();
        allDatesList = dates;
        populateNowcastDateSelect(dates);
      } catch (e) {
        console.error("Failed to load dates for nowcast sim:", e);
      }
    }
  }
  renderNowcastHourPills();
  await loadNowcastSim(nowcastSimDate, nowcastSimHour, nowcastSimTarget);
}

function populateNowcastDateSelect(dates) {
  const sel = document.getElementById("nowcastDateSelect");
  if (!sel) return;
  sel.innerHTML = "";
  dates.forEach(d => {
    const cloudTag = d.cloud_events > 0 ? ` ⛅ (${d.cloud_events} จุดเมฆ)` : ` ☀️`;
    const opt = document.createElement("option");
    opt.value = d.date;
    opt.textContent = `${d.date} ${cloudTag} - ${d.total_kwh} kWh`;
    if (d.date === nowcastSimDate) opt.selected = true;
    sel.appendChild(opt);
  });
  if (!sel.value && dates.length > 0) {
    sel.value = dates[0].date;
    nowcastSimDate = dates[0].date;
  }
}

function renderNowcastHourPills() {
  const container = document.getElementById("nowcastHourPills");
  if (!container) return;

  const hours = Array.from({ length: 12 }, (_, i) => i + 6); // 6 to 17
  container.innerHTML = hours.map(h => {
    const isActive = h === nowcastSimHour;
    const label = `${String(h).padStart(2, '0')}:00`;
    const activeClass = isActive
      ? "bg-indigo-600 text-white font-bold shadow-sm scale-105"
      : "bg-slate-100 text-slate-700 hover:bg-indigo-50 hover:text-indigo-700 font-medium";
    return `
      <button onclick="changeNowcastHour(${h})" class="px-3 py-1.5 rounded-xl text-xs whitespace-nowrap transition-all duration-150 cursor-pointer font-mono-nums ${activeClass}">
        ${label}
      </button>
    `;
  }).join('');
}

function changeNowcastHour(h) {
  nowcastSimHour = parseInt(h);
  renderNowcastHourPills();
  loadNowcastSim(nowcastSimDate, nowcastSimHour, nowcastSimTarget);
}

function stepNowcastHour(delta) {
  let next = nowcastSimHour + delta;
  if (next < 6) next = 6;
  if (next > 17) next = 17;
  nowcastSimHour = next;
  renderNowcastHourPills();
  loadNowcastSim(nowcastSimDate, nowcastSimHour, nowcastSimTarget);
}

function changeNowcastDate(date) {
  nowcastSimDate = date;
  loadNowcastSim(nowcastSimDate, nowcastSimHour, nowcastSimTarget);
}

function changeNowcastTarget(target) {
  nowcastSimTarget = target;
  ["total", "pv1", "pv2", "pv3", "pv4"].forEach(t => {
    const btn = document.getElementById(`nc-tgt-${t}`);
    if (btn) {
      if (t === target) {
        btn.className = "px-2.5 py-1 rounded-lg bg-indigo-600 text-white font-bold shadow-sm";
      } else {
        btn.className = "px-2.5 py-1 rounded-lg text-slate-600 font-medium hover:text-slate-900";
      }
    }
  });
  loadNowcastSim(nowcastSimDate, nowcastSimHour, nowcastSimTarget);
}

async function loadNowcastSim(date, hour, target) {
  // Reset reveal state on hour/date change
  nowcastSimRevealed = false;
  const revealResult = document.getElementById("nc-reveal-result");
  const revealPrompt = document.getElementById("nc-reveal-prompt-box");
  if (revealResult) revealResult.classList.add("hidden");
  if (revealPrompt) revealPrompt.classList.remove("opacity-50");

  try {
    const res = await fetch(`/api/nowcast/simulate?date=${date}&hour=${hour}&target=${target}`);
    const data = await res.json();
    if (data.status !== "success") {
      console.error("Simulation error:", data);
      return;
    }

    nowcastSimData = data;

    // 1. Update Timeline Badges
    const badge = document.getElementById("nc-timeline-badge");
    if (badge) badge.textContent = `${data.current_time} น.`;

    // 2. Update Current Hour Status (t)
    const curTimePill = document.getElementById("nc-cur-time-pill");
    if (curTimePill) curTimePill.textContent = `${data.current_time} น.`;

    const curAct = document.getElementById("nc-cur-act");
    if (curAct) curAct.textContent = `${data.current_actual.toFixed(2)} kW`;

    const curGhi = document.getElementById("nc-cur-ghi");
    if (curGhi) curGhi.textContent = `${data.current_ghi} W/m²`;

    const curCloud = document.getElementById("nc-cur-cloud");
    if (curCloud) curCloud.textContent = `${data.current_cloud}%`;

    const curTemp = document.getElementById("nc-cur-temp");
    if (curTemp) curTemp.textContent = `${data.current_temp} °C`;

    const curBiasDesc = document.getElementById("nc-cur-bias-desc");
    if (curBiasDesc) {
      const sign = data.current_bias >= 0 ? "+" : "";
      curBiasDesc.innerHTML = `ส่วนต่างความผิดพลาดปัจจุบัน: <b class="${data.current_bias >= 0 ? 'text-emerald-600' : 'text-rose-600'} font-mono-nums">${sign}${data.current_bias.toFixed(2)} kW</b> (ส่งต่อ 65% ไปชดเชยโมเดลในชั่วโมงถัดไป)`;
    }

    // 3. Update Prediction Target (t+1)
    const tgtTimePill = document.getElementById("nc-tgt-time-pill");
    if (tgtTimePill) tgtTimePill.textContent = `${data.target_time} น.`;

    const tgtPred = document.getElementById("nc-tgt-pred");
    if (tgtPred) tgtPred.textContent = `${data.target_pred.toFixed(2)} kW`;

    const tgtGhi = document.getElementById("nc-tgt-ghi");
    if (tgtGhi) tgtGhi.textContent = `${data.target_ghi} W/m²`;

    const tgtMult = document.getElementById("nc-tgt-mult");
    if (tgtMult) tgtMult.textContent = `x ${data.target_angle_mult.toFixed(3)}`;

    // 4. Render Initial Blind Chart
    renderNowcastSimChart(data, false);

  } catch (err) {
    console.error("Failed to load nowcast sim data:", err);
  }
}

function revealNowcastTruth() {
  if (!nowcastSimData) return;
  nowcastSimRevealed = true;

  const revealResult = document.getElementById("nc-reveal-result");
  if (revealResult) revealResult.classList.remove("hidden");

  // Emoji and Badge
  const emoji = document.getElementById("nc-result-emoji");
  if (emoji) {
    if (nowcastSimData.accuracy_pct >= 95) emoji.textContent = "🏆";
    else if (nowcastSimData.accuracy_pct >= 85) emoji.textContent = "⭐";
    else if (nowcastSimData.accuracy_pct >= 70) emoji.textContent = "⛅";
    else emoji.textContent = "⛈️";
  }

  const badge = document.getElementById("nc-result-badge");
  if (badge) badge.textContent = nowcastSimData.badge_text;

  const detail = document.getElementById("nc-result-detail");
  if (detail) {
    detail.textContent = `เปรียบเทียบเวลา ${nowcastSimData.target_time} น. — โมเดล 1-Hour Nowcast ทายไว้ ${nowcastSimData.target_pred.toFixed(2)} kW | ค่าจริงที่ Inverter วัดได้ ${nowcastSimData.target_actual.toFixed(2)} kW`;
  }

  const acc = document.getElementById("nc-result-acc");
  if (acc) {
    acc.textContent = `${nowcastSimData.accuracy_pct.toFixed(1)}%`;
    acc.className = `text-3xl font-black font-mono-nums ${
      nowcastSimData.accuracy_pct >= 85 ? 'text-emerald-600' : (nowcastSimData.accuracy_pct >= 70 ? 'text-sky-600' : 'text-amber-600')
    }`;
  }

  const statPred = document.getElementById("nc-stat-pred");
  if (statPred) statPred.textContent = `${nowcastSimData.target_pred.toFixed(2)} kW`;

  const statAct = document.getElementById("nc-stat-act");
  if (statAct) statAct.textContent = `${nowcastSimData.target_actual.toFixed(2)} kW`;

  const statErr = document.getElementById("nc-stat-err");
  if (statErr) statErr.textContent = `${nowcastSimData.error_kw.toFixed(2)} kW`;

  const statWatt = document.getElementById("nc-stat-watt");
  if (statWatt) statWatt.textContent = `(${nowcastSimData.error_watt} Watts)`;

  // Comparison Bar
  const progLabel = document.getElementById("nc-progress-label");
  if (progLabel) {
    progLabel.textContent = `โมเดลเดา: ${nowcastSimData.target_pred.toFixed(2)} kW | ค่าจริง: ${nowcastSimData.target_actual.toFixed(2)} kW`;
  }

  const maxVal = Math.max(nowcastSimData.target_pred, nowcastSimData.target_actual, 1.0);
  const pctPred = Math.min(100, Math.round((nowcastSimData.target_pred / maxVal) * 100));
  const pctAct = Math.min(100, Math.round((nowcastSimData.target_actual / maxVal) * 100));

  const barPred = document.getElementById("nc-bar-pred");
  if (barPred) barPred.style.width = `${pctPred}%`;

  const barAct = document.getElementById("nc-bar-act");
  if (barAct) barAct.style.width = `${pctAct}%`;

  // Update chart to reveal actual point
  renderNowcastSimChart(nowcastSimData, true);
}

function renderNowcastSimChart(data, isRevealed) {
  const ctx = document.getElementById("nowcastSimChart");
  if (!ctx) return;

  if (nowcastChart) nowcastChart.destroy();

  const traj = data.trajectory || [];
  const labels = traj.map(t => t.time_label);

  // Past actuals (up to current_hour)
  const pastActuals = traj.map(t => t.is_past ? t.actual : null);

  // Future actuals (after current_hour) - if revealed show line or point
  const allActualsRevealed = traj.map(t => (isRevealed || t.is_past) ? t.actual : null);

  // Prediction point at target_hour
  const predPoints = traj.map(t => t.is_target ? data.target_pred : null);

  // Actual point at target_hour (only when revealed)
  const actualTargetPoint = traj.map(t => (isRevealed && t.is_target) ? data.target_actual : null);

  // Open-Meteo Base reference curve (dashed)
  const baseOm = traj.map(t => t.base_om);

  const datasets = [
    {
      label: "ค่าจริงช่วงที่ผ่านมา (Actual Power)",
      data: pastActuals,
      borderColor: "#0f172a",
      backgroundColor: "rgba(15, 23, 42, 0.08)",
      borderWidth: 2.5,
      pointRadius: 4,
      pointBackgroundColor: "#0f172a",
      fill: true,
      tension: 0.25
    },
    {
      label: "โมเดล 1-Hour Nowcast เดา (t+1)",
      data: predPoints,
      borderColor: "#4f46e5",
      backgroundColor: "#6366f1",
      pointRadius: 8,
      pointHoverRadius: 10,
      pointBorderColor: "#ffffff",
      pointBorderWidth: 2.5,
      showLine: false,
      type: "line"
    },
    {
      label: "Open-Meteo Solar Base (จำลองก่อนชดเชย)",
      data: baseOm,
      borderColor: "rgba(148, 163, 184, 0.8)",
      borderDash: [4, 4],
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      tension: 0.3
    }
  ];

  if (isRevealed) {
    datasets.push({
      label: "เฉลย: ค่าผลิตจริงที่ Inverter (t+1)",
      data: actualTargetPoint,
      borderColor: "#10b981",
      backgroundColor: "#10b981",
      pointRadius: 9,
      pointHoverRadius: 11,
      pointBorderColor: "#ffffff",
      pointBorderWidth: 2.5,
      showLine: false,
      type: "line"
    });

    datasets.push({
      label: "เส้นค่าจริงตลอดวัน (Revealed)",
      data: allActualsRevealed,
      borderColor: "rgba(16, 185, 129, 0.4)",
      borderWidth: 1.8,
      borderDash: [2, 2],
      pointRadius: 2,
      pointBackgroundColor: "rgba(16, 185, 129, 0.6)",
      fill: false,
      tension: 0.25
    });
  }

  nowcastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "#475569",
            font: { size: 11, family: "'Plus Jakarta Sans'" },
            boxWidth: 12
          }
        },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#38bdf8",
          bodyColor: "#ffffff",
          cornerRadius: 8,
          padding: 10,
          callbacks: {
            label: (ctx) => {
              if (ctx.raw === null || ctx.raw === undefined) return null;
              return ` ${ctx.dataset.label}: ${Number(ctx.raw).toFixed(2)} kW`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#64748b",
            maxTicksLimit: window.innerWidth < 640 ? 6 : 13,
            autoSkip: true,
            font: { size: 10, family: "'Plus Jakarta Sans'" }
          },
          grid: { color: "rgba(226, 232, 240, 0.7)" }
        },
        y: {
          title: { display: true, text: "กำลังผลิต (kW)", color: "#64748b", font: { size: 11 } },
          ticks: { color: "#64748b" },
          grid: { color: "rgba(226, 232, 240, 0.8)" },
          min: 0
        }
      }
    }
  });
}

// ==========================================
// 6. AUTO-REFRESH TIMER (1-HOUR BACKGROUND POLLING)
// ==========================================
let autoRefreshTimer = null;
const AUTO_REFRESH_INTERVAL_MS = 2 * 60 * 1000; // Check every 2 minutes for freshness

function startAutoRefreshTimer() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  autoRefreshTimer = setInterval(() => {
    // Automatically re-fetch future forecast
    loadFutureForecast(currentFutureTarget, false);
  }, AUTO_REFRESH_INTERVAL_MS);
}

