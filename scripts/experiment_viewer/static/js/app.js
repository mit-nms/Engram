(function () {
  const API = "/api";
  const MAX_ITERATIONS_DISPLAY = 100;
  let scoreChart = null;
  let currentRuns = [];
  let aggregateSpecsMap = {};

  function el(id) {
    return document.getElementById(id);
  }

  function show(elId, visible) {
    const e = el(elId);
    if (e) e.classList.toggle("hidden", !visible);
  }

  function setStatus(id, text, klass) {
    const e = el(id);
    if (!e) return;
    e.textContent = text || "";
    e.className = "status " + (klass || "");
  }

  function setRunSelectDisabled(disabled) {
    el("runSelect").disabled = disabled || !currentRuns.length;
  }

  async function getConfig() {
    const r = await fetch(API + "/config");
    return r.json();
  }

  async function postConfig(resultRoots) {
    const r = await fetch(API + "/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result_roots: resultRoots }),
    });
    return r.json();
  }

  async function getRuns() {
    const r = await fetch(API + "/runs");
    return r.json();
  }

  async function getRun(spec) {
    const r = await fetch(API + "/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    });
    return r.json();
  }

  async function postAggregate(specs) {
    const r = await fetch(API + "/aggregate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runs: specs }),
    });
    return r.json();
  }

  async function getLog(path) {
    if (!path) return { content: "" };
    const r = await fetch(API + "/log?path=" + encodeURIComponent(path));
    return r.json();
  }

  function initResultRoots() {
    getConfig().then((c) => {
      const roots = (c.result_roots || []).join("\n");
      el("resultRoots").value = roots;
    });
  }

  function setAndRefresh() {
    const raw = el("resultRoots").value.trim();
    const roots = raw ? raw.split("\n").map((s) => s.trim()).filter(Boolean) : [];
    setStatus("configStatus", "Saving…");
    postConfig(roots)
      .then(() => {
        setStatus("configStatus", "Saved. Loading runs…", "success");
        return getRuns();
      })
      .then((runs) => {
        setStatus("configStatus", roots.length ? "Ready. " + runs.length + " run(s) found." : "No roots set.", "success");
        fillRunSelect(runs);
      })
      .catch((err) => {
        setStatus("configStatus", "Error: " + err.message, "error");
      });
  }

  function fillRunSelect(runs) {
    currentRuns = runs || [];
    aggregateSpecsMap = {};
    const sel = el("runSelect");
    sel.innerHTML = "";
    sel.disabled = !currentRuns.length;
    if (!currentRuns.length) {
      sel.appendChild(new Option("— Set results root above, then click Set & Refresh —", ""));
      return;
    }
    sel.appendChild(new Option("— Select a run or aggregate —", ""));

    const groups = {};
    currentRuns.forEach((r, i) => {
      r._index = i;
      if (r.group_key) {
        const k = r.group_key + "|" + r.type;
        if (!groups[k]) groups[k] = [];
        groups[k].push(r);
      }
    });

    currentRuns.forEach((r, i) => {
      if (!r.group_key) {
        const opt = new Option(r.name + " (" + r.type + ")", "run:" + i);
        opt.dataset.path = r.path;
        opt.dataset.type = r.type;
        opt.dataset.canonical = r.canonical_json || "";
        sel.appendChild(opt);
      }
    });

    Object.keys(groups).forEach((key) => {
      const list = groups[key];
      const prefix = key.indexOf("|") >= 0 ? key.split("|").slice(1).join("|") : key;
      const optgroup = document.createElement("optgroup");
      optgroup.label = "Aggregate (" + list.length + " runs): " + prefix;
      aggregateSpecsMap["aggregate:" + key] = list.map((r) => ({
        path: r.path,
        type: r.type,
        canonical_json: r.canonical_json || null,
      }));
      const aggOpt = new Option("Aggregate (" + list.length + " runs)", "aggregate:" + key);
      optgroup.appendChild(aggOpt);
      list.forEach((r) => {
        const opt = new Option("  " + r.name + " (" + r.type + ")", "run:" + r._index);
        opt.dataset.path = r.path;
        opt.dataset.type = r.type;
        opt.dataset.canonical = r.canonical_json || "";
        optgroup.appendChild(opt);
      });
      sel.appendChild(optgroup);
    });
  }

  function getSelectedRunSpec() {
    const sel = el("runSelect");
    const opt = sel.options[sel.selectedIndex];
    if (!opt || !opt.value) return null;
    if (opt.value.startsWith("aggregate:")) {
      const specs = aggregateSpecsMap[opt.value];
      return specs && specs.length ? { aggregate: specs } : null;
    }
    if (opt.value.startsWith("run:")) {
      const i = parseInt(opt.value.slice(4), 10);
      const r = currentRuns[i];
      if (!r) return null;
      return {
        single: {
          path: r.path,
          type: r.type,
          canonical_json: r.canonical_json || null,
        },
      };
    }
    return null;
  }

  function loadRunData() {
    const sel = getSelectedRunSpec();
    if (!sel) {
      hideAllSections();
      setStatus("runStatus", "");
      return;
    }
    setRunSelectDisabled(true);
    setStatus("runStatus", "Loading…");
    if (sel.single) {
      getRun(sel.single)
        .then((data) => {
          if (data.error) {
            setStatus("runStatus", "Error: " + data.error, "error");
            hideAllSections();
            return;
          }
          setStatus("runStatus", "");
          renderRun(data);
        })
        .catch((err) => {
          setStatus("runStatus", "Error: " + err.message, "error");
          hideAllSections();
        })
        .finally(() => setRunSelectDisabled(false));
    } else {
      postAggregate(sel.aggregate)
        .then((data) => {
          if (data.error) {
            setStatus("runStatus", "Error: " + data.error, "error");
            hideAllSections();
            return;
          }
          setStatus("runStatus", "");
          renderAggregate(data);
        })
        .catch((err) => {
          setStatus("runStatus", "Error: " + err.message, "error");
          hideAllSections();
        })
        .finally(() => setRunSelectDisabled(false));
    }
  }

  function hideAllSections() {
    show("summarySection", false);
    show("chartSection", false);
    show("baselinesSection", false);
    show("codeSection", false);
    show("logSection", false);
    show("iterationsSection", false);
  }

  function renderRun(data) {
    const best = data.best_solution || {};
    const scores = data.scores || [];
    const envelope = data.envelope || [];
    const baselines = data.baselines || {};
    const iters = data.all_iterations || [];
    const total = data.total_iterations != null ? data.total_iterations : iters.length;

    show("summarySection", true);
    el("metricBestScore").textContent =
      best.score != null ? Number(best.score).toFixed(6) : "—";
    el("metricIterations").textContent = total;
    const successCount = iters.filter((i) => i.success).length;
    el("metricSuccess").textContent = successCount + " / " + total;
    el("metricType").textContent = data.run_type || "—";

    show("chartSection", scores.length > 0 || envelope.length > 0);
    if (scores.length || envelope.length) {
      renderChart(scores, envelope, baselines);
    }

    const blKeys = Object.keys(baselines);
    show("baselinesSection", blKeys.length > 0);
    if (blKeys.length) {
      const tbody = el("baselinesTable").querySelector("tbody");
      tbody.innerHTML = "";
      blKeys.forEach((k) => {
        const v = baselines[k];
        const score = typeof v === "object" && v !== null && "score" in v ? v.score : v;
        const tr = document.createElement("tr");
        tr.innerHTML = "<td>" + escapeHtml(k) + "</td><td>" + (score != null ? Number(score).toFixed(6) : "—") + "</td>";
        tbody.appendChild(tr);
      });
    }

    show("codeSection", true);
    const codeEl = el("bestCode");
    codeEl.textContent = best.code || "(no code)";
    if (window.Prism) Prism.highlightElement(codeEl);

    const logPath = data.log_path;
    show("logSection", !!logPath);
    if (logPath) {
      el("logContent").textContent = "Loading…";
      getLog(logPath).then((logData) => {
        let content = logData.content || logData.error || "(empty)";
        if (logData.truncated) {
          content = "[Log truncated — showing last 300,000 characters]\n\n" + content;
        }
        el("logContent").textContent = content;
      });
    }

    const shown = Math.min(iters.length, MAX_ITERATIONS_DISPLAY);
    show("iterationsSection", iters.length > 0);
    if (iters.length) {
      const header = el("iterationsHeader");
      if (header) {
        header.textContent = iters.length > MAX_ITERATIONS_DISPLAY
          ? "Iterations (showing " + shown + " of " + iters.length + ")"
          : "Iterations (" + iters.length + ")";
      }
      const tbody = el("iterationsTable").querySelector("tbody");
      tbody.innerHTML = "";
      iters.slice(0, MAX_ITERATIONS_DISPLAY).forEach((it, i) => {
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" +
          (i + 1) +
          "</td><td>" +
          (it.score != null ? Number(it.score).toFixed(6) : "—") +
          "</td><td>" +
          (it.success ? "✓" : "—") +
          "</td><td>" +
          (it.round != null ? it.round : "—") +
          "</td><td>" +
          escapeHtml(it.node_id || "—") +
          "</td>";
        tbody.appendChild(tr);
      });
    }
  }

  const CHART_COLORS = [
    "rgba(88, 166, 255, 0.9)",
    "rgba(63, 185, 80, 0.9)",
    "rgba(255, 154, 88, 0.9)",
    "rgba(255, 107, 107, 0.9)",
    "rgba(171, 130, 255, 0.9)",
    "rgba(255, 207, 88, 0.9)",
  ];

  const BASELINE_COLORS = [
    "rgba(255, 207, 88, 0.85)",
    "rgba(255, 154, 88, 0.85)",
    "rgba(171, 130, 255, 0.85)",
  ];

  function renderAggregate(data) {
    const runs = data.runs || [];
    const bestScore = data.best_score_ever;
    const bestCode = data.best_code_ever != null ? data.best_code_ever : "";

    show("summarySection", true);
    el("metricBestScore").textContent =
      bestScore != null ? Number(bestScore).toFixed(6) : "—";
    el("metricIterations").textContent = runs.reduce((sum, r) => sum + (r.total_iterations || 0), 0);
    el("metricSuccess").textContent = runs.length + " run(s)";
    el("metricType").textContent = "aggregate";

    const hasEnvelope = runs.some((r) => (r.envelope || []).length > 0);
    show("chartSection", hasEnvelope);
    if (hasEnvelope) {
      renderChartAggregate(runs);
    }

    // Collect baselines from all runs and merge (first occurrence wins per name)
    const mergedBaselines = {};
    runs.forEach((r) => {
      Object.assign(mergedBaselines, r.baselines || {});
    });
    const blKeys = Object.keys(mergedBaselines);
    show("baselinesSection", blKeys.length > 0);
    if (blKeys.length) {
      const tbody = el("baselinesTable").querySelector("tbody");
      tbody.innerHTML = "";
      blKeys.forEach((k) => {
        const v = mergedBaselines[k];
        const score = typeof v === "object" && v !== null && "score" in v ? v.score : v;
        const tr = document.createElement("tr");
        tr.innerHTML = "<td>" + escapeHtml(k) + "</td><td>" + (score != null ? Number(score).toFixed(6) : "—") + "</td>";
        tbody.appendChild(tr);
      });
    }

    show("codeSection", true);
    const codeEl = el("bestCode");
    codeEl.textContent = bestCode || "(best code across runs)";
    if (window.Prism) Prism.highlightElement(codeEl);

    show("logSection", false);
    show("iterationsSection", false);
  }

  function renderChartAggregate(runs) {
    const ctx = el("scoreChart").getContext("2d");
    const maxLen = Math.max(0, ...runs.map((r) => (r.envelope || []).length));
    const labels = Array.from({ length: maxLen }, (_, i) => i + 1);
    const datasets = runs.map((r, i) => ({
      label: r.run_name || "run " + (i + 1),
      data: r.envelope || [],
      borderColor: CHART_COLORS[i % CHART_COLORS.length],
      backgroundColor: "transparent",
      tension: 0.1,
      pointRadius: 0,
    }));

    // Add merged baselines as horizontal reference lines
    const mergedBaselines = {};
    runs.forEach((r) => Object.assign(mergedBaselines, r.baselines || {}));
    Object.keys(mergedBaselines).forEach((k, i) => {
      const v = mergedBaselines[k];
      const score = typeof v === "object" && v !== null && "score" in v ? v.score : v;
      if (score != null && maxLen > 0) {
        datasets.push({
          label: "Baseline: " + k,
          data: Array(maxLen).fill(score),
          borderColor: BASELINE_COLORS[i % BASELINE_COLORS.length],
          backgroundColor: "transparent",
          borderDash: [8, 4],
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        });
      }
    });

    if (scoreChart) scoreChart.destroy();
    scoreChart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "top" } },
        scales: {
          x: {
            title: { display: true, text: "Iteration" },
            grid: { color: "rgba(48, 54, 61, 0.5)" },
            ticks: { color: "#8b949e" },
          },
          y: {
            title: { display: true, text: "Score" },
            grid: { color: "rgba(48, 54, 61, 0.5)" },
            ticks: { color: "#8b949e" },
          },
        },
      },
    });
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function renderChart(scores, envelope, baselines) {
    const ctx = el("scoreChart").getContext("2d");
    const n = Math.max(scores.length, envelope.length);
    const labels = Array.from({ length: n }, (_, i) => i + 1);

    const datasets = [
      {
        label: "Raw score",
        data: scores,
        borderColor: "rgba(88, 166, 255, 0.8)",
        backgroundColor: "rgba(88, 166, 255, 0.1)",
        borderDash: [4, 2],
        tension: 0.1,
        pointRadius: 2,
      },
      {
        label: "Best so far (envelope)",
        data: envelope,
        borderColor: "rgba(63, 185, 80, 0.9)",
        backgroundColor: "rgba(63, 185, 80, 0.1)",
        tension: 0.1,
        pointRadius: 0,
      },
    ];

    // Add baselines as horizontal reference lines
    const blKeys = Object.keys(baselines || {});
    blKeys.forEach((k, i) => {
      const v = baselines[k];
      const score = typeof v === "object" && v !== null && "score" in v ? v.score : v;
      if (score != null && n > 0) {
        datasets.push({
          label: "Baseline: " + k,
          data: Array(n).fill(score),
          borderColor: BASELINE_COLORS[i % BASELINE_COLORS.length],
          backgroundColor: "transparent",
          borderDash: [8, 4],
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
        });
      }
    });

    if (scoreChart) scoreChart.destroy();
    scoreChart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "top" },
        },
        scales: {
          x: {
            title: { display: true, text: "Iteration" },
            grid: { color: "rgba(48, 54, 61, 0.5)" },
            ticks: { color: "#8b949e" },
          },
          y: {
            title: { display: true, text: "Score" },
            grid: { color: "rgba(48, 54, 61, 0.5)" },
            ticks: { color: "#8b949e" },
          },
        },
      },
    });
  }

  el("btnSetRoots").addEventListener("click", setAndRefresh);
  el("runSelect").addEventListener("change", loadRunData);

  initResultRoots();
  getRuns().then((runs) => {
    fillRunSelect(runs);
  });
})();
