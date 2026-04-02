(function () {

  var FALLBACK = {
    generated: new Date().toISOString(),
    kpi: {
      task_families: 2,
      datasets: 3,
      tracked_samples: 2941,
      checkpoints: 2,
      preview_panels: 5,
      artifacts: 11
    },
    datasets: [
      { name: "roads",      train: 72,   val: 14,  status: "READY" },
      { name: "water_land", train: 2273, val: 568, status: "READY" },
      { name: "changes",    train: 12,   val: 2,   status: "READY" }
    ],
    histories: {
      roads: {
        train_loss: [0.521, 0.468, 0.412, 0.375, 0.348, 0.327, 0.311, 0.298, 0.288, 0.279],
        val_loss:   [0.554, 0.498, 0.445, 0.412, 0.387, 0.368, 0.354, 0.342, 0.334, 0.328],
        train_dice: [],
        val_dice:   []
      },
      changes: {
        train_loss: [],
        val_loss:   [],
        train_dice: [0.31, 0.40, 0.48, 0.54, 0.59, 0.63, 0.66, 0.69, 0.71, 0.73],
        val_dice:   [0.28, 0.36, 0.44, 0.50, 0.55, 0.58, 0.61, 0.63, 0.65, 0.66]
      }
    },
    metrics: {
      roads: {
        epochs: 10,
        val_loss: 0.328,
        val_iou: 0.612,
        val_dice: 0.759,
        val_precision: 0.781,
        val_recall: 0.738
      },
      changes: {
        epochs: 10,
        val_loss: 0.441,
        val_iou: 0.493,
        val_dice: 0.661,
        val_precision: 0.704,
        val_recall: 0.622
      }
    },
    predictions: {
      roads:   ["../outputs/predictions/roads/roads_preview_0.png"],
      changes: ["../outputs/predictions/changes/changes_preview_0.png"]
    },
    registry: [
      { key: "Model Architecture", val: "U-Net",          sub: "encoder: ResNet-34" },
      { key: "Road Checkpoint",    val: "epoch_10.pt",    sub: "roads task" },
      { key: "Change Checkpoint",  val: "epoch_10.pt",    sub: "changes task" },
      { key: "Optimizer",          val: "Adam",           sub: "lr=1e-4" },
      { key: "Loss Function",      val: "BCE + Dice",     sub: "roads / changes" },
      { key: "Input Resolution",   val: "512 x 512",      sub: "px" },
      { key: "Report Generated",   val: "--",             sub: "build_report.py" },
      { key: "Serve Root",         val: "repo root",      sub: "python serve.py" }
    ]
  };

  var D = (typeof window !== "undefined" && window.GEOSYNTH_REPORT) ? window.GEOSYNTH_REPORT : FALLBACK;

  var DARK = "#23262b";
  var MID  = "#8a909b";
  var GRID = "#e8eaed";
  var FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";

  function fmt(n, decimals) {
    if (n === null || n === undefined) return "--";
    return Number(n).toFixed(decimals !== undefined ? decimals : 3);
  }

  function niceYTicks(minV, maxV, count) {
    var range = maxV - minV;
    var step  = Math.pow(10, Math.floor(Math.log10(range / count)));
    var nice  = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10];
    for (var i = 0; i < nice.length; i++) {
      if (nice[i] * count >= range) { step = nice[i]; break; }
    }
    var start = Math.floor(minV / step) * step;
    var ticks = [];
    for (var t = start; t <= maxV + step * 0.1; t = Math.round((t + step) * 1e10) / 1e10) {
      ticks.push(t);
      if (ticks.length > count + 2) break;
    }
    return ticks;
  }

  function setupCanvas(canvas) {
    var dpr  = window.devicePixelRatio || 1;
    var rect = canvas.parentElement.getBoundingClientRect();
    var w    = Math.floor(rect.width);
    var h    = 180;
    canvas.width  = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width  = w + "px";
    canvas.style.height = h + "px";
    var ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { ctx: ctx, W: w, H: h };
  }

  function drawAxes(ctx, pad, W, H) {
    ctx.strokeStyle = "#c8ccd2";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, H - pad.b);
    ctx.lineTo(W - pad.r, H - pad.b);
    ctx.stroke();
  }

  function drawPoint(ctx, x, y, color) {
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 2.8, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawLineChart(canvasId, series, opts) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var _ = setupCanvas(canvas);
    var ctx = _.ctx, W = _.W, H = _.H;
    var pad = { t: 14, r: 16, b: 38, l: 52 };
    var cW  = W - pad.l - pad.r;
    var cH  = H - pad.t - pad.b;

    var allVals = [];
    series.forEach(function (s) { allVals = allVals.concat(s.data); });
    if (!allVals.length) return;

    var minV = Math.min.apply(null, allVals);
    var maxV = Math.max.apply(null, allVals);
    if (minV === maxV) { minV -= 0.05; maxV += 0.05; }

    var ticks = niceYTicks(minV, maxV, 5);
    var tMin  = ticks[0];
    var tMax  = ticks[ticks.length - 1];

    ctx.clearRect(0, 0, W, H);
    ctx.font = "10px " + FONT;
    ctx.textAlign = "right";

    ticks.forEach(function (t) {
      var y = pad.t + cH - ((t - tMin) / (tMax - tMin)) * cH;
      if (y < pad.t - 2 || y > H - pad.b + 2) return;
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(W - pad.r, y);
      ctx.stroke();
      ctx.fillStyle = "#8a909b";
      ctx.fillText(t.toFixed(3), pad.l - 5, y + 3.5);
    });

    drawAxes(ctx, pad, W, H);

    var numPts = series[0].data.length;
    var xStep  = numPts > 1 ? cW / (numPts - 1) : 0;

    if (numPts === 1) {
      var centerX = pad.l + cW / 2;
      ctx.strokeStyle = "#dfe3e8";
      ctx.beginPath();
      ctx.moveTo(centerX, pad.t);
      ctx.lineTo(centerX, H - pad.b);
      ctx.stroke();
    }

    series.forEach(function (s, seriesIndex) {
      if (!s.data.length) return;
      ctx.strokeStyle = s.color;
      ctx.lineWidth   = 1.8;
      ctx.lineJoin    = "round";
      ctx.beginPath();
      s.data.forEach(function (v, i) {
        var x = pad.l + i * xStep;
        if (numPts === 1) {
          x = pad.l + cW / 2 + (seriesIndex - (series.length - 1) / 2) * 18;
        }
        var y = pad.t + cH - ((v - tMin) / (tMax - tMin)) * cH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      s.data.forEach(function (v, i) {
        var x = pad.l + i * xStep;
        if (numPts === 1) {
          x = pad.l + cW / 2 + (seriesIndex - (series.length - 1) / 2) * 18;
        }
        var y = pad.t + cH - ((v - tMin) / (tMax - tMin)) * cH;
        drawPoint(ctx, x, y, s.color);
        ctx.fillStyle = s.color;
        ctx.textAlign = "center";
        ctx.font = "10px " + FONT;
        ctx.fillText(v.toFixed(3), x, y - 10);
      });
    });

    var labelStep = Math.max(1, Math.floor(numPts / 8));
    ctx.fillStyle  = "#8a909b";
    ctx.textAlign  = "center";
    ctx.font       = "10px " + FONT;
    for (var i = 0; i < numPts; i += labelStep) {
      var x = pad.l + i * xStep;
      if (numPts === 1) {
        x = pad.l + cW / 2;
      }
      ctx.fillText(i + 1, x, H - pad.b + 12);
    }
    ctx.fillStyle  = "#8a909b";
    ctx.textAlign  = "center";
    ctx.font       = "10px " + FONT;
    ctx.fillText("Epoch", pad.l + cW / 2, H - pad.b + 26);

    if (numPts === 1) {
      ctx.fillStyle = "#8a909b";
      ctx.font = "10px " + FONT;
      ctx.textAlign = "right";
      ctx.fillText("single epoch available", W - pad.r, pad.t - 2);
    }
  }

  function drawGroupedBarChart(canvasId, groups, labels, opts) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var _ = setupCanvas(canvas);
    var ctx = _.ctx, W = _.W, H = _.H;
    var pad     = { t: 14, r: 16, b: 46, l: 52 };
    var cW      = W - pad.l - pad.r;
    var cH      = H - pad.t - pad.b;
    var numG    = labels.length;
    var numS    = groups.length;
    var groupW  = cW / numG;
    var barW    = Math.min(18, (groupW - 12) / numS);
    var gap     = 2;

    var allVals = [];
    groups.forEach(function (g) { allVals = allVals.concat(g.data); });
    var maxV = Math.max.apply(null, allVals) * 1.05;
    var ticks = niceYTicks(0, maxV, 5);
    var tMax  = ticks[ticks.length - 1];

    ctx.clearRect(0, 0, W, H);
    ctx.font = "10px " + FONT;
    ctx.textAlign = "right";

    ticks.forEach(function (t) {
      var y = pad.t + cH - (t / tMax) * cH;
      if (y < pad.t - 2 || y > H - pad.b + 2) return;
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(W - pad.r, y);
      ctx.stroke();
      var label = t >= 1000 ? (t / 1000).toFixed(1) + "k" : t.toFixed(0);
      ctx.fillStyle = "#8a909b";
      ctx.fillText(label, pad.l - 5, y + 3.5);
    });

    drawAxes(ctx, pad, W, H);

    labels.forEach(function (lbl, gi) {
      var gx = pad.l + gi * groupW + groupW / 2;
      var totalW = numS * barW + (numS - 1) * gap;
      var startX = gx - totalW / 2;

      groups.forEach(function (s, si) {
        var val = s.data[gi] || 0;
        var bh  = (val / tMax) * cH;
        var bx  = startX + si * (barW + gap);
        var by  = pad.t + cH - bh;
        ctx.fillStyle = s.color;
        ctx.fillRect(bx, by, barW, bh);
        ctx.fillStyle = "#6f7780";
        ctx.textAlign = "center";
        ctx.font = "10px " + FONT;
        ctx.fillText(val.toLocaleString(), bx + barW / 2, by - 6);
      });

      ctx.fillStyle  = "#8a909b";
      ctx.textAlign  = "center";
      ctx.font       = "10px " + FONT;
      var shortLbl = lbl.replace("water_land", "water").replace("changes", "chg");
      ctx.fillText(shortLbl, gx, H - pad.b + 12);
    });
  }

  function drawMetricBarChart(canvasId, tasks, metricLabels, seriesData) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var _ = setupCanvas(canvas);
    var ctx = _.ctx, W = _.W, H = _.H;
    var pad    = { t: 14, r: 16, b: 46, l: 38 };
    var cW     = W - pad.l - pad.r;
    var cH     = H - pad.t - pad.b;
    var numG   = metricLabels.length;
    var numS   = seriesData.length;
    var groupW = cW / numG;
    var barW   = Math.min(16, (groupW - 12) / numS);
    var gap    = 2;

    ctx.clearRect(0, 0, W, H);

    var tickVals = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    ctx.font = "10px " + FONT;
    ctx.textAlign = "right";

    tickVals.forEach(function (t) {
      var y = pad.t + cH - t * cH;
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(W - pad.r, y);
      ctx.stroke();
      ctx.fillStyle = "#8a909b";
      ctx.fillText(t.toFixed(1), pad.l - 4, y + 3.5);
    });

    drawAxes(ctx, pad, W, H);

    metricLabels.forEach(function (lbl, gi) {
      var gx    = pad.l + gi * groupW + groupW / 2;
      var totalW = numS * barW + (numS - 1) * gap;
      var startX = gx - totalW / 2;

      seriesData.forEach(function (s, si) {
        var val = Math.max(0, Math.min(1, s.data[gi] || 0));
        var bh  = val * cH;
        var bx  = startX + si * (barW + gap);
        var by  = pad.t + cH - bh;
        ctx.fillStyle = s.color;
        ctx.fillRect(bx, by, barW, bh);
        ctx.fillStyle = "#6f7780";
        ctx.textAlign = "center";
        ctx.font = "10px " + FONT;
        ctx.fillText(val.toFixed(2), bx + barW / 2, by - 6);
      });

      ctx.fillStyle  = "#8a909b";
      ctx.textAlign  = "center";
      ctx.font       = "10px " + FONT;
      ctx.fillText(lbl, gx, H - pad.b + 12);
    });
  }

  function renderKPIs(d) {
    var k = d.kpi;
    var ts = d.generated ? d.generated.replace("T", "  ").substring(0, 19) : "--";
    document.getElementById("k-tasks").textContent       = k.task_families;
    document.getElementById("k-datasets").textContent    = k.datasets;
    document.getElementById("k-samples").textContent     = k.tracked_samples.toLocaleString();
    document.getElementById("k-checkpoints").textContent = k.checkpoints;
    document.getElementById("k-panels").textContent      = k.preview_panels;
    document.getElementById("k-artifacts").textContent   = k.artifacts;
    var el = document.getElementById("report-ts");
    if (el) el.textContent = ts;
    var ft = document.getElementById("footer-ts");
    if (ft) ft.textContent = "Generated " + ts;
  }

  function renderCharts(d) {
    var h = d.histories;

    drawLineChart("c-road-loss", [
      { data: h.roads.train_loss, color: DARK },
      { data: h.roads.val_loss,   color: MID  }
    ], {});

    drawLineChart("c-change-dice", [
      { data: h.changes.train_dice, color: DARK },
      { data: h.changes.val_dice,   color: MID  }
    ], {});

    var dsLabels = d.datasets.map(function (ds) { return ds.name; });
    var trainVals = d.datasets.map(function (ds) { return ds.train; });
    var valVals   = d.datasets.map(function (ds) { return ds.val;   });
    drawGroupedBarChart("c-ds-volume",
      [
        { data: trainVals, color: DARK },
        { data: valVals,   color: MID  }
      ],
      dsLabels, {}
    );

    var m = d.metrics;
    var metricLabels = ["IoU", "Dice", "Prec.", "Recall"];
    var roadsRow   = [m.roads.val_iou, m.roads.val_dice, m.roads.val_precision, m.roads.val_recall];
    var changesRow = [m.changes.val_iou, m.changes.val_dice, m.changes.val_precision, m.changes.val_recall];
    drawMetricBarChart("c-val-metrics", ["roads", "changes"], metricLabels, [
      { data: roadsRow,   color: DARK },
      { data: changesRow, color: MID  }
    ]);
  }

  function renderMiniStatGrid(targetId, items) {
    var target = document.getElementById(targetId);
    if (!target) return;

    target.innerHTML = items.map(function (item) {
      return "<div class='mini-stat-card'>" +
        "<span class='mini-stat-label'>" + item.label + "</span>" +
        "<span class='mini-stat-value'>" + item.value + "</span>" +
        "<span class='mini-stat-sub'>" + item.sub + "</span>" +
        "</div>";
    }).join("");
  }

  function renderPanelStats(d) {
    var roadTrainLoss = d.histories.roads.train_loss.length ? d.histories.roads.train_loss[d.histories.roads.train_loss.length - 1] : null;
    var roadValLoss = d.histories.roads.val_loss.length ? d.histories.roads.val_loss[d.histories.roads.val_loss.length - 1] : null;
    var changeTrainDice = d.histories.changes.train_dice.length ? d.histories.changes.train_dice[d.histories.changes.train_dice.length - 1] : null;
    var changeValDice = d.histories.changes.val_dice.length ? d.histories.changes.val_dice[d.histories.changes.val_dice.length - 1] : null;

    var totalTrain = d.datasets.reduce(function (sum, ds) { return sum + ds.train; }, 0);
    var totalVal = d.datasets.reduce(function (sum, ds) { return sum + ds.val; }, 0);
    var largestDataset = d.datasets.slice().sort(function (a, b) {
      return (b.train + b.val) - (a.train + a.val);
    })[0];

    renderMiniStatGrid("road-stat-grid", [
      { label: "epochs", value: String(d.metrics.roads.epochs || 0), sub: "history rows" },
      { label: "train loss", value: fmt(roadTrainLoss), sub: "latest" },
      { label: "val loss", value: fmt(roadValLoss), sub: "latest" }
    ]);

    renderMiniStatGrid("change-stat-grid", [
      { label: "epochs", value: String(d.metrics.changes.epochs || 0), sub: "history rows" },
      { label: "train dice", value: fmt(changeTrainDice), sub: "latest" },
      { label: "val dice", value: fmt(changeValDice), sub: "latest" }
    ]);

    renderMiniStatGrid("volume-stat-grid", [
      { label: "train total", value: totalTrain.toLocaleString(), sub: "all datasets" },
      { label: "val total", value: totalVal.toLocaleString(), sub: "all datasets" },
      { label: "largest", value: largestDataset ? largestDataset.name : "--", sub: largestDataset ? (largestDataset.train + largestDataset.val).toLocaleString() + " samples" : "--" }
    ]);

    renderMiniStatGrid("validation-stat-grid", [
      { label: "road iou", value: fmt(d.metrics.roads.val_iou), sub: "validation" },
      { label: "road dice", value: fmt(d.metrics.roads.val_dice), sub: "validation" },
      { label: "change iou", value: fmt(d.metrics.changes.val_iou), sub: "validation" },
      { label: "change dice", value: fmt(d.metrics.changes.val_dice), sub: "validation" }
    ]);
  }

  function renderPerfTable(d) {
    var tbody = document.getElementById("perf-tbody");
    if (!tbody) return;
    var rows = [
      ["roads",   d.metrics.roads],
      ["changes", d.metrics.changes]
    ];
    tbody.innerHTML = rows.map(function (r) {
      var name = r[0], m = r[1];
      return "<tr>" +
        "<td>" + name + "</td>" +
        "<td>" + (m.epochs || "--") + "</td>" +
        "<td>" + fmt(m.val_loss) + "</td>" +
        "<td>" + fmt(m.val_iou) + "</td>" +
        "<td>" + fmt(m.val_dice) + "</td>" +
        "<td>" + fmt(m.val_precision) + "</td>" +
        "<td>" + fmt(m.val_recall) + "</td>" +
        "</tr>";
    }).join("");
  }

  function renderDatasetTable(d) {
    var tbody = document.getElementById("ds-tbody");
    if (!tbody) return;
    tbody.innerHTML = d.datasets.map(function (ds) {
      var cls = ds.status === "READY" ? "status-ready" : "status-pending";
      return "<tr>" +
        "<td>" + ds.name + "</td>" +
        "<td><span class='status-badge " + cls + "'>" + ds.status + "</span></td>" +
        "<td>" + ds.train.toLocaleString() + "</td>" +
        "<td>" + ds.val.toLocaleString() + "</td>" +
        "<td>" + (ds.train + ds.val).toLocaleString() + "</td>" +
        "</tr>";
    }).join("");
  }

  function loadResultImages(d) {
    var rRoads   = d.predictions.roads;
    var rChanges = d.predictions.changes;
    var imgR = document.getElementById("img-road");
    var imgC = document.getElementById("img-change");
    if (imgR && rRoads && rRoads.length)   imgR.src = rRoads[0];
    if (imgC && rChanges && rChanges.length) imgC.src = rChanges[0];
  }

  function renderRegistry(d) {
    var grid = document.getElementById("registry-grid");
    if (!grid) return;
    var items = d.registry || [];
    if (!items.length) return;

    if (d.generated) {
      items = items.map(function (item) {
        if (item.key === "Report Generated") {
          return { key: item.key, val: d.generated.replace("T"," ").substring(0, 19), sub: item.sub };
        }
        return item;
      });
    }

    grid.innerHTML = items.map(function (item) {
      return "<div class='reg-card'>" +
        "<div class='reg-key'>" + item.key + "</div>" +
        "<div class='reg-val'>" + item.val + "</div>" +
        (item.sub ? "<div class='reg-sub'>" + item.sub + "</div>" : "") +
        "</div>";
    }).join("");
  }

  function init() {
    renderKPIs(D);
    renderCharts(D);
    renderPanelStats(D);
    renderPerfTable(D);
    renderDatasetTable(D);
    loadResultImages(D);
    renderRegistry(D);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.addEventListener("resize", function () {
    renderCharts(D);
  });

})();
