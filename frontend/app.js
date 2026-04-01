/*
  This file renders the report page in the browser.

  We keep the JavaScript small and readable:
  - read the prepared report data
  - fill the HTML sections
  - draw a few charts on canvas without external libraries

  That keeps the page lightweight and easy to study.
*/

const report = window.GEOSYNTH_REPORT;

function qs(selector) {
  return document.querySelector(selector);
}

function createCard(card) {
  const article = document.createElement("article");
  article.className = "stat-card";
  article.innerHTML = `
    <p class="stat-label">${card.label}</p>
    <p class="stat-value">${card.value}</p>
    <p class="stat-detail">${card.detail}</p>
  `;
  return article;
}

function renderTopCards() {
  const container = qs("#stat-grid");
  report.report_cards.forEach((card) => container.appendChild(createCard(card)));
}

function createInventoryCard([key, dataset]) {
  const article = document.createElement("article");
  article.className = "inventory-card";

  article.innerHTML = `
    <div class="inventory-header">
      <h3>${dataset.title}</h3>
      <span class="status-pill">${dataset.status}</span>
    </div>
    <div class="inventory-metrics">
      <div>
        <span class="metric-label">Train Samples</span>
        <strong>${dataset.train_samples}</strong>
      </div>
      <div>
        <span class="metric-label">Validation Samples</span>
        <strong>${dataset.val_samples}</strong>
      </div>
    </div>
    <p class="inventory-note">${dataset.notes}</p>
  `;

  return article;
}

function renderInventory() {
  const container = qs("#inventory-grid");
  Object.entries(report.inventory).forEach((entry) => {
    container.appendChild(createInventoryCard(entry));
  });
}

function renderList(selector, items) {
  const container = qs(selector);
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    container.appendChild(li);
  });
}

function createPipelineCard(step) {
  const article = document.createElement("article");
  article.className = "pipeline-card";
  article.innerHTML = `
    <h3>${step.title}</h3>
    <p>${step.body}</p>
  `;
  return article;
}

function renderPipeline() {
  const container = qs("#pipeline-grid");
  report.pipeline_steps.forEach((step) => {
    container.appendChild(createPipelineCard(step));
  });
}

function createArtifactCard(item) {
  const article = document.createElement("article");
  article.className = "artifact-card";
  article.innerHTML = `
    <p class="stat-label">${item.title}</p>
    <p class="stat-value">${item.value}</p>
    <p class="stat-detail">${item.detail}</p>
  `;
  return article;
}

function renderArtifacts() {
  const container = qs("#artifact-grid");
  report.artifacts.forEach((item) => {
    container.appendChild(createArtifactCard(item));
  });
}

function createGalleryCard(item) {
  const article = document.createElement("article");
  article.className = "gallery-card";
  article.innerHTML = `
    <img src="${item.image}" alt="${item.title}" />
    <div class="gallery-copy">
      <p class="gallery-task">${item.task}</p>
      <h3>${item.title}</h3>
    </div>
  `;
  return article;
}

function renderGallery() {
  const container = qs("#gallery-grid");
  report.gallery.forEach((item) => {
    container.appendChild(createGalleryCard(item));
  });
}

function drawLineChart(canvas, history, series) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height;
  const padding = { top: 20, right: 18, bottom: 36, left: 46 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;

  for (let row = 0; row <= 4; row += 1) {
    const y = padding.top + (chartHeight / 4) * row;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  if (!history.length) {
    ctx.fillStyle = "#f4f1e7";
    ctx.font = '16px "Avenir Next", "Trebuchet MS", sans-serif';
    ctx.fillText("No history available yet", padding.left, padding.top + 30);
    return;
  }

  const allValues = history.flatMap((epoch) => series.map((item) => epoch[item.key] ?? 0));
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const safeMax = maxValue === minValue ? maxValue + 1 : maxValue;

  function projectX(index) {
    if (history.length === 1) {
      return padding.left + chartWidth / 2;
    }
    return padding.left + (chartWidth * index) / (history.length - 1);
  }

  function projectY(value) {
    const ratio = (value - minValue) / (safeMax - minValue);
    return padding.top + chartHeight - ratio * chartHeight;
  }

  series.forEach((line) => {
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 3;
    ctx.beginPath();

    history.forEach((epoch, index) => {
      const x = projectX(index);
      const y = projectY(epoch[line.key] ?? 0);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    history.forEach((epoch, index) => {
      const x = projectX(index);
      const y = projectY(epoch[line.key] ?? 0);
      ctx.fillStyle = line.color;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  ctx.fillStyle = "rgba(244, 241, 231, 0.8)";
  ctx.font = '12px "Avenir Next", "Trebuchet MS", sans-serif';
  history.forEach((epoch, index) => {
    const x = projectX(index);
    ctx.fillText(`E${epoch.epoch}`, x - 10, height - 12);
  });
}

function drawBarChart(canvas, inventory) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const width = rect.width;
  const height = rect.height;
  const padding = { top: 20, right: 18, bottom: 48, left: 46 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const entries = Object.entries(inventory);
  const values = entries.map(([, value]) => value.train_samples + value.val_samples);
  const maxValue = Math.max(...values, 1);
  const barWidth = chartWidth / Math.max(entries.length * 1.5, 1);
  const gap = barWidth / 2;

  ctx.clearRect(0, 0, width, height);

  entries.forEach(([key, value], index) => {
    const total = value.train_samples + value.val_samples;
    const barHeight = (total / maxValue) * chartHeight;
    const x = padding.left + index * (barWidth + gap) + gap / 2;
    const y = padding.top + chartHeight - barHeight;

    const gradient = ctx.createLinearGradient(x, y, x, padding.top + chartHeight);
    gradient.addColorStop(0, "#1cc5b7");
    gradient.addColorStop(1, "#f3a53b");

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barHeight, 14);
    ctx.fill();

    ctx.fillStyle = "#f4f1e7";
    ctx.font = '12px "Avenir Next", "Trebuchet MS", sans-serif';
    ctx.fillText(String(total), x + barWidth / 2 - 8, y - 8);
    ctx.fillText(key.replace("_", " "), x - 4, height - 16);
  });
}

function renderCommandList() {
  const container = qs("#command-list");
  report.run_commands.forEach((command) => {
    const pre = document.createElement("pre");
    pre.textContent = command;
    container.appendChild(pre);
  });
}

function renderNarrative() {
  qs("#page-title").textContent = report.project.title;
  qs("#page-subtitle").textContent = report.project.subtitle;
  qs("#page-summary").textContent = report.project.summary;
  qs("#mentor-pitch").textContent = report.project.mentor_pitch;
  qs("#generated-on").textContent = `Generated on ${report.project.generated_on}`;
}

function renderCharts() {
  drawLineChart(qs("#roads-chart"), report.histories.roads, [
    { key: "train_loss", color: "#1cc5b7" },
    { key: "val_loss", color: "#f3a53b" },
  ]);

  drawLineChart(qs("#changes-chart"), report.histories.changes, [
    { key: "train_dice", color: "#f3a53b" },
    { key: "val_dice", color: "#1cc5b7" },
  ]);

  drawBarChart(qs("#inventory-chart"), report.inventory);
}

function init() {
  renderNarrative();
  renderTopCards();
  renderPipeline();
  renderInventory();
  renderArtifacts();
  renderList("#achievements-list", report.achievements);
  renderList("#next-steps-list", report.next_steps);
  renderGallery();
  renderCommandList();
  renderCharts();
}

window.addEventListener("load", init);
window.addEventListener("resize", renderCharts);
