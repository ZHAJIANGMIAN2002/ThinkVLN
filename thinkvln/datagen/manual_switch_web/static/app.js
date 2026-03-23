const state = {
  page: 1,
  pageSize: Number(document.body.dataset.pageSize || 20),
  totalPages: 1,
  view: "pending",
  loading: false,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setFlash(message) {
  const flash = document.getElementById("flash");
  if (!message) {
    flash.hidden = true;
    flash.textContent = "";
    return;
  }
  flash.hidden = false;
  flash.textContent = message;
}

function renderStatus(status) {
  document.getElementById("total-count").textContent = String(status.total);
  document.getElementById("labeled-count").textContent = String(status.labeled);
  document.getElementById("remaining-count").textContent = String(status.remaining);
}

function renderPager(page, totalPages) {
  state.page = page;
  state.totalPages = totalPages;
  document.getElementById("page-info").textContent = `Page ${page} / ${totalPages}`;
  document.getElementById("prev-page").disabled = page <= 1 || state.loading;
  document.getElementById("next-page").disabled = page >= totalPages || state.loading;
}

async function fetchSamples(page) {
  const response = await fetch(`/api/samples?page=${page}&page_size=${state.pageSize}&view=${state.view}`);
  if (!response.ok) {
    throw new Error(`failed to load samples: ${response.status}`);
  }
  return response.json();
}

async function saveAnnotation(sampleId, shouldSwitch) {
  const response = await fetch("/api/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_id: sampleId, should_switch: shouldSwitch }),
  });
  if (!response.ok) {
    throw new Error(`failed to save annotation: ${response.status}`);
  }
  return response.json();
}

function renderDecision(card, annotation) {
  const valueNode = card.querySelector(".decision-value");
  const buttons = card.querySelectorAll(".decision-btn");
  card.classList.remove("is-switch", "is-stay");
  buttons.forEach((button) => button.classList.remove("is-active"));
  if (!annotation || typeof annotation.should_switch !== "boolean") {
    valueNode.textContent = "unset";
    return;
  }
  const rawValue = annotation.should_switch ? "true" : "false";
  const activeButton = card.querySelector(`.decision-btn[data-value="${rawValue}"]`);
  if (activeButton) {
    activeButton.classList.add("is-active");
  }
  card.classList.add(annotation.should_switch ? "is-switch" : "is-stay");
  valueNode.textContent = rawValue;
}

function loadImageFrame(url) {
  return new Promise((resolve, reject) => {
    const frame = new Image();
    frame.decoding = "async";
    frame.loading = "eager";
    frame.onload = async () => {
      if (typeof frame.decode === "function") {
        try {
          await frame.decode();
        } catch (error) {
        }
      }
      resolve(frame);
    };
    frame.onerror = () => reject(new Error(`Failed to load frame: ${url}`));
    frame.src = url;
  });
}

async function ensureFramesLoaded(card) {
  if (card._frameLoadPromise) {
    return card._frameLoadPromise;
  }
  const frames = JSON.parse(card.dataset.frames);
  card._frameLoadPromise = Promise.all(frames.map((url) => loadImageFrame(url)))
    .then((loadedFrames) => {
      card._loadedFrames = loadedFrames;
      return loadedFrames;
    })
    .catch((error) => {
      card._frameLoadPromise = null;
      throw error;
    });
  return card._frameLoadPromise;
}

function initPlayer(card) {
  const frames = JSON.parse(card.dataset.frames);
  const image = card.querySelector(".rollout-frame");
  const slider = card.querySelector(".frame-slider");
  const playButton = card.querySelector(".play-btn");
  const counter = card.querySelector(".frame-count");
  let frameIndex = 0;
  let timer = null;

  function renderFrame(nextIndex) {
    frameIndex = Math.max(0, Math.min(nextIndex, frames.length - 1));
    const loadedFrames = card._loadedFrames || [];
    image.src = loadedFrames[frameIndex] ? loadedFrames[frameIndex].src : frames[frameIndex];
    slider.value = String(frameIndex);
    counter.textContent = `${frameIndex + 1} / ${frames.length}`;
  }

  function stop() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
    playButton.textContent = "Play";
  }

  playButton.addEventListener("click", async () => {
    if (timer !== null) {
      stop();
      return;
    }
    if (frameIndex >= frames.length - 1) {
      renderFrame(0);
    }
    playButton.disabled = true;
    playButton.textContent = "Loading...";
    try {
      await ensureFramesLoaded(card);
    } catch (error) {
      playButton.textContent = "Retry";
      playButton.disabled = false;
      return;
    }
    playButton.disabled = false;
    playButton.textContent = "Pause";
    timer = window.setInterval(() => {
      if (frameIndex >= frames.length - 1) {
        stop();
        return;
      }
      renderFrame(frameIndex + 1);
    }, 180);
  });

  slider.addEventListener("input", () => {
    stop();
    renderFrame(Number(slider.value));
  });

  renderFrame(0);
}

function buildSampleCard(sample) {
  const doneSteps = sample.done_steps.length ? `<ul>${sample.done_steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul>` : "<p>None</p>";
  const pendingSteps = sample.pending_steps.length ? `<ul>${sample.pending_steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul>` : "<p>None</p>";
  return `
    <section class="sample-card" data-sample-id="${escapeHtml(sample.sample_id)}" data-frames='${escapeHtml(JSON.stringify(sample.rollout_frame_urls))}'>
      <div class="card-head">
        <div>
          <h2>${escapeHtml(sample.sample_id)}</h2>
          <p class="muted">Episode ${escapeHtml(sample.episode_id)} | Pivot ${escapeHtml(sample.pivot_frame)} | Rollout ${escapeHtml(sample.rollout_id)}</p>
        </div>
        <div class="decision-group">
          <button class="decision-btn" data-value="true" type="button">Switch</button>
          <button class="decision-btn" data-value="false" type="button">Stay</button>
          <button class="decision-btn" data-value="clear" type="button">Clear</button>
        </div>
      </div>
      <div class="context-grid">
        <div class="context-block">
          <span class="label">Instruction</span>
          <p>${escapeHtml(sample.instruction)}</p>
        </div>
        <div class="context-block">
          <span class="label">Done Plan</span>
          ${doneSteps}
        </div>
        <div class="context-block">
          <span class="label">Current Task</span>
          <p>${escapeHtml(sample.active_subtask)}</p>
        </div>
        <div class="context-block">
          <span class="label">Pending Plan</span>
          ${pendingSteps}
        </div>
        <div class="context-block">
          <span class="label">Next Task If Switched</span>
          <p>${escapeHtml(sample.next_subtask)}</p>
        </div>
      </div>
      <div class="viewer">
        <div class="pivot-pane">
          <span class="label">Pivot Frame</span>
          <img class="pivot-image" src="${escapeHtml(sample.pivot_image_url)}" alt="pivot frame" loading="lazy" />
        </div>
        <div class="player-pane">
          <span class="label">Rollout Video</span>
          <img class="rollout-frame" src="${escapeHtml(sample.rollout_frame_urls[0])}" alt="rollout frame" loading="lazy" />
          <div class="controls">
            <button class="play-btn" type="button">Play</button>
            <input class="frame-slider" type="range" min="0" max="${Math.max(0, sample.rollout_frame_urls.length - 1)}" value="0" />
            <span class="frame-count">1 / ${sample.rollout_frame_urls.length}</span>
          </div>
        </div>
      </div>
      <p class="decision-state">should_switch: <strong class="decision-value">unset</strong></p>
    </section>
  `;
}

function bindSampleCard(card, sample) {
  initPlayer(card);
  renderDecision(card, sample.annotation);
  card.querySelectorAll(".decision-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const rawValue = button.dataset.value;
      const shouldSwitch = rawValue === "clear" ? null : rawValue === "true";
      try {
        state.loading = true;
        setFlash("Saving...");
        const payload = await saveAnnotation(sample.sample_id, shouldSwitch);
        renderStatus(payload.status);
        await loadPage(state.page);
        setFlash("");
      } catch (error) {
        setFlash(error.message);
      } finally {
        state.loading = false;
      }
    });
  });
}

async function loadPage(page) {
  state.loading = true;
  renderPager(state.page, state.totalPages);
  const list = document.getElementById("sample-list");
  try {
    const payload = await fetchSamples(page);
    renderStatus(payload.status);
    renderPager(payload.page, payload.total_pages);
    if (!payload.samples.length && payload.page > 1) {
      state.loading = false;
      return loadPage(payload.page - 1);
    }
    list.innerHTML = payload.samples.length ? payload.samples.map((sample) => buildSampleCard(sample)).join("") : "<p class=\"muted\">No samples on this page.</p>";
    payload.samples.forEach((sample) => {
      const card = list.querySelector(`[data-sample-id="${CSS.escape(sample.sample_id)}"]`);
      if (card) {
        bindSampleCard(card, sample);
      }
    });
    setFlash("");
  } catch (error) {
    list.innerHTML = "";
    setFlash(error.message);
  } finally {
    state.loading = false;
    renderPager(state.page, state.totalPages);
  }
}

document.getElementById("view-select").addEventListener("change", (event) => {
  state.view = event.target.value;
  loadPage(1);
});

document.getElementById("prev-page").addEventListener("click", () => {
  if (state.page > 1 && !state.loading) {
    loadPage(state.page - 1);
  }
});

document.getElementById("next-page").addEventListener("click", () => {
  if (state.page < state.totalPages && !state.loading) {
    loadPage(state.page + 1);
  }
});

loadPage(1);
