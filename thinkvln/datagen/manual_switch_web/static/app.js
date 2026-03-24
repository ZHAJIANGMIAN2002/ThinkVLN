const state = {
  loading: false,
  token: localStorage.getItem("manualSwitchToken") || "",
  user: null,
  sample: null,
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

function renderStatus(status = {}) {
  document.getElementById("total-count").textContent = String(status.total);
  document.getElementById("labeled-count").textContent = String(status.labeled);
  document.getElementById("remaining-count").textContent = String(status.remaining);
  document.getElementById("mine-done-count").textContent = String(status.mine_done || 0);
}

function authHeaders() {
  if (!state.token) {
    return {};
  }
  return { Authorization: `Bearer ${state.token}` };
}

async function postJson(url, body, withAuth = false) {
  const headers = { "Content-Type": "application/json" };
  if (withAuth) {
    Object.assign(headers, authHeaders());
  }
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `request failed: ${response.status}`);
  }
  return response.json();
}

async function getJson(url, withAuth = false) {
  const response = await fetch(url, {
    headers: withAuth ? authHeaders() : {},
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `request failed: ${response.status}`);
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

function applyAuthState() {
  const authed = Boolean(state.user);
  document.getElementById("auth-panel").hidden = authed;
  document.getElementById("workspace-panel").hidden = !authed;
  document.getElementById("admin-panel").hidden = !authed || state.user.role !== "admin";
  document.getElementById("current-user").textContent = authed ? state.user.username : "guest";
  document.getElementById("current-role").textContent = authed ? state.user.role : "-";
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
          <span class="label">Current Task</span>
          <p>${escapeHtml(sample.active_subtask)}</p>
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
        const payload = await postJson(`/api/tasks/${encodeURIComponent(sample.sample_id)}/annotate`, { should_switch: shouldSwitch }, true);
        renderStatus(payload.status);
        await claimNext();
        setFlash("");
      } catch (error) {
        setFlash(error.message);
      } finally {
        state.loading = false;
      }
    });
  });
}

function renderSample(sample) {
  state.loading = true;
  const list = document.getElementById("sample-list");
  if (!sample) {
    list.innerHTML = "<p class=\"muted\">No pending tasks in queue.</p>";
    return;
  }
  list.innerHTML = buildSampleCard(sample);
  const card = list.querySelector(`[data-sample-id="${CSS.escape(sample.sample_id)}"]`);
  if (card) {
    bindSampleCard(card, sample);
  }
}

async function claimNext() {
  const list = document.getElementById("sample-list");
  try {
    list.innerHTML = "<p class=\"muted\">Claiming next task...</p>";
    const payload = await postJson("/api/tasks/claim", {}, true);
    renderStatus(payload.status);
    state.sample = payload.sample;
    renderSample(payload.sample);
    setFlash("");
  } catch (error) {
    list.innerHTML = "";
    setFlash(error.message);
  } finally {
    state.loading = false;
  }
}

async function refreshMe() {
  if (!state.token) {
    state.user = null;
    state.sample = null;
    applyAuthState();
    renderStatus({ total: 0, labeled: 0, remaining: 0, mine_done: 0 });
    renderSample(null);
    return;
  }
  try {
    const payload = await getJson("/api/me", true);
    state.user = payload.user;
    applyAuthState();
    renderStatus(payload.status || {});
    await claimNext();
    if (state.user.role === "admin") {
      await refreshAdminUsers();
    }
  } catch (error) {
    localStorage.removeItem("manualSwitchToken");
    state.token = "";
    state.user = null;
    applyAuthState();
    setFlash(error.message);
  }
}

async function refreshAdminUsers() {
  if (!state.user || state.user.role !== "admin") {
    return;
  }
  const box = document.getElementById("admin-users");
  try {
    const payload = await getJson("/api/admin/users", true);
    const rows = payload.users || [];
    if (!rows.length) {
      box.innerHTML = "<p class=\"muted\">No users.</p>";
      return;
    }
    box.innerHTML = `
      <table class="user-table">
        <thead><tr><th>Username</th><th>Role</th><th>Active</th><th>Done</th></tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${escapeHtml(row.username)}</td>
              <td>${escapeHtml(row.role)}</td>
              <td>${row.is_active ? "yes" : "no"}</td>
              <td>${escapeHtml(row.done_count)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  } catch (error) {
    box.innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  }
}

document.getElementById("login-btn").addEventListener("click", async () => {
  try {
    setFlash("Logging in...");
    const payload = await postJson("/api/auth/login", {
      username: document.getElementById("login-username").value.trim(),
      password: document.getElementById("login-password").value,
    });
    state.token = payload.token;
    localStorage.setItem("manualSwitchToken", state.token);
    state.user = payload.user;
    applyAuthState();
    setFlash("");
    await refreshMe();
  } catch (error) {
    setFlash(error.message);
  }
});

document.getElementById("register-btn").addEventListener("click", async () => {
  try {
    setFlash("Registering...");
    await postJson("/api/auth/register", {
      username: document.getElementById("register-username").value.trim(),
      password: document.getElementById("register-password").value,
      invite_code: document.getElementById("register-invite").value.trim(),
    });
    setFlash("Register success. Please login.");
  } catch (error) {
    setFlash(error.message);
  }
});

document.getElementById("claim-btn").addEventListener("click", async () => {
  await claimNext();
});

document.getElementById("refresh-btn").addEventListener("click", async () => {
  await refreshMe();
});

document.getElementById("logout-btn").addEventListener("click", () => {
  localStorage.removeItem("manualSwitchToken");
  state.token = "";
  state.user = null;
  state.sample = null;
  applyAuthState();
  renderSample(null);
  renderStatus({ total: 0, labeled: 0, remaining: 0, mine_done: 0 });
  setFlash("");
});

document.getElementById("create-invite-btn").addEventListener("click", async () => {
  try {
    const payload = await postJson("/api/admin/invites", {
      remaining_uses: Number(document.getElementById("invite-uses").value || 1),
      expires_in_days: document.getElementById("invite-days").value
        ? Number(document.getElementById("invite-days").value)
        : null,
    }, true);
    const invite = payload.invite;
    document.getElementById("invite-output").textContent = `Invite code: ${invite.code} (uses=${invite.remaining_uses})`;
    await refreshAdminUsers();
  } catch (error) {
    setFlash(error.message);
  }
});

applyAuthState();
refreshMe();
