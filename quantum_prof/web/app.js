/* Quantum Professor — vanilla JS front-end */
'use strict';

const API = '';

// ── State ──────────────────────────────────────────────────────────────────
let allTopics = [];
let currentSection = 'topics';

// ── Helpers ────────────────────────────────────────────────────────────────
async function apiFetch(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

function el(id) { return document.getElementById(id); }

function renderKatex(container) {
  if (window.renderMathInElement) {
    window.renderMathInElement(container, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true },
      ],
      throwOnError: false,
    });
  }
}

function badgeClass(level) {
  return 'badge badge-' + (level || 'basics').replace(/\s/g, '_');
}

function showError(container, msg) {
  container.innerHTML = `<div class="error">${msg}</div>`;
}

// ── Navigation ─────────────────────────────────────────────────────────────
function navigate(section) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const sectionEl = el('section-' + section);
  if (sectionEl) sectionEl.classList.add('active');
  const navEl = document.querySelector(`[data-section="${section}"]`);
  if (navEl) navEl.classList.add('active');
  currentSection = section;
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => navigate(item.dataset.section));
});

// ── Topics Browser ─────────────────────────────────────────────────────────
async function loadTopics(level) {
  const out = el('topics-list');
  out.innerHTML = '<p class="loading">Loading topics…</p>';
  try {
    const url = level ? `/api/topics?level=${encodeURIComponent(level)}` : '/api/topics';
    const data = await apiFetch(url);
    allTopics = data.topics;
    renderTopicList(allTopics, out);
  } catch (e) {
    showError(out, e.message);
  }
}

function renderTopicList(topics, container) {
  if (!topics.length) { container.innerHTML = '<p class="loading">No topics found.</p>'; return; }
  container.innerHTML = topics.map(t => `
    <div class="card" style="cursor:pointer" onclick="openTopic('${t.id}')">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="card-title">${t.title}</span>
        <span class="${badgeClass(t.level)}">${t.level}</span>
      </div>
      <div class="card-meta">${t.prerequisites.length ? 'Prereqs: ' + t.prerequisites.join(', ') : 'No prerequisites'}</div>
      <div class="card-body">${t.intuition.slice(0, 100)}…</div>
    </div>
  `).join('');
}

async function openTopic(topicId) {
  navigate('topic-detail');
  const out = el('topic-detail-content');
  out.innerHTML = '<p class="loading">Loading…</p>';
  try {
    const t = await apiFetch(`/api/topic?id=${encodeURIComponent(topicId)}`);
    out.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <h2>${t.title}</h2>
        <span class="${badgeClass(t.level)}">${t.level}</span>
      </div>
      ${t.prerequisites.length ? `<p><strong>Prerequisites:</strong> ${t.prerequisites.map(p => `<a href="#" onclick="openTopic('${p}');return false">${p}</a>`).join(', ')}</p>` : ''}
      <h3>Intuition</h3>
      <p>${t.intuition}</p>
      <h3>Key Concepts</h3>
      <ul>${t.key_concepts.map(c => `<li>${c}</li>`).join('')}</ul>
      <h3>Key Equations</h3>
      ${t.key_equations.map(eq => `<div class="eq">$${eq}$</div>`).join('')}
      <h3>Recommended Books</h3>
      <p>${t.book_refs.join(', ')}</p>
      ${t.problems && t.problems.length ? `
        <h3>Practice Problems</h3>
        ${t.problems.map(p => `
          <div class="card problem-card">
            <div class="card-title">[${p.difficulty}] ${p.stem}</div>
            <div class="problem-hint">Hint: ${p.hint}</div>
            <button class="secondary" style="margin-top:8px;font-size:0.8rem;padding:4px 10px" onclick="toggleAnswer(this)">Show Answer</button>
            <div class="problem-answer">$${p.answer_latex}$</div>
          </div>
        `).join('')}
      ` : ''}
      <div style="margin-top:16px">
        <button class="secondary" onclick="navigate('topics')">← Back to Topics</button>
        <button onclick="loadPath('${t.id}')" style="margin-left:8px">View Learning Path</button>
      </div>
    `;
    renderKatex(out);
  } catch (e) {
    showError(out, e.message);
  }
}

function toggleAnswer(btn) {
  const ans = btn.nextElementSibling;
  ans.classList.toggle('visible');
  btn.textContent = ans.classList.contains('visible') ? 'Hide Answer' : 'Show Answer';
  if (ans.classList.contains('visible')) renderKatex(ans);
}

el('topic-level-filter').addEventListener('change', e => loadTopics(e.target.value || null));

// ── Learning Path ──────────────────────────────────────────────────────────
async function loadPath(topicId) {
  navigate('path');
  const out = el('path-content');
  out.innerHTML = '<p class="loading">Computing path…</p>';
  try {
    const data = await apiFetch(`/api/path?id=${encodeURIComponent(topicId)}`);
    out.innerHTML = `
      <h2>Path to: ${data.target}</h2>
      <p style="color:var(--muted);margin-bottom:16px">${data.path.length} steps</p>
      ${data.path.map((t, i) => `
        <div class="path-step">
          <div class="path-num">${i + 1}</div>
          <div class="path-info">
            <div class="step-title" style="cursor:pointer" onclick="openTopic('${t.id}')">${t.title}</div>
            <div class="step-level">${t.level}</div>
            <div class="step-concepts">${t.key_concepts.slice(0, 2).join(' · ')}</div>
          </div>
        </div>
      `).join('')}
    `;
    renderKatex(out);
  } catch (e) {
    showError(out, e.message);
  }
}

el('path-topic-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') loadPath(el('path-topic-input').value.trim());
});
el('path-go-btn').addEventListener('click', () => loadPath(el('path-topic-input').value.trim()));

// Path topic suggestions datalist
function buildTopicDatalist() {
  const dl = el('topics-datalist');
  if (!dl) return;
  dl.innerHTML = allTopics.map(t => `<option value="${t.id}">${t.title}</option>`).join('');
}

// ── Books ──────────────────────────────────────────────────────────────────
async function loadBooks(level) {
  const out = el('books-list');
  out.innerHTML = '<p class="loading">Loading books…</p>';
  try {
    const url = level ? `/api/books?level=${encodeURIComponent(level)}` : '/api/books';
    const data = await apiFetch(url);
    out.innerHTML = data.books.map(b => `
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <span class="card-title">${b.title}</span>
          <span class="${badgeClass(b.level)}" style="margin-left:8px;flex-shrink:0">${b.level}</span>
        </div>
        <div class="card-meta">${b.authors.join(', ')}</div>
        <div class="card-body">${b.why_read}</div>
        <div class="card-meta" style="margin-top:6px">Topics: ${b.topics.join(', ')}</div>
      </div>
    `).join('');
  } catch (e) {
    showError(out, e.message);
  }
}

el('book-level-filter').addEventListener('change', e => loadBooks(e.target.value || null));

// ── Physics Solver ─────────────────────────────────────────────────────────
const SOLVER_PARAMS = {
  'particle-in-a-box': [
    { name: 'n', label: 'Quantum number n', default: '1', type: 'number' },
    { name: 'L', label: 'Well width L (m)', default: '1e-9', type: 'text' },
  ],
  'harmonic-oscillator': [
    { name: 'n', label: 'Quantum number n', default: '0', type: 'number' },
    { name: 'omega', label: 'Angular freq ω (rad/s)', default: '1e14', type: 'text' },
  ],
  'hydrogen-atom': [
    { name: 'n', label: 'Principal quantum number n', default: '1', type: 'number' },
  ],
  'hydrogen-transition': [
    { name: 'n_i', label: 'Initial level n_i', default: '3', type: 'number' },
    { name: 'n_f', label: 'Final level n_f', default: '2', type: 'number' },
  ],
  'de-broglie': [
    { name: 'mass_kg', label: 'Mass (kg)', default: '9.109e-31', type: 'text' },
    { name: 'speed_m_s', label: 'Speed (m/s)', default: '1e6', type: 'text' },
  ],
  'photon-energy': [
    { name: 'wavelength_nm', label: 'Wavelength (nm)', default: '500', type: 'number' },
  ],
  'uncertainty-principle': [
    { name: 'delta_x_m', label: 'Position uncertainty Δx (m)', default: '1e-10', type: 'text' },
  ],
  'quantum-information': [
    { name: 'gate', label: 'Gate (I/X/Y/Z/H/S/T)', default: 'H', type: 'text' },
    { name: 'alpha', label: 'α (real part of |0⟩ amplitude)', default: '1', type: 'number' },
    { name: 'beta', label: 'β (real part of |1⟩ amplitude)', default: '0', type: 'number' },
  ],
};

function updateSolverParams() {
  const topic = el('solver-topic').value;
  const params = SOLVER_PARAMS[topic] || [];
  const container = el('solver-params');
  container.innerHTML = params.map(p => `
    <div class="form-group">
      <label>${p.label}</label>
      <input type="${p.type}" id="param-${p.name}" value="${p.default}" style="width:180px">
    </div>
  `).join('');
}

el('solver-topic').addEventListener('change', updateSolverParams);

el('solver-run-btn').addEventListener('click', async () => {
  const topic = el('solver-topic').value;
  const params = SOLVER_PARAMS[topic] || [];
  const out = el('solver-result');

  let url = `/api/solve?topic=${encodeURIComponent(topic)}`;
  for (const p of params) {
    const val = el('param-' + p.name)?.value || '';
    if (val) url += `&${p.name}=${encodeURIComponent(val)}`;
  }

  out.innerHTML = '<p class="loading">Computing…</p>';
  try {
    const data = await apiFetch(url);
    if (data.error) { showError(out, data.error); return; }
    out.innerHTML = Object.entries(data).map(([k, v]) => `
      <div class="result-row">
        <span class="result-key">${k}</span>
        <span class="result-val">${typeof v === 'object' ? JSON.stringify(v) : v}</span>
      </div>
    `).join('');
  } catch (e) {
    showError(out, e.message);
  }
});

// ── Init ───────────────────────────────────────────────────────────────────
async function init() {
  updateSolverParams();
  await loadTopics();
  buildTopicDatalist();
  await loadBooks();
  navigate('topics');
}

init().catch(console.error);
