import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

const content = document.getElementById('content');
const navItems = [...document.querySelectorAll('.nav-item')];
const toggle = document.getElementById('themeToggle');

marked.setOptions({
  gfm: true,
  breaks: false,
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
    return hljs.highlightAuto(code).value;
  }
});

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('wenet-theme', theme);
  mermaid.initialize({
    startOnLoad: false,
    theme: theme === 'light' ? 'default' : 'dark',
    securityLevel: 'loose'
  });
}

applyTheme(localStorage.getItem('wenet-theme') || 'dark');

toggle.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  renderMermaid();
});

async function renderMermaid() {
  document.querySelectorAll('pre code.language-mermaid').forEach((code) => {
    const pre = code.parentElement;
    const wrapper = document.createElement('div');
    wrapper.className = 'mermaid';
    wrapper.textContent = code.textContent;
    pre.replaceWith(wrapper);
  });
  try { await mermaid.run({ querySelector: '.mermaid' }); } catch (e) { console.warn(e); }
}

async function loadNote(path, updateHash = true) {
  content.innerHTML = '<div class="loading">正在加载笔记…</div>';
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const md = await res.text();
    content.innerHTML = marked.parse(md);
    content.querySelectorAll('pre code:not(.language-mermaid)').forEach((el) => hljs.highlightElement(el));
    await renderMermaid();
    if (window.MathJax?.typesetPromise) await MathJax.typesetPromise([content]);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (updateHash) history.replaceState(null, '', `#${path}`);
  } catch (err) {
    content.innerHTML = `<div class="error"><h2>加载失败</h2><p>${err.message}</p></div>`;
  }
}

navItems.forEach((item) => {
  item.addEventListener('click', () => {
    navItems.forEach((n) => n.classList.remove('active'));
    item.classList.add('active');
    loadNote(item.dataset.note);
  });
});

const hashPath = decodeURIComponent(location.hash.replace(/^#/, ''));
const target = navItems.find((n) => n.dataset.note === hashPath) || navItems[0];
navItems.forEach((n) => n.classList.toggle('active', n === target));
loadNote(target.dataset.note, false);
