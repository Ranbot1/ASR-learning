# WeNet Learning Notes Site

一个用于记录 WeNet / ASR 第一周学习内容的 GitHub Pages 静态站点。

## 内容

- Day 1：Waveform、分帧、加窗、FFT、Mel、Log-Mel FBank
- Day 2：CTC、Forward-Backward、CTC Head、Greedy / Prefix Beam Search、Attention Decoder、联合训练、Attention Rescoring
- Day 3：Self-Attention、Q/K/V、Scaled Dot-Product Attention、Multi-Head Attention

## GitHub Pages 部署

推荐仓库名：`Ranbot1.github.io`

把本目录全部文件放在仓库 `main` 分支根目录，Pages 使用根目录发布即可。

站点入口：`index.html`

## 本地预览

由于页面通过 `fetch()` 加载 Markdown，不建议直接双击 `index.html`。可在目录内运行：

```bash
python -m http.server 8000
```

浏览器打开：

```text
http://localhost:8000
```

## 技术

- 原生 HTML / CSS / JavaScript
- marked.js：Markdown 渲染
- MathJax：数学公式
- Mermaid：思维导图
- highlight.js：代码高亮
