  import init, {
    alloc, dealloc, wasm_memory,
    to_grayscale_ptr, gaussian_blur_ptr, sobel_edges_ptr,
    module_info,
  } from './pkg/image_wasm.js';

  const MAX_DIM = 1600;

  const els = {
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    regenBtn: document.getElementById('regenBtn'),
    gallery: document.getElementById('gallery'),
    sigmaRange: document.getElementById('sigmaRange'),
    sigmaValue: document.getElementById('sigmaValue'),
    thresholdRange: document.getElementById('thresholdRange'),
    thresholdValue: document.getElementById('thresholdValue'),
    runsRange: document.getElementById('runsRange'),
    runsValue: document.getElementById('runsValue'),
    runBtn: document.getElementById('runBtn'),
    statusNote: document.getElementById('statusNote'),
    resultsGrid: document.getElementById('resultsGrid'),
    resultsTableBody: document.getElementById('resultsTableBody'),
    initTime: document.getElementById('initTime'),
    moduleInfo: document.getElementById('moduleInfo'),
    moduleStatus: document.getElementById('moduleStatus'),
  };

  const STAGES = [
    { key: 'original', label: 'Original', dotColor: 'var(--ink-faint)' },
    { key: 'gray', label: 'Escala de cinza', dotColor: '#6B7A8A' },
    { key: 'blur', label: 'Desfoque gaussiano', dotColor: '#3E7CB8' },
    { key: 'sobel', label: 'Sobel (bordas)', dotColor: 'var(--accent)' },
  ];

  const state = { images: [], selectedId: null, busy: false, wasmReady: false };

  let wasmMemory = null;

  function stats(times) {
    const n = times.length;
    const avg = times.reduce((a, b) => a + b, 0) / n;
    const min = Math.min(...times);
    const max = Math.max(...times);
    const variance = times.reduce((a, b) => a + (b - avg) ** 2, 0) / n;
    return { avg, min, max, std: Math.sqrt(variance), n };
  }

  function writeToWasm(ptr, len, bytes) {
    new Uint8Array(wasmMemory.buffer, ptr, len).set(bytes);
  }

  function readFromWasm(ptr, len) {
    return new Uint8ClampedArray(wasmMemory.buffer.slice(ptr, ptr + len));
  }

  function benchmarkPtr(fn, srcPtr, dstPtr, width, height, args, runs) {
    const times = [];
    for (let i = 0; i < runs; i++) {
      const t0 = performance.now();
      fn(srcPtr, dstPtr, width, height, ...args);
      const t1 = performance.now();
      times.push(t1 - t0);
    }
    return stats(times);
  }

  const fmt = (n) => (n < 0.005 ? '< 0.01' : n.toFixed(2));

  function generateSampleImage(w, h, seed) {
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    let s = seed;
    const rand = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return (s / 0x7fffffff); };

    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, `hsl(${Math.floor(rand() * 360)}, 45%, 88%)`);
    grad.addColorStop(1, `hsl(${Math.floor(rand() * 360)}, 55%, 62%)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    const shapes = 6 + Math.floor(rand() * 4);
    for (let i = 0; i < shapes; i++) {
      ctx.fillStyle = `hsla(${Math.floor(rand() * 360)}, 70%, ${30 + rand() * 35}%, 0.85)`;
      const cx = rand() * w, cy = rand() * h, r = 20 + rand() * (Math.min(w, h) * 0.18);
      const kind = Math.floor(rand() * 3);
      ctx.beginPath();
      if (kind === 0) { ctx.arc(cx, cy, r, 0, Math.PI * 2); }
      else if (kind === 1) { ctx.rect(cx - r, cy - r * 0.6, r * 2, r * 1.2); }
      else { ctx.moveTo(cx, cy - r); ctx.lineTo(cx + r, cy + r); ctx.lineTo(cx - r, cy + r); ctx.closePath(); }
      ctx.fill();
    }

    ctx.strokeStyle = 'rgba(20,20,25,0.25)';
    ctx.lineWidth = 1.5;
    const spacing = Math.max(24, Math.round(w / 20));
    for (let x = spacing; x < w; x += spacing) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }

    const imgData = ctx.getImageData(0, 0, w, h);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const n = (rand() - 0.5) * 22;
      d[i] = Math.min(255, Math.max(0, d[i] + n));
      d[i + 1] = Math.min(255, Math.max(0, d[i + 1] + n));
      d[i + 2] = Math.min(255, Math.max(0, d[i + 2] + n));
    }
    ctx.putImageData(imgData, 0, 0);

    ctx.fillStyle = 'rgba(15,15,20,0.55)';
    ctx.font = `600 ${Math.round(h * 0.09)}px "IBM Plex Mono", monospace`;
    ctx.fillText('TEST', w * 0.06, h * 0.92);

    return canvas;
  }

  function addImage(source, name, w, h) {
    const thumbCanvas = document.createElement('canvas');
    const tw = 96, th = 96;
    thumbCanvas.width = tw; thumbCanvas.height = th;
    const tctx = thumbCanvas.getContext('2d');
    const scale = Math.max(tw / w, th / h);
    const sw = tw / scale, sh = th / scale;
    tctx.drawImage(source, (w - sw) / 2, (h - sh) / 2, sw, sh, 0, 0, tw, th);

    const entry = { id: 'img_' + Math.random().toString(36).slice(2, 10), name, source, width: w, height: h, thumbUrl: thumbCanvas.toDataURL('image/png') };
    state.images.push(entry);
    state.selectedId = entry.id;
    renderGallery();
    renderOriginal();
    clearResults();
  }

  function loadFile(file) {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => addImage(img, file.name, img.naturalWidth, img.naturalHeight);
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function buildGrid() {
    els.resultsGrid.innerHTML = '';
    STAGES.forEach((stage, i) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="card-head"><h3>${stage.label}</h3><span class="step-num">${i === 0 ? 'entrada' : '0' + i}</span></div>
        <div class="card-canvas-wrap" id="wrap-${stage.key}"><span class="placeholder">Nenhuma imagem processada ainda</span></div>
        <div class="card-foot"><span id="dims-${stage.key}">—</span><span class="time-badge idle" id="time-${stage.key}">${i === 0 ? '' : 'não processado'}</span></div>
      `;
      els.resultsGrid.appendChild(card);
    });
  }

  function paintStage(key, imageData) {
    const wrap = document.getElementById('wrap-' + key);
    wrap.innerHTML = '';
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width; canvas.height = imageData.height;
    canvas.setAttribute('role', 'img');
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    wrap.appendChild(canvas);
    document.getElementById('dims-' + key).textContent = imageData.width + ' × ' + imageData.height + ' px';
  }

  function renderOriginal() {
    const img = getSelected();
    if (!img) return;
    buildGrid();
    const canvas = document.createElement('canvas');
    canvas.width = img.width; canvas.height = img.height;
    canvas.getContext('2d').drawImage(img.source, 0, 0, img.width, img.height);
    const wrap = document.getElementById('wrap-original');
    wrap.innerHTML = '';
    wrap.appendChild(canvas);
    document.getElementById('dims-original').textContent = img.width + ' × ' + img.height + ' px';
    document.getElementById('time-original').textContent = img.name;
    document.getElementById('time-original').classList.add('idle');
  }

  function renderGallery() {
    els.gallery.innerHTML = '';
    state.images.forEach((img) => {
      const btn = document.createElement('button');
      btn.className = 'thumb' + (img.id === state.selectedId ? ' active' : '');
      btn.type = 'button';
      btn.setAttribute('aria-label', 'Selecionar ' + img.name);
      btn.innerHTML = `<img src="${img.thumbUrl}" alt=""><span class="thumb-tag">${img.width}×${img.height}</span>`;
      btn.addEventListener('click', () => { state.selectedId = img.id; renderGallery(); renderOriginal(); clearResults(); });
      els.gallery.appendChild(btn);
    });
  }

  function clearResults() {
    els.resultsTableBody.innerHTML = '<tr class="empty-row"><td colspan="6">Clique em “Processar imagem” para medir os três algoritmos.</td></tr>';
    ['gray', 'blur', 'sobel'].forEach((key) => {
      const wrap = document.getElementById('wrap-' + key);
      if (wrap) wrap.innerHTML = '<span class="placeholder">Nenhuma imagem processada ainda</span>';
      const t = document.getElementById('time-' + key);
      if (t) { t.textContent = 'não processado'; t.classList.add('idle'); }
      const d = document.getElementById('dims-' + key);
      if (d) d.textContent = '—';
    });
  }

  function getSelected() { return state.images.find((i) => i.id === state.selectedId) || null; }

  function setStatus(text, mode) {
    const busy = mode === 'busy';
    els.statusNote.innerHTML = (busy ? '<span class="dot"></span>' : '') + text;
    els.statusNote.classList.toggle('busy', busy);
    els.statusNote.classList.toggle('error', mode === 'error');
  }

  function getScaledImageData(img) {
    let { width: w, height: h } = img;
    if (Math.max(w, h) > MAX_DIM) {
      const scale = MAX_DIM / Math.max(w, h);
      w = Math.round(w * scale); h = Math.round(h * scale);
    }
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img.source, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  }

  function addTableRow(label, dotColor, s) {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><span class="algo-name"><span class="algo-dot" style="background:${dotColor}"></span>${label}</span></td>
      <td class="num">${fmt(s.avg)}</td><td class="num">${fmt(s.min)}</td><td class="num">${fmt(s.max)}</td>
      <td class="num">${fmt(s.std)}</td><td class="num">${s.n}</td>
    `;
    els.resultsTableBody.appendChild(row);
  }

  function yieldFrame() { return new Promise((resolve) => setTimeout(resolve, 0)); }

  async function runPipeline() {
    const img = getSelected();
    if (!img || state.busy || !state.wasmReady) return;
    state.busy = true;
    els.runBtn.disabled = true;
    els.resultsTableBody.innerHTML = '';

    const runs = parseInt(els.runsRange.value, 10);
    const sigma = parseFloat(els.sigmaRange.value);
    const threshold = parseInt(els.thresholdRange.value, 10);

    setStatus('Carregando pixels da imagem…', 'busy');
    await yieldFrame();
    const base = getScaledImageData(img);
    const len = base.width * base.height * 4;

    let srcPtr = null;
    let dstPtr = null;

    try {
      srcPtr = alloc(len);
      dstPtr = alloc(len);
      writeToWasm(srcPtr, len, base.data);

      setStatus('Executando escala de cinza (' + runs + '×)…', 'busy');
      await yieldFrame();
      const grayRes = benchmarkPtr(to_grayscale_ptr, srcPtr, dstPtr, base.width, base.height, [], runs);
      paintStage('gray', new ImageData(readFromWasm(dstPtr, len), base.width, base.height));
      document.getElementById('time-gray').textContent = fmt(grayRes.avg) + ' ms';
      document.getElementById('time-gray').classList.remove('idle');
      addTableRow('Escala de cinza', '#6B7A8A', grayRes);
      await yieldFrame();

      setStatus('Executando desfoque gaussiano (' + runs + '×)…', 'busy');
      await yieldFrame();
      const blurRes = benchmarkPtr(gaussian_blur_ptr, srcPtr, dstPtr, base.width, base.height, [sigma], runs);
      paintStage('blur', new ImageData(readFromWasm(dstPtr, len), base.width, base.height));
      document.getElementById('time-blur').textContent = fmt(blurRes.avg) + ' ms';
      document.getElementById('time-blur').classList.remove('idle');
      addTableRow('Desfoque gaussiano (σ=' + sigma.toFixed(1) + ')', '#3E7CB8', blurRes);
      await yieldFrame();

      setStatus('Executando detecção de bordas — Sobel (' + runs + '×)…', 'busy');
      await yieldFrame();
      const sobelRes = benchmarkPtr(sobel_edges_ptr, srcPtr, dstPtr, base.width, base.height, [threshold], runs);
      paintStage('sobel', new ImageData(readFromWasm(dstPtr, len), base.width, base.height));
      document.getElementById('time-sobel').textContent = fmt(sobelRes.avg) + ' ms';
      document.getElementById('time-sobel').classList.remove('idle');
      addTableRow('Sobel (bordas)', 'var(--accent)', sobelRes);

      const total = grayRes.avg + blurRes.avg + sobelRes.avg;
      setStatus('Concluído — ' + base.width + '×' + base.height + ' px · pipeline completo em ' + fmt(total) + ' ms (médias somadas).', 'ok');
    } catch (err) {
      console.error(err);
      setStatus('Erro ao chamar o módulo Wasm — veja o console (F12) para detalhes.', 'error');
    } finally {
      if (srcPtr !== null) dealloc(srcPtr, len);
      if (dstPtr !== null) dealloc(dstPtr, len);
    }

    state.busy = false;
    els.runBtn.disabled = false;
  }

  els.dropzone.addEventListener('click', () => els.fileInput.click());
  els.dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); els.fileInput.click(); } });
  ['dragenter', 'dragover'].forEach((evt) => els.dropzone.addEventListener(evt, (e) => { e.preventDefault(); els.dropzone.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach((evt) => els.dropzone.addEventListener(evt, (e) => { e.preventDefault(); els.dropzone.classList.remove('drag-over'); }));
  els.dropzone.addEventListener('drop', (e) => Array.from(e.dataTransfer.files || []).forEach(loadFile));
  els.fileInput.addEventListener('change', (e) => { Array.from(e.target.files || []).forEach(loadFile); els.fileInput.value = ''; });

  els.regenBtn.addEventListener('click', () => {
    const seed = Math.floor(Math.random() * 1e9) || 1;
    addImage(generateSampleImage(640, 480, seed), 'Padrão de teste (gerado)', 640, 480);
  });

  els.sigmaRange.addEventListener('input', () => { els.sigmaValue.textContent = parseFloat(els.sigmaRange.value).toFixed(1); });
  els.thresholdRange.addEventListener('input', () => { const v = parseInt(els.thresholdRange.value, 10); els.thresholdValue.textContent = v === 0 ? '0 (desativado)' : String(v); });
  els.runsRange.addEventListener('input', () => { els.runsValue.textContent = els.runsRange.value; });

  els.runBtn.addEventListener('click', runPipeline);

  // ---------- Init ----------
  buildGrid();
  addImage(generateSampleImage(640, 480, 42), 'Imagem de exemplo (gerada)', 640, 480);

  (async () => {
    try {
      const t0 = performance.now();
      await init(); 
      wasmMemory = wasm_memory();
      const t1 = performance.now();
      els.initTime.innerHTML = (t1 - t0).toFixed(2) + '<span class="unit">ms</span>';
      els.moduleInfo.textContent = module_info();
      els.moduleStatus.textContent = 'pronto · memória compartilhada (zero-copy)';
      state.wasmReady = true;
      els.runBtn.disabled = false;
      els.runBtn.textContent = 'Processar imagem';
      setStatus('Pronto — imagem de exemplo carregada.', 'ok');
    } catch (err) {
      console.error(err);
      els.moduleStatus.textContent = 'falha ao carregar';
      setStatus('Não foi possível carregar ./pkg/image_wasm.js — rode "wasm-pack build --target web" e sirva a pasta por HTTP (veja README.md).', 'error');
    }
  })();
