/* Dashboard transport and views; all displayed events come from the same-origin service. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = { token: '', status: null, photos: [], cursor: null, filter: 'all', range: 'all',
    query: '', paused: false, preview: null, selected: null, original: false, photoRequest: 0, photoBusy: false, connected: false };
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const number = (value, places = 2) => Number.isFinite(value) ? value.toFixed(places) : '—';
  const date = (value) => value ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : '—';
  const mode = (value) => value === 'live' ? '实机' : '音频回放';
  const setText = (id, value) => { $(id).textContent = value; };

  async function api(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch(path, { cache: 'no-store', signal: controller.signal,
        ...(body ? { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Control-Token': state.token }, body: JSON.stringify(body) } : {}) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      return result;
    } finally { clearTimeout(timeout); }
  }

  let errorTimer;
  function showError(message) {
    setText('service-message', message);
    $('service-message').style.color = 'var(--danger)';
    $('toast').textContent = message;
    $('toast').hidden = false;
    clearTimeout(errorTimer);
    errorTimer = setTimeout(() => { $('toast').hidden = true; }, 4000);
  }

  function showPreview(status) {
    if (state.paused) return;
    const preview = status?.preview;
    const image = $('preview-image');
    if (!preview) {
      image.hidden = true;
      $('frame-empty').hidden = false;
      const message = status?.run?.mode === 'replay' ? '音频回放没有同步相机图片，请查看实时事件。' : '未收到新鲜相机缓存。相机未启用、尚未取到帧或当前会话已结束。';
      setText('frame-empty', message);
      setText('frame-state', '无画面');
      setText('live-badge', '无画面');
      setText('feed-clock-date', '—'); setText('feed-clock-time', '');
      setText('preview-meta', '不以示例图片替代设备画面');
      state.preview = null;
      return;
    }
    if (preview.sequence === state.preview) return;
    state.preview = preview.sequence;
    image.onload = () => { if (!state.paused && state.connected && state.status?.preview?.sequence === preview.sequence) { image.hidden = false; $('frame-empty').hidden = true; } };
    image.onerror = () => { image.hidden = true; $('frame-empty').hidden = false; setText('frame-empty', '相机图片暂不可用'); setText('frame-state', '图片不可用'); setText('live-badge', '无画面'); };
    image.src = `${preview.url}?sequence=${encodeURIComponent(preview.sequence)}`;
    setText('frame-state', '相机缓存预览');
    setText('live-badge', 'PREVIEW');
    setText('preview-meta', `低频 JPEG 预览 · 更新于 ${date(preview.timestamp)} · 曝光时间未验证`);
    const stamp = new Date(preview.timestamp);
    setText('feed-clock-date', stamp.toLocaleDateString('zh-CN'));
    setText('feed-clock-time', stamp.toLocaleTimeString('zh-CN', { hour12: false }));
  }

  function renderStatus(data) {
    state.status = data;
    state.token = data.control_token;
    state.connected = true;
    const run = data.run;
    const latest = data.latest;
    setText('connection-state', '后端已连接');
    setText('mode-label', run ? `${mode(run.mode)} · ${run.status}` : '未启动');
    setText('device-state', data.active ? '运行中' : '未采集');
    setText('camera-state', run?.mode === 'replay' ? '回放未接相机' : data.preview ? '已收到 JPEG 缓存' : data.vision_enabled ? '未收到新鲜画面' : '未启用 / 待标定');
    setText('audio-state', data.active ? (run.mode === 'live' ? '实机采集中 · 待持续验收' : '14路 / 48 kHz 回放') : '采集未运行');
    setText('inference-state', latest ? `${number(latest.processing_seconds * 1000, 1)} ms / 窗` : '等待分析窗口');
    setText('storage-state', '本地持久保存 · 不自动清理');
    setText('operating-point', `${data.operating_point.confirm}窗确认 · 门限 ${number(data.operating_point.threshold)}`);
    setText('metric-today', data.metrics.photos_today);
    setText('metric-horn', data.metrics.horns_today);
    setText('metric-gaps', latest ? `${latest.capture_gaps ?? 0} / ${latest.skipped_windows ?? 0}` : '—');
    setText('readout-bearing', Number.isFinite(latest?.azimuth) ? `${number(latest.azimuth, 0)}°` : '—');
    setText('readout-confidence', number(latest?.score));
    setText('readout-latency', latest ? `${number(latest.processing_seconds, 3)} s` : '—');
    $('start-btn').disabled = Boolean(data.active);
    $('stop-btn').disabled = !data.active || run?.status === 'stopping';
    $('service-message').style.color = 'var(--muted)';
    setText('service-message', run?.error || (run ? `${mode(run.mode)}会话 ${run.id.slice(0, 8)} · ${run.status}。处理耗时不含采集等待与相机保存；声源点不代表车辆身份。` : '选择音频回放，或在 PYNQ-Z2 上启动实机采集。'));
    showPreview(data);
  }

  function renderEvents(rows) {
    $('event-list').innerHTML = rows.length ? rows.slice(0, 4).map((row) => {
      const event = row.event;
      const invalid = event.input_clipped || event.activation_clipped || event.frontend_clipped;
      const label = invalid ? '无效音频，未确认' : event.horn_confirmed ? '鸣笛证据确认' : event.capture_candidate ? '未确认候选' : '未达鸣笛门限';
      return `<div class="event-item"><span class="event-time">${escape(date(row.timestamp))}</span><span class="event-main"><span class="event-title">${label}</span><span class="event-meta">${mode(row.mode)} · 分数 ${number(event.score)} · 音频 ${number(event.sample_first / 48000)}–${number(event.sample_end_exclusive / 48000)} s · ${escape(event.acoustic_state)}</span></span></div>`;
    }).join('') : '<p class="history-sub">暂无分析记录。启动回放可验证接口与算法。</p>';
  }

  function renderPhotos(total) {
    $('snap-grid').innerHTML = state.photos.map((item) => `<button class="snap-card" type="button" data-photo="${item.id}" aria-label="查看抓拍 ${item.id}"><span class="snap-media" style="aspect-ratio:16/9"><img src="${item.image}" alt="声源候选点图" loading="lazy"><span class="snap-kind ${item.stage === 'confirmed' ? 'is-horn' : 'is-candidate'}">${item.stage === 'confirmed' ? '鸣笛证据' : '未确认'}</span></span><span class="snap-body"><span class="snap-title">${escape(date(item.timestamp))}</span><span class="snap-meta">${escape(item.camera)} · ${mode(item.mode)} · 分数 ${number(item.score)}</span></span></button>`).join('');
    $('snap-grid').querySelectorAll('img').forEach((image) => { image.onerror = () => { image.hidden = true; const note = document.createElement('span'); note.className = 'snap-missing'; note.textContent = '图片文件不可用'; image.parentElement.append(note); }; });
    $('snap-grid').hidden = !state.photos.length;
    $('empty-state').hidden = Boolean(state.photos.length);
    setText('result-count', `共 ${total} 张抓拍 · 已加载 ${state.photos.length} 张`);
    $('load-more').parentElement.hidden = !state.cursor;
    $('load-more').disabled = state.photoBusy;
    setText('count-all', state.filter === 'all' ? total : '—');
    setText('count-horn', state.filter === 'horn' ? total : '—');
    setText('count-candidate', state.filter === 'candidate' ? total : '—');
  }

  async function loadPhotos(append = false) {
    if (append && !state.cursor) return;
    const request = ++state.photoRequest;
    state.photoBusy = true;
    const query = new URLSearchParams({ kind: state.filter, range: state.range, q: state.query, limit: '9' });
    if (append) query.set('cursor', state.cursor);
    try {
      const result = await api(`/api/photos?${query}`);
      if (request !== state.photoRequest) return;
      state.photos = append ? [...state.photos, ...result.items] : result.items;
      state.cursor = result.next_cursor;
      renderPhotos(result.total);
    } catch (error) { if (request === state.photoRequest) showError(`抓拍查询失败：${error.message}`); }
    finally { if (request === state.photoRequest) { state.photoBusy = false; $('load-more').disabled = false; } }
  }

  function detail(item) {
    state.selected = item;
    setText('detail-title', `声源点图 #${item.id}`);
    setText('detail-sub', `${mode(item.mode)} · 会话 ${item.run.slice(0, 8)} · 事件 ${item.event_id}`);
    setText('detail-time', date(item.timestamp));
    setText('detail-location', '广角相机监控范围');
    setText('detail-camera', item.camera);
    setText('detail-kind', item.stage === 'confirmed' ? '声学鸣笛证据确认' : '未确认声学候选');
    setText('detail-bearing', Number.isFinite(item.azimuth) ? `${item.azimuth}°` : '方位不确定');
    setText('detail-confidence', number(item.score));
    setText('detail-caption', `${mode(item.mode)} · ${date(item.timestamp)} · 车辆归属与曝光同步未验证`);
    setText('detail-evidence', `音频采样区间：${item.sample_first}–${item.sample_end}\n${JSON.stringify(item.evidence?.projections || {}, null, 2)}`);
    $('detail-image').hidden = false;
    $('detail-image').onerror = () => { $('detail-image').hidden = true; setText('detail-caption', '图片文件不可用，事件记录仍保留'); };
    const image = state.original ? item.original : item.image;
    $('detail-image').src = image;
    $('download-photo').href = image;
    setText('toggle-original', state.original ? '查看声源点图' : '查看原图');
    $('detail-prev').disabled = state.photos.length < 2;
    $('detail-next').disabled = state.photos.length < 2;
  }

  $('snap-grid').addEventListener('click', (event) => {
    const button = event.target.closest('[data-photo]');
    if (!button) return;
    state.original = false;
    detail(state.photos.find((item) => item.id === Number(button.dataset.photo)));
    $('detail-dialog').showModal();
  });
  $('toggle-original').onclick = () => { state.original = !state.original; detail(state.selected); };
  $('detail-close').onclick = () => $('detail-dialog').close();
  $('detail-dialog').onclick = (event) => { if (event.target === $('detail-dialog')) $('detail-dialog').close(); };
  for (const [id, delta] of [['detail-prev', -1], ['detail-next', 1]]) $(id).onclick = () => {
    const index = state.photos.findIndex((item) => item.id === state.selected.id);
    detail(state.photos[(index + delta + state.photos.length) % state.photos.length]);
  };
  document.querySelectorAll('[data-filter]').forEach((button) => button.onclick = () => {
    state.filter = button.dataset.filter;
    document.querySelectorAll('[data-filter]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    loadPhotos();
  });
  $('range-filter').onchange = () => { state.range = $('range-filter').value; loadPhotos(); };
  let searchTimer;
  $('snap-search').oninput = () => { state.query = $('snap-search').value; clearTimeout(searchTimer); searchTimer = setTimeout(() => loadPhotos(), 200); };
  $('empty-clear').onclick = () => {
    state.filter = 'all'; state.range = 'all'; state.query = '';
    $('range-filter').value = 'all'; $('snap-search').value = '';
    document.querySelectorAll('[data-filter]').forEach((item) => item.setAttribute('aria-pressed', String(item.dataset.filter === 'all')));
    loadPhotos();
  };
  $('load-more').onclick = () => loadPhotos(true);
  $('refresh-photos').onclick = () => loadPhotos();
  $('pause-btn').onclick = () => {
    state.paused = !state.paused;
    $('pause-btn').setAttribute('aria-pressed', String(state.paused));
    setText('pause-label', state.paused ? '继续画面' : '暂停画面');
    $('pause-icon-pause').hidden = state.paused; $('pause-icon-play').hidden = !state.paused;
    $('feed-pause-overlay').hidden = !state.paused;
    if (!state.paused) { state.preview = null; showPreview(state.status); }
  };
  $('fullscreen-btn').onclick = async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('live-feed').requestFullscreen(); }
    catch (error) { showError('浏览器未允许全屏'); }
  };
  document.addEventListener('fullscreenchange', () => setText('fullscreen-label', document.fullscreenElement ? '退出全屏' : '全屏'));
  $('start-btn').onclick = async () => {
    $('start-btn').disabled = true;
    try {
      const selected = $('run-mode').value;
      await api('/api/runs', selected === 'live' ? { mode: 'live' } : { mode: 'replay', source: selected });
      renderStatus(await api('/api/status'));
    } catch (error) { showError(`启动失败：${error.message}`); $('start-btn').disabled = !state.connected; }
  };
  $('stop-btn').onclick = async () => {
    $('stop-btn').disabled = true;
    try { await api('/api/stop', {}); } catch (error) { showError(`停止失败：${error.message}`); }
  };

  let ticks = 0;
  async function poll() {
    try {
      renderStatus(await api('/api/status'));
      renderEvents((await api('/api/events?limit=4')).items);
      if (!state.photoBusy && (ticks % 3 === 0) && state.photos.length <= 9) await loadPhotos();
    } catch (error) {
      state.connected = false;
      setText('connection-state', '后端连接中断');
      setText('device-state', '状态未知');
      $('start-btn').disabled = true; $('stop-btn').disabled = true;
      $('preview-image').hidden = true; $('frame-empty').hidden = false;
      setText('frame-empty', '后端连接中断，无法确认设备状态');
      setText('frame-state', '连接中断'); setText('live-badge', '离线');
      showError(`连接失败，将自动重试：${error.message}`);
    } finally { ticks += 1; setTimeout(poll, 1000); }
  }
  function clock() {
    const now = new Date();
    setText('header-clock-date', now.toLocaleDateString('zh-CN'));
    setText('header-clock-time', now.toLocaleTimeString('zh-CN', { hour12: false }));
    $('header-clock').dateTime = now.toISOString();
  }
  clock(); setInterval(clock, 1000); poll();
})();
