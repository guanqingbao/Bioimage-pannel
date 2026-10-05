const fileInput = document.querySelector('#fileInput');
const attachButton = document.querySelector('#attachButton');
const removeFileButton = document.querySelector('#removeFile');
const selectedFile = document.querySelector('#selectedFile');
const selectedFileName = document.querySelector('#selectedFileName');
const sendButton = document.querySelector('#sendButton');
const composer = document.querySelector('#composer');
const prompt = document.querySelector('#prompt');
const messages = document.querySelector('#messages');
const serviceStatus = document.querySelector('#serviceStatus');
const sidebarToggleButton = document.querySelector('#sidebarToggle');
const initialMessage = messages.innerHTML;
const editorDialog = document.querySelector('#editorDialog');
const closeEditorButton = document.querySelector('#closeEditor');
const editorSourceName = document.querySelector('#editorSourceName');
const editorImage = document.querySelector('#editorImage');
const editorCanvas = document.querySelector('#editorCanvas');
const editorPreviewCanvas = document.querySelector('#editorPreviewCanvas');
const editorPreviewEmpty = document.querySelector('#editorPreviewEmpty');
const editorSelectionSize = document.querySelector('#editorSelectionSize');
const editorLabel = document.querySelector('#editorLabel');
const editorLabelOptions = document.querySelector('#editorLabelOptions');
const editorCurrentVersion = document.querySelector('#editorCurrentVersion');
const editorCurrentImage = document.querySelector('#editorCurrentImage');
const editorCurrentEmpty = document.querySelector('#editorCurrentEmpty');
const editorHistory = document.querySelector('#editorHistory');
const understandingCurrentVersion = document.querySelector('#understandingCurrentVersion');
const panelUnderstanding = document.querySelector('#panelUnderstanding');
const saveUnderstandingButton = document.querySelector('#saveUnderstanding');
const understandingHistory = document.querySelector('#understandingHistory');
const editorError = document.querySelector('#editorError');
const resetSelectionButton = document.querySelector('#resetSelection');
const deletePanelButton = document.querySelector('#deletePanel');
const confirmCropButton = document.querySelector('#confirmCrop');
const folderInput = document.querySelector('#folderInput');
const browseFolderButton = document.querySelector('#browseFolder');
const folderBrowserDialog = document.querySelector('#folderBrowserDialog');
const closeFolderBrowserButton = document.querySelector('#closeFolderBrowser');
const folderRootList = document.querySelector('#folderRootList');
const folderCurrentPath = document.querySelector('#folderCurrentPath');
const folderUpButton = document.querySelector('#folderUp');
const folderBrowserMessage = document.querySelector('#folderBrowserMessage');
const folderDirectoryList = document.querySelector('#folderDirectoryList');
const folderPdfCount = document.querySelector('#folderPdfCount');
const chooseFolderButton = document.querySelector('#chooseFolder');
const recursiveInput = document.querySelector('#recursiveInput');
const concurrencyInput = document.querySelector('#concurrencyInput');
const folderStartButton = document.querySelector('#folderStart');
const refreshBatchesButton = document.querySelector('#refreshBatches');
const rebuildCollectionButton = document.querySelector('#rebuildCollection');
const collectionStatus = document.querySelector('#collectionStatus');
const batchStatus = document.querySelector('#batchStatus');
const batchProgress = document.querySelector('#batchProgress');
const batchTitle = document.querySelector('#batchTitle');
const batchPercent = document.querySelector('#batchPercent');
const batchProgressBar = document.querySelector('#batchProgressBar');
const batchItems = document.querySelector('#batchItems');
const batchHistory = document.querySelector('#batchHistory');
const historyRefreshButton = document.querySelector('#historyRefresh');

// The toolbox runs either at `/` by itself or below `/image-processing/`
// inside the BioMat application. Keep every API request on the same mount.
const appBasePath = window.location.pathname
  .replace(/\/+$/, '')
  .replace(/\/index\.html$/, '');
const appUrl = path => `${appBasePath}${path}`;

function installPdfReviewUi() {
  const style = document.createElement('style');
  style.textContent = `
    .pdf-review-dialog { width: min(1500px, calc(100vw - 28px)); height: min(920px, calc(100vh - 28px)); max-width: none; max-height: none; padding: 0; overflow: hidden; border: 1px solid #cfd8d4; border-radius: 12px; color: #18201f; background: #f5f7f6; box-shadow: 0 28px 90px rgba(23,41,37,.3); }
    .pdf-review-dialog::backdrop { background: rgba(24,32,31,.62); backdrop-filter: blur(3px); }
    .pdf-review-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; height: 78px; padding: 14px 18px 14px 22px; border-bottom: 1px solid #dfe5e3; background: #fff; }
    .pdf-review-header > div { min-width: 0; }
    .pdf-review-header span { color: #146b5c; font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .pdf-review-header h2 { overflow: hidden; margin: 3px 0 0; font-size: 17px; text-overflow: ellipsis; white-space: nowrap; }
    .pdf-review-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1px; height: calc(100% - 134px); background: #dce3e0; }
    .pdf-review-pane { display: grid; grid-template-rows: auto minmax(0, 1fr); min-width: 0; min-height: 0; background: #edf1ef; }
    .pdf-review-pane header { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 14px; border-bottom: 1px solid #dce3e0; background: #fff; }
    .pdf-review-pane header strong { font-size: 12px; }
    .pdf-review-pane header span { color: #71817e; font-size: 10px; }
    .pdf-review-image-wrap { position: relative; display: grid; min-height: 0; place-items: center; overflow: auto; padding: 14px; }
    .pdf-review-image-wrap img { display: block; max-width: 100%; max-height: 100%; object-fit: contain; background: #fff; box-shadow: 0 3px 16px rgba(23,41,37,.13); }
    .pdf-review-status { position: absolute; inset: 0; display: grid; place-items: center; padding: 20px; color: #667672; background: #edf1ef; text-align: center; font-size: 12px; }
    .pdf-review-status[hidden] { display: none; }
    .pdf-review-footer { display: flex; align-items: center; justify-content: space-between; gap: 14px; height: 56px; padding: 10px 18px; border-top: 1px solid #dfe5e3; background: #fff; }
    .pdf-review-footer span { color: #667672; font-size: 11px; }
    .pdf-review-footer a { text-decoration: none; }
    @media (max-width: 820px) {
      .pdf-review-dialog { width: 100vw; height: 100vh; border: 0; border-radius: 0; }
      .pdf-review-grid { grid-template-columns: 1fr; grid-template-rows: repeat(2, minmax(0, 1fr)); }
    }
  `;
  document.head.append(style);

  const dialog = document.createElement('dialog');
  dialog.className = 'pdf-review-dialog';
  dialog.setAttribute('aria-labelledby', 'pdfReviewTitle');
  dialog.innerHTML = `
    <div class="pdf-review-header">
      <div><span>整图提取复核</span><h2 id="pdfReviewTitle">PDF 原页与提取整图对照</h2></div>
      <button class="icon-button" type="button" data-close-pdf-review aria-label="关闭原页对照" title="关闭">${icon('x')}<span class="fallback-icon">×</span></button>
    </div>
    <div class="pdf-review-grid">
      <section class="pdf-review-pane">
        <header><strong>PDF 原始页面</strong><span id="pdfReviewPageLabel"></span></header>
        <div class="pdf-review-image-wrap">
          <img id="pdfReviewPageImage" alt="标出 Figure 提取区域的 PDF 原始页面">
          <div id="pdfReviewPageStatus" class="pdf-review-status">正在渲染 PDF 原始页面…</div>
        </div>
      </section>
      <section class="pdf-review-pane">
        <header><strong>提取后的 300 DPI 整图</strong><span id="pdfReviewFigureSize"></span></header>
        <div class="pdf-review-image-wrap">
          <img id="pdfReviewFigureImage" alt="提取后的完整 Figure">
        </div>
      </section>
    </div>
    <div class="pdf-review-footer">
      <span>红框表示实际提取区域；可据此检查是否裁少、裁多或匹配错误。</span>
      <a id="pdfReviewOpenPdf" class="secondary-button" target="_blank" rel="noopener">在浏览器中打开原 PDF</a>
    </div>`;
  document.body.append(dialog);
  dialog.querySelector('[data-close-pdf-review]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target === dialog) dialog.close();
  });
  return {
    dialog,
    title: dialog.querySelector('#pdfReviewTitle'),
    pageLabel: dialog.querySelector('#pdfReviewPageLabel'),
    pageImage: dialog.querySelector('#pdfReviewPageImage'),
    pageStatus: dialog.querySelector('#pdfReviewPageStatus'),
    figureImage: dialog.querySelector('#pdfReviewFigureImage'),
    figureSize: dialog.querySelector('#pdfReviewFigureSize'),
    openPdf: dialog.querySelector('#pdfReviewOpenPdf'),
  };
}

let activeResult = null;
let activeResultRoot = null;
let activeEditorRecord = null;
let editorSelection = null;
let dragStart = null;
const resultSessions = new Map();
const navigationBatchCache = new Map();
let activeBatchId = '';
let batchPollTimer = null;
let folderBrowserState = null;

const icon = (name) => `<i data-lucide="${name}"></i>`;
const refreshIcons = () => window.lucide?.createIcons();
const pdfReviewUi = installPdfReviewUi();
const scrollToBottom = () => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

function setSidebarCollapsed(collapsed) {
  document.body.classList.toggle('sidebar-collapsed', collapsed);
  sidebarToggleButton.setAttribute('aria-expanded', String(!collapsed));
  sidebarToggleButton.setAttribute('aria-label', collapsed ? '展开左侧导航' : '收起左侧导航');
  sidebarToggleButton.title = collapsed ? '展开左侧导航' : '收起左侧导航';
  sidebarToggleButton.innerHTML = `${icon(collapsed ? 'panel-left-open' : 'panel-left-close')}<span class="fallback-icon">${collapsed ? '›' : '‹'}</span>`;
  refreshIcons();
}

function updateSelection() {
  const files = Array.from(fileInput.files || []);
  selectedFile.hidden = files.length === 0;
  selectedFileName.textContent = files.length === 1
    ? files[0].name
    : files.length
      ? `已选择 ${files.length} 篇 PDF`
      : '';
  sendButton.disabled = files.length === 0;
  prompt.placeholder = files.length ? '可填写批次备注，或直接开始' : '选择多个 PDF 后创建批次';
  refreshIcons();
}

function setBatchStatus(message, isError = false) {
  batchStatus.textContent = message || '';
  batchStatus.classList.toggle('error', Boolean(isError));
}

function batchStatusLabel(status) {
  return ({
    queued: '排队中',
    running: '处理中',
    processing: '处理中',
    completed: '已完成',
    completed_with_errors: '完成（有失败）',
    failed: '失败',
  })[status] || status || '未知';
}

function renderBatchState(state) {
  activeBatchId = state.batch_id;
  batchProgress.hidden = false;
  const progress = Math.max(0, Math.min(1, Number(state.progress || 0)));
  const completed = Number(state.completed_count || 0);
  const failed = Number(state.failed_count || 0);
  batchTitle.textContent = `${state.source_label || '当前批次'} · ${batchStatusLabel(state.status)}`;
  batchPercent.textContent = `${Math.round(progress * 100)}% · ${completed}/${state.total}${failed ? ` · 失败 ${failed}` : ''}`;
  batchProgressBar.style.width = `${progress * 100}%`;

  const rows = Array.isArray(state.items) ? state.items : [];
  batchItems.innerHTML = rows.map(item => {
    const metrics = item.metrics || {};
    const processingDetails = item.status === 'completed'
      ? `${metrics.figures || 0} 整图 · ${metrics.panels || 0} 子图 · ${metrics.needs_review || 0} 待复核`
      : item.error || batchStatusLabel(item.status);
    const details = item.source_group && item.source_group !== '浏览器上传'
      ? `${item.source_group} · ${processingDetails}`
      : processingDetails;
    const action = item.status === 'completed'
      ? `<button class="open-job-button secondary-button" type="button" data-job-id="${escapeHtml(item.job_id)}">打开结果</button>`
      : '';
    return `<div class="batch-item ${escapeHtml(item.status)}">
      <div><strong title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</strong><span>${escapeHtml(details)}</span></div>
      <span class="batch-item-status">${escapeHtml(batchStatusLabel(item.status))}</span>${action}
    </div>`;
  }).join('') || '<div class="preview-empty">等待任务进入队列…</div>';
  if (state.items_total > rows.length) {
    batchItems.insertAdjacentHTML('beforeend', `<div class="batch-item-more">仅展示前 ${rows.length} 项，共 ${state.items_total} 项；完整状态保存在 SQLite 队列中。</div>`);
  }
  setBatchStatus(`批次 ${state.batch_id.slice(0, 8)} · 并发 ${state.concurrency} · ${batchStatusLabel(state.status)}`, failed > 0);
}

async function fetchBatchState(batchId) {
  const response = await fetch(appUrl(`/api/batches/${batchId}?limit=200`));
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || '批次状态读取失败');
  renderBatchState(payload);
  return payload;
}

function stopBatchPolling() {
  if (batchPollTimer) window.clearInterval(batchPollTimer);
  batchPollTimer = null;
}

function startBatchPolling(batchId) {
  stopBatchPolling();
  activeBatchId = batchId;
  const poll = async () => {
    try {
      const state = await fetchBatchState(batchId);
      if (state.status === 'completed' || state.status === 'completed_with_errors') {
        stopBatchPolling();
        loadBatchHistory();
      }
    } catch (error) {
      setBatchStatus(error.message, true);
      stopBatchPolling();
    }
  };
  poll();
  batchPollTimer = window.setInterval(poll, 2000);
}

function localDateKey(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'unknown';
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function historyDateLabel(key) {
  if (key === 'unknown') return '时间未知';
  const [year, month, day] = key.split('-').map(Number);
  const target = new Date(year, month - 1, day);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const difference = Math.round((today.getTime() - target.getTime()) / 86400000);
  if (difference === 0) return '今天';
  if (difference === 1) return '昨天';
  return `${year}年${month}月${day}日`;
}

function historyTimeLabel(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--:--';
  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date);
}

function reviewBadge(item) {
  if (item.status !== 'completed') {
    return { className: item.status === 'failed' ? 'failed' : 'processing', text: batchStatusLabel(item.status) };
  }
  const total = Number(item.figure_count || 0);
  const verified = Number(item.verified_count || 0);
  const ambiguous = Number(item.ambiguous_count || 0);
  if (item.review_state === 'verified' && total > 0) {
    return { className: 'verified', text: '已校验完成' };
  }
  if (verified > 0) {
    return { className: 'partial', text: `已校验 ${verified}/${total}` };
  }
  if (ambiguous > 0) {
    return { className: 'ambiguous', text: `${ambiguous} 张有歧义` };
  }
  return { className: 'unreviewed', text: '未校验' };
}

function navigationJobMarkup(item) {
  const badge = reviewBadge(item);
  const metrics = item.metrics || {};
  const disabled = item.status === 'completed' ? '' : ' disabled';
  return `
    <button type="button" class="analysis-job-item${activeResult?.job_id === item.job_id ? ' active' : ''}"
      data-navigation-job-id="${escapeHtml(item.job_id)}"${disabled}>
      <span class="analysis-job-main">
        <strong title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</strong>
        <small>${metrics.figures || item.figure_count || 0} 张整图 · ${metrics.panels || 0} 个子图</small>
      </span>
      <span class="review-badge ${badge.className}">${escapeHtml(badge.text)}</span>
    </button>`;
}

function navigationDirectoryMarkup(name, items, index) {
  return `
    <details class="analysis-directory-group"${index === 0 ? ' open' : ''}>
      <summary>
        <span>${icon('folder')}<strong title="${escapeHtml(name)}">${escapeHtml(name)}</strong></span>
        <small>${items.length} 篇</small>
      </summary>
      <div class="analysis-directory-jobs">${items.map(navigationJobMarkup).join('')}</div>
    </details>`;
}

function groupedNavigationMarkup(items) {
  const groups = new Map();
  items.forEach(item => {
    const name = item.source_group || '未分类';
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(item);
  });
  return Array.from(groups.entries())
    .map(([name, groupedItems], index) => navigationDirectoryMarkup(name, groupedItems, index))
    .join('');
}

async function loadNavigationBatch(details, force = false) {
  const batchId = details?.dataset.navigationBatchId;
  const target = details?.querySelector('.analysis-job-list');
  if (!batchId || !target || target.dataset.loading === 'true') return;
  if (!force && target.dataset.loaded === 'true') return;
  target.dataset.loading = 'true';
  target.innerHTML = '<div class="history-loading"><span class="spinner"></span><span>正在读取文献…</span></div>';
  try {
    let offset = 0;
    let total = 0;
    const items = [];
    do {
      const response = await fetch(appUrl(`/api/batches/${batchId}?offset=${offset}&limit=1000&include_review=true`));
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || '解析记录读取失败');
      items.push(...(payload.items || []));
      total = Number(payload.items_total || items.length);
      offset = items.length;
    } while (offset < total);
    navigationBatchCache.set(batchId, items);
    target.innerHTML = groupedNavigationMarkup(items) || '<div class="history-empty">这个批次还没有文献</div>';
    target.dataset.loaded = 'true';
    refreshIcons();
  } catch (error) {
    target.innerHTML = `<div class="history-error">${escapeHtml(error.message)}</div>`;
  } finally {
    target.dataset.loading = 'false';
  }
}

function updateNavigationJobFromResult(result) {
  if (!result?.job_id) return;
  const records = Array.isArray(result.records) ? result.records : [];
  const total = records.length;
  const verified = records.filter(record => record.review_status === 'verified' && record.annotation_complete).length;
  const ambiguous = records.filter(record => record.review_status === 'ambiguous').length;
  const item = {
    status: 'completed',
    figure_count: total,
    verified_count: verified,
    ambiguous_count: ambiguous,
    review_state: total > 0 && verified === total
      ? 'verified'
      : verified > 0
        ? 'partial'
        : ambiguous > 0 ? 'ambiguous' : 'unreviewed',
  };
  const badge = reviewBadge(item);
  const button = batchHistory.querySelector(`[data-navigation-job-id="${result.job_id}"]`);
  if (!button) return;
  button.querySelector('.review-badge').className = `review-badge ${badge.className}`;
  button.querySelector('.review-badge').textContent = badge.text;
}

async function loadBatchHistory(force = false) {
  if (force) navigationBatchCache.clear();
  if (historyRefreshButton) historyRefreshButton.disabled = true;
  try {
    const response = await fetch(appUrl('/api/batches?limit=200'));
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '批次记录加载失败');
    const groups = new Map();
    (payload.batches || []).forEach(batch => {
      const key = localDateKey(batch.created_at);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(batch);
    });
    batchHistory.innerHTML = Array.from(groups.entries()).map(([key, batches], groupIndex) => `
      <details class="analysis-date-group"${groupIndex === 0 ? ' open' : ''}>
        <summary><span>${escapeHtml(historyDateLabel(key))}</span><small>${batches.reduce((sum, batch) => sum + Number(batch.total || 0), 0)} 篇</small></summary>
        <div class="analysis-date-content">
          ${batches.map((batch, batchIndex) => `
            <details class="analysis-batch" data-navigation-batch-id="${escapeHtml(batch.batch_id)}"${groupIndex === 0 && batchIndex === 0 ? ' open' : ''}>
              <summary>
                <span><strong>${escapeHtml(batch.source_label || `批次 ${batch.batch_id.slice(0, 8)}`)}</strong><small>${historyTimeLabel(batch.created_at)} · ${escapeHtml(batchStatusLabel(batch.status))}</small></span>
                <span>${batch.completed_count || 0}/${batch.total || 0}</span>
              </summary>
              <div class="analysis-job-list" data-loaded="false"></div>
            </details>`).join('')}
        </div>
      </details>`).join('') || '<div class="history-empty">还没有解析记录</div>';
    batchHistory.querySelectorAll('.analysis-batch[open]').forEach(details => loadNavigationBatch(details));
    refreshIcons();
  } catch (error) {
    batchHistory.innerHTML = `<div class="history-error">${escapeHtml(error.message)}</div>`;
  } finally {
    if (historyRefreshButton) historyRefreshButton.disabled = false;
  }
}

async function openJobResult(jobId, replaceExisting = false) {
  if (replaceExisting) {
    messages.innerHTML = initialMessage;
  }
  const waiting = appendMessage('assistant', '<div class="thinking"><span class="spinner"></span><span>正在载入整图、子图和人工版本…</span></div>');
  waiting.classList.add('job-result-message');
  waiting.dataset.jobId = jobId;
  try {
    const response = await fetch(appUrl(`/api/jobs/${jobId}/result`));
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '结果加载失败');
    resultSessions.set(data.job_id, data);
    activeResult = data;
    const resultRoot = waiting.querySelector('.message-body');
    resultRoot.innerHTML = renderResult(data);
    initializeFigureCarousels(resultRoot);
    batchHistory.querySelectorAll('.analysis-job-item').forEach(item => {
      item.classList.toggle('active', item.dataset.navigationJobId === jobId);
    });
    updateNavigationJobFromResult(data);
    refreshIcons();
  } catch (error) {
    waiting.querySelector('.message-body').innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
  }
}

async function startFolderBatch() {
  const folder = folderInput.value.trim();
  if (!folder) {
    setBatchStatus('请输入服务器上的 PDF 文件夹路径。', true);
    return;
  }
  folderStartButton.disabled = true;
  setBatchStatus('正在扫描文件夹并创建批次…');
  try {
    const response = await fetch(appUrl('/api/batches/folder'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        folder,
        recursive: recursiveInput.checked,
        concurrency: Number(concurrencyInput.value || 1),
      }),
    });
    const state = await response.json();
    if (!response.ok) throw new Error(state.detail || '批次创建失败');
    renderBatchState(state);
    startBatchPolling(state.batch_id);
    loadBatchHistory();
  } catch (error) {
    setBatchStatus(error.message, true);
  } finally {
    folderStartButton.disabled = false;
  }
}

function renderServerFolders(state) {
  folderBrowserState = state;
  folderRootList.innerHTML = (state.roots || []).map(root => `
    <button type="button" class="folder-root-button${root.path === state.current ? ' active' : ''}"
      data-folder-path="${escapeHtml(root.path)}"${root.available ? '' : ' disabled'}>
      ${icon('hard-drive')}<span>${escapeHtml(root.name)}</span>
    </button>`).join('');
  folderCurrentPath.textContent = state.current || '尚未选择目录';
  folderUpButton.disabled = !state.parent;
  folderBrowserMessage.textContent = state.message || '';
  folderBrowserMessage.hidden = !state.message;
  folderDirectoryList.innerHTML = (state.directories || []).map(directory => `
    <button type="button" class="folder-directory-button" data-folder-path="${escapeHtml(directory.path)}">
      ${icon('folder')}<span><strong>${escapeHtml(directory.name)}</strong><small>${Number(directory.pdf_count || 0)} 篇 PDF</small></span>
      ${icon('chevron-right')}
    </button>`).join('') || (state.current
      ? '<div class="folder-directory-empty">当前目录没有子目录</div>'
      : '');
  folderPdfCount.textContent = state.current
    ? `当前目录 ${Number(state.pdf_count || 0)} 篇 PDF`
    : '当前目录不可用';
  chooseFolderButton.disabled = !state.current;
  refreshIcons();
}

async function loadServerFolders(path = '') {
  folderBrowserMessage.hidden = false;
  folderBrowserMessage.textContent = '正在读取服务器目录…';
  folderDirectoryList.innerHTML = '<div class="history-loading"><span class="spinner"></span><span>正在读取目录…</span></div>';
  const query = path ? `?path=${encodeURIComponent(path)}` : '';
  const response = await fetch(appUrl(`/api/server-folders${query}`));
  const state = await response.json();
  if (!response.ok) throw new Error(state.detail || '服务器目录读取失败');
  renderServerFolders(state);
  return state;
}

async function navigateServerFolder(path) {
  try {
    await loadServerFolders(path);
  } catch (error) {
    folderBrowserMessage.hidden = false;
    folderBrowserMessage.textContent = error.message;
  }
}

async function openFolderBrowser() {
  folderBrowserDialog.showModal();
  refreshIcons();
  const requestedPath = folderInput.value.trim();
  try {
    await loadServerFolders(requestedPath);
  } catch (error) {
    if (requestedPath) {
      try {
        await loadServerFolders();
        folderBrowserMessage.hidden = false;
        folderBrowserMessage.textContent = `${error.message}，已返回默认根目录。`;
        return;
      } catch (fallbackError) {
        folderBrowserMessage.textContent = fallbackError.message;
      }
    } else {
      folderBrowserMessage.textContent = error.message;
    }
    folderDirectoryList.innerHTML = '';
    chooseFolderButton.disabled = true;
  }
}

function renderCollectionSummary(summary) {
  const ready = Number(summary.ready_for_training || 0);
  const samples = Number(summary.samples || 0);
  const noSplit = Number(summary.no_split || 0);
  const verifiedPanels = Number(summary.verified_panels || 0);
  const failures = Array.isArray(summary.failures) ? summary.failures.length : 0;
  collectionStatus.textContent = `训练素材库：${samples} 张完整图 · ${ready} 张已审核可训练 · ${noSplit} 张无需划分 · ${verifiedPanels} 个真值框${failures ? ` · ${failures} 项失败` : ''}`;
  collectionStatus.classList.toggle('error', failures > 0);
}

async function loadTrainingCollection() {
  try {
    const response = await fetch(appUrl('/api/training-collection'));
    const summary = await response.json();
    if (!response.ok) throw new Error(summary.detail || '训练素材库状态读取失败');
    renderCollectionSummary(summary);
  } catch (error) {
    collectionStatus.textContent = error.message;
    collectionStatus.classList.add('error');
  }
}

async function rebuildTrainingCollection() {
  rebuildCollectionButton.disabled = true;
  collectionStatus.textContent = '正在汇总完整图、标注版本和训练标签…';
  collectionStatus.classList.remove('error');
  try {
    const response = await fetch(appUrl('/api/training-collection/rebuild'), { method: 'POST' });
    const summary = await response.json();
    if (!response.ok) throw new Error(summary.detail || '训练数据整理失败');
    renderCollectionSummary(summary);
  } catch (error) {
    collectionStatus.textContent = error.message;
    collectionStatus.classList.add('error');
  } finally {
    rebuildCollectionButton.disabled = false;
  }
}

function appendMessage(type, html) {
  const template = document.querySelector(`#${type}MessageTemplate`);
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector('.message-body').innerHTML = html;
  messages.append(node);
  refreshIcons();
  scrollToBottom();
  return node;
}

function metric(label, value, warning = false) {
  return `<div class="metric${warning ? ' warning' : ''}"><strong>${value}</strong><span>${label}</span></div>`;
}

function recordStatus(record) {
  if (record.review_status === 'verified' && record.annotation_complete) {
    return record.annotation_mode === 'no_split'
      ? '已校验完成 · 无需划分'
      : '已校验完成（人工确认）';
  }
  if (record.review_status === 'ambiguous') return '人工标记：存在歧义';
  if (record.extraction_needs_review) return '整图提取需复核';
  if (record.low_confidence_panels?.length) return '切分结果需复核';
  return record.panel_count ? '自动候选，尚未整图确认' : '证据不足，等待人工标注';
}

function renderPanelTarget(target, jobId, recordId) {
  const understanding = target.understanding_text?.trim();
  const understandingVersion = target.understanding_current_version || 0;
  return `
    <figure class="panel-output" data-panel-label="${escapeHtml(target.label)}">
      <a href="${target.current_url}" target="_blank" rel="noopener">
        <img src="${target.current_url}" alt="面板 ${escapeHtml(target.label)}" loading="lazy">
      </a>
      <figcaption><strong>${escapeHtml(target.label)}</strong><span>图片 v${target.current_version}</span></figcaption>
      <div class="panel-understanding">
        <div><span>图片理解</span><strong>${understandingVersion ? `v${understandingVersion}` : '未填写'}</strong></div>
        <p class="${understanding ? '' : 'empty'}">${understanding ? escapeHtml(understanding) : '尚未填写对子图的理解'}</p>
        <button class="understanding-button" type="button" data-job-id="${escapeHtml(jobId)}" data-record-id="${escapeHtml(recordId)}" data-panel-label="${escapeHtml(target.label)}">${understanding ? '修改理解' : '填写理解'}</button>
      </div>
    </figure>`;
}

function renderPanelTargets(record, jobId) {
  if (record.review_status === 'verified' && record.annotation_complete && record.annotation_mode === 'no_split') {
    return '<p class="panel-empty verified-no-split">人工确认：这张完整 Figure 无需划分。自动候选未纳入真值，训练标签为空。</p>';
  }
  if (!record.panel_targets?.length) {
    return '<p class="panel-empty">当前未生成独立子图，整图已完整保留。</p>';
  }
  return record.panel_targets
    .map(target => renderPanelTarget(target, jobId, record.record_id))
    .join('');
}

function figureRecordTitle(record, index) {
  const kind = record.caption_kind === 'scheme' ? 'Scheme' : 'Figure';
  return record.figure_number ? `${kind} ${record.figure_number}` : `图片 ${index + 1}`;
}

function renderFigureRecord(record, jobId, index) {
  const title = escapeHtml(figureRecordTitle(record, index));
  const page = record.page_number ? `PDF 第 ${record.page_number} 页` : '直接上传图片';
  const caption = record.caption?.trim()
    ? escapeHtml(record.caption)
    : '未提取到对应图注；该图片仍可进行自动切分和人工框选。';
  const flags = record.extraction_quality_flags?.length
    ? record.extraction_quality_flags.map(flag => `<span>${escapeHtml(flag)}</span>`).join('')
    : '<span>无提取警告</span>';
  const lowConfidence = record.low_confidence_panels?.length
    ? record.low_confidence_panels.join(', ')
    : '-';
  const verified = record.review_status === 'verified' && record.annotation_complete;
  const noSplit = verified && record.annotation_mode === 'no_split';
  const outputLabels = noSplit ? '无需划分' : (record.labels || '-');
  const panelCount = noSplit ? 0 : record.panel_count;
  const sourceReviewActions = record.source_pdf_available
    ? `<button class="source-page-compare-button secondary-button" type="button" data-job-id="${escapeHtml(jobId)}" data-record-id="${escapeHtml(record.record_id)}">原页对照</button>
       <button class="source-pdf-button secondary-button" type="button" data-job-id="${escapeHtml(jobId)}" data-record-id="${escapeHtml(record.record_id)}">打开原 PDF</button>`
    : '';

  return `
    <article class="figure-result-card" data-record-id="${escapeHtml(record.record_id)}">
      <header class="figure-card-header">
        <div>
          <span class="figure-kicker">${escapeHtml(page)}</span>
          <h3>${title}</h3>
        </div>
        <span class="figure-status${verified ? ' verified' : ' needs-review'}">${escapeHtml(recordStatus(record))}</span>
      </header>
      <div class="figure-card-body">
        <figure class="whole-figure">
          <a href="${record.whole_image_url}" target="_blank" rel="noopener">
            <img src="${record.whole_image_url}" alt="${title} 整图" loading="lazy">
          </a>
          <figcaption>300 DPI 整图 · ${record.whole_image_width_px} × ${record.whole_image_height_px}px</figcaption>
        </figure>
        <aside class="figure-info">
          <section class="caption-block">
            <span>Caption</span>
            <p>${caption}</p>
          </section>
          <dl class="figure-metadata">
            <div><dt>源文件</dt><dd>${escapeHtml(record.source)}</dd></div>
            <div><dt>图注标签</dt><dd>${escapeHtml(record.expected_labels || '-')}</dd></div>
            <div><dt>输出标签</dt><dd>${escapeHtml(outputLabels)}</dd></div>
            <div><dt>面板数量</dt><dd>${panelCount}</dd></div>
            <div><dt>平均置信度</dt><dd>${record.mean_confidence.toFixed(3)}</dd></div>
            <div><dt>低置信面板</dt><dd>${escapeHtml(lowConfidence)}</dd></div>
            <div><dt>整图提取方式</dt><dd>${escapeHtml(record.extraction_method)}</dd></div>
          </dl>
          <div class="quality-flags">${flags}</div>
          <div class="figure-review-actions">
            ${sourceReviewActions}
            <button class="edit-panel-button" type="button" data-job-id="${escapeHtml(jobId)}" data-record-id="${escapeHtml(record.record_id)}">在整图上框选修正</button>
            <button class="verify-figure-button" type="button" data-job-id="${escapeHtml(jobId)}" data-record-id="${escapeHtml(record.record_id)}"${verified ? ' disabled' : ''}>${verified && !noSplit ? '已校验完成' : '确认面板划分完成'}</button>
            <button class="no-split-figure-button secondary-button" type="button" data-job-id="${escapeHtml(jobId)}" data-record-id="${escapeHtml(record.record_id)}"${noSplit ? ' disabled' : ''}>${noSplit ? '整图无需划分' : '确认整图无需划分'}</button>
            <button class="ambiguous-figure-button secondary-button" type="button" data-job-id="${escapeHtml(jobId)}" data-record-id="${escapeHtml(record.record_id)}">标记歧义</button>
          </div>
        </aside>
      </div>
      <section class="panel-results">
        <div class="panel-results-title"><strong>子图结果</strong><span>点击图片查看原始尺寸</span></div>
        <div class="panel-grid">${renderPanelTargets(record, jobId)}</div>
      </section>
    </article>
  `;
}

function renderResult(data) {
  const m = data.metrics;
  const documents = data.documents?.length
    ? data.documents
    : [{ name: data.records[0]?.document_name || '未命名文献', record_ids: data.records.map(record => record.record_id) }];
  const documentGroups = documents.map(document => {
    const records = document.record_ids
      .map(id => data.records.find(record => record.record_id === id))
      .filter(Boolean);
    const verifiedCount = records.filter(record => record.review_status === 'verified' && record.annotation_complete).length;
    const fullyVerified = records.length > 0 && verifiedCount === records.length;
    return `
      <section class="document-group">
        <header class="document-header">
          <div><span>文献</span><h2>${escapeHtml(document.name)}</h2></div>
          <div class="document-summary">
            <span>${records.length} 张整图</span>
            <span>${records.reduce((sum, record) => sum + record.panel_count, 0)} 个子图</span>
            <span class="document-review-tag${fullyVerified ? ' verified' : ''}">${fullyVerified ? '已校验完成' : `已校验 ${verifiedCount}/${records.length}`}</span>
          </div>
        </header>
        <div class="figure-carousel" data-figure-carousel data-current-index="0">
          <div class="figure-carousel-toolbar">
            <label class="figure-carousel-picker">
              <span>按整图浏览</span>
              <select class="figure-carousel-select" aria-label="选择要查看的整图">
                ${records.map((record, index) => `<option value="${index}">${escapeHtml(figureRecordTitle(record, index))}</option>`).join('')}
              </select>
            </label>
            <div class="figure-carousel-controls">
              <button class="figure-carousel-nav" type="button" data-carousel-direction="-1" aria-label="上一张整图" title="上一张整图" disabled>${icon('chevron-left')}<span>上一张</span></button>
              <span class="figure-carousel-count">第 <strong>1</strong> / ${records.length} 张</span>
              <button class="figure-carousel-nav" type="button" data-carousel-direction="1" aria-label="下一张整图" title="下一张整图"${records.length <= 1 ? ' disabled' : ''}><span>下一张</span>${icon('chevron-right')}</button>
            </div>
          </div>
          <div class="figure-carousel-track figure-list" tabindex="0" aria-label="整图结果，可左右滑动切换">
            ${records.map((record, index) => renderFigureRecord(record, data.job_id, index)).join('')}
          </div>
        </div>
      </section>`;
  }).join('');

  return `
    <p>${escapeHtml(data.message)}</p>
    <div class="metrics">
      ${metric('整图', m.figures)}${metric('已处理', m.processed)}${metric('面板', m.panels)}
      ${metric('保留整图', m.preserved)}${metric('平均置信度', m.mean_confidence.toFixed(3))}${metric('需复核', m.needs_review ?? m.low_confidence, (m.needs_review ?? m.low_confidence) > 0)}
    </div>
    ${documentGroups}
  `;
}

function resultRecord(jobId, recordId) {
  const result = resultSessions.get(jobId) || (activeResult?.job_id === jobId ? activeResult : null);
  return result?.records?.find(record => record.record_id === recordId) || null;
}

function openSourcePdf(jobId, recordId) {
  const record = resultRecord(jobId, recordId);
  if (!record?.source_pdf_available) {
    window.alert('该任务没有可访问的原始 PDF。');
    return;
  }
  const url = `${record.source_pdf_url || appUrl(`/api/jobs/${jobId}/source-pdf`)}#page=${record.page_number || 1}`;
  window.open(url, '_blank', 'noopener');
}

function openPdfReview(jobId, recordId) {
  const record = resultRecord(jobId, recordId);
  if (!record?.source_pdf_available) {
    window.alert('该任务没有可访问的原始 PDF。');
    return;
  }
  const pageNumber = Number(record.page_number || 1);
  const pdfUrl = record.source_pdf_url || appUrl(`/api/jobs/${jobId}/source-pdf`);
  const previewUrl = record.source_page_preview_url
    || appUrl(`/api/jobs/${jobId}/figures/${recordId}/source-page-preview`);
  pdfReviewUi.title.textContent = `${figureRecordTitle(record, 0)} · 原页对照`;
  pdfReviewUi.pageLabel.textContent = `第 ${pageNumber} 页 · 红框为提取区域`;
  pdfReviewUi.figureSize.textContent = `${record.whole_image_width_px} × ${record.whole_image_height_px}px`;
  pdfReviewUi.openPdf.href = `${pdfUrl}#page=${pageNumber}`;
  pdfReviewUi.pageStatus.hidden = false;
  pdfReviewUi.pageStatus.textContent = '正在渲染 PDF 原始页面…';
  pdfReviewUi.pageImage.removeAttribute('src');
  pdfReviewUi.figureImage.src = record.whole_image_url;
  pdfReviewUi.pageImage.onload = () => {
    pdfReviewUi.pageStatus.hidden = true;
  };
  pdfReviewUi.pageImage.onerror = () => {
    pdfReviewUi.pageStatus.hidden = false;
    pdfReviewUi.pageStatus.textContent = '原页预览生成失败，请尝试直接打开原 PDF。';
  };
  pdfReviewUi.pageImage.src = `${previewUrl}?dpi=120`;
  pdfReviewUi.dialog.showModal();
  refreshIcons();
}

function updateFigureCarouselControls(carousel, requestedIndex) {
  const track = carousel.querySelector('.figure-carousel-track');
  const slides = Array.from(track?.children || []);
  if (!slides.length) return 0;
  const index = Math.max(0, Math.min(requestedIndex, slides.length - 1));
  carousel.dataset.currentIndex = String(index);
  const counter = carousel.querySelector('.figure-carousel-count strong');
  const select = carousel.querySelector('.figure-carousel-select');
  const previous = carousel.querySelector('[data-carousel-direction="-1"]');
  const next = carousel.querySelector('[data-carousel-direction="1"]');
  if (counter) counter.textContent = String(index + 1);
  if (select) select.value = String(index);
  if (previous) previous.disabled = index === 0;
  if (next) next.disabled = index === slides.length - 1;
  return index;
}

function scrollFigureCarousel(carousel, requestedIndex, behavior = 'smooth') {
  const track = carousel.querySelector('.figure-carousel-track');
  const slides = Array.from(track?.children || []);
  const index = updateFigureCarouselControls(carousel, requestedIndex);
  if (!track || !slides[index]) return;
  track.scrollTo({ left: slides[index].offsetLeft, behavior });
}

function initializeFigureCarousels(root) {
  root.querySelectorAll('[data-figure-carousel]').forEach(carousel => {
    const track = carousel.querySelector('.figure-carousel-track');
    const select = carousel.querySelector('.figure-carousel-select');
    if (!track) return;

    carousel.querySelectorAll('.figure-carousel-nav').forEach(button => {
      button.addEventListener('click', () => {
        const current = Number(carousel.dataset.currentIndex || 0);
        scrollFigureCarousel(carousel, current + Number(button.dataset.carouselDirection || 0));
      });
    });
    select?.addEventListener('change', () => {
      scrollFigureCarousel(carousel, Number(select.value || 0));
    });
    track.addEventListener('keydown', event => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const current = Number(carousel.dataset.currentIndex || 0);
      scrollFigureCarousel(carousel, current + (event.key === 'ArrowRight' ? 1 : -1));
    });

    let scrollFrame = null;
    track.addEventListener('scroll', () => {
      if (scrollFrame !== null) cancelAnimationFrame(scrollFrame);
      scrollFrame = requestAnimationFrame(() => {
        const slides = Array.from(track.children);
        if (!slides.length) return;
        const closest = slides.reduce((best, slide, index) => (
          Math.abs(slide.offsetLeft - track.scrollLeft) < Math.abs(slides[best].offsetLeft - track.scrollLeft)
            ? index
            : best
        ), 0);
        updateFigureCarouselControls(carousel, closest);
        scrollFrame = null;
      });
    }, { passive: true });
    updateFigureCarouselControls(carousel, 0);
  });
}

function escapeHtml(value) {
  const element = document.createElement('span');
  element.textContent = String(value);
  return element.innerHTML;
}

function normalizedEditorLabel() {
  return editorLabel.value.trim().toUpperCase().replace(/[^A-Z]/g, '').slice(0, 1);
}

function selectedPanelTarget() {
  const label = normalizedEditorLabel();
  return activeEditorRecord?.panel_targets?.find(target => target.label === label) || null;
}

function renderVersionHistory(target) {
  if (!target?.versions?.length) {
    editorHistory.innerHTML = '<div class="preview-empty">确认第一次框选后会在这里出现 v1</div>';
    return;
  }
  editorHistory.innerHTML = [...target.versions].reverse().map(version => {
    const current = version.version === target.current_version;
    const kind = version.kind === 'automatic' ? '自动切分' : '人工框选';
    const dimensions = `${version.width_px} × ${version.height_px}px`;
    return `<div class="history-item${current ? ' current' : ''}">
      <a href="${version.url}" target="_blank" rel="noopener"><img src="${version.url}" alt="${escapeHtml(target.label)} v${version.version}"></a>
      <div><strong>v${version.version} · ${kind}${current ? ' · 当前' : ''}</strong><span>${dimensions}</span></div>
    </div>`;
  }).join('');
}

function renderUnderstandingHistory(target) {
  const versions = target?.understanding_versions || [];
  if (!versions.length) {
    understandingHistory.innerHTML = '<div class="understanding-history-empty">还没有人工填写记录</div>';
    return;
  }
  understandingHistory.innerHTML = [...versions].reverse().map(version => `
    <article class="understanding-history-item">
      <div><strong>v${version.version}</strong><time>${escapeHtml(version.created_at)}</time></div>
      <p class="${version.text ? '' : 'empty'}">${version.text ? escapeHtml(version.text) : '（空白版本）'}</p>
    </article>`).join('');
}

function updateUnderstandingSaveState() {
  const target = selectedPanelTarget();
  const changed = target && panelUnderstanding.value.trim() !== (target.understanding_text || '').trim();
  saveUnderstandingButton.disabled = !changed;
}

function renderCurrentUnderstanding(target) {
  if (!target) {
    understandingCurrentVersion.textContent = '请先生成子图';
    panelUnderstanding.value = '';
    panelUnderstanding.disabled = true;
    saveUnderstandingButton.disabled = true;
    renderUnderstandingHistory(null);
    return;
  }
  const version = target.understanding_current_version || 0;
  understandingCurrentVersion.textContent = version ? `v${version}` : '尚未填写';
  panelUnderstanding.disabled = false;
  panelUnderstanding.value = target.understanding_text || '';
  renderUnderstandingHistory(target);
  updateUnderstandingSaveState();
}

function renderCurrentTarget() {
  const target = selectedPanelTarget();
  deletePanelButton.disabled = !target;
  if (target) {
    editorCurrentVersion.textContent = target.current_version === 0
      ? 'v0 自动版'
      : `v${target.current_version} 人工版`;
    editorCurrentImage.src = target.current_url;
    editorCurrentImage.hidden = false;
    editorCurrentEmpty.hidden = true;
  } else {
    editorCurrentVersion.textContent = '未生成';
    editorCurrentImage.removeAttribute('src');
    editorCurrentImage.hidden = true;
    editorCurrentEmpty.hidden = false;
  }
  renderVersionHistory(target);
  renderCurrentUnderstanding(target);
}

function syncEditorCanvas() {
  const rect = editorImage.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  editorCanvas.width = Math.round(rect.width * ratio);
  editorCanvas.height = Math.round(rect.height * ratio);
  editorCanvas.style.width = `${rect.width}px`;
  editorCanvas.style.height = `${rect.height}px`;
  drawEditorSelection();
}

function drawEditorSelection() {
  const rect = editorCanvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const context = editorCanvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  if (!editorSelection || !editorImage.naturalWidth || !editorImage.naturalHeight) return;

  const scaleX = rect.width / editorImage.naturalWidth;
  const scaleY = rect.height / editorImage.naturalHeight;
  const x = editorSelection.x0 * scaleX;
  const y = editorSelection.y0 * scaleY;
  const width = (editorSelection.x1 - editorSelection.x0) * scaleX;
  const height = (editorSelection.y1 - editorSelection.y0) * scaleY;
  context.fillStyle = 'rgba(12, 23, 20, .42)';
  context.fillRect(0, 0, rect.width, rect.height);
  context.clearRect(x, y, width, height);
  context.strokeStyle = '#17a673';
  context.lineWidth = 2;
  context.setLineDash([7, 4]);
  context.strokeRect(x + 1, y + 1, Math.max(0, width - 2), Math.max(0, height - 2));
  context.setLineDash([]);
}

function updateCropPreview() {
  const selection = editorSelection;
  const width = selection ? selection.x1 - selection.x0 : 0;
  const height = selection ? selection.y1 - selection.y0 : 0;
  const valid = width >= 8 && height >= 8;
  confirmCropButton.disabled = !valid || !normalizedEditorLabel();
  if (!valid) {
    editorSelectionSize.textContent = '尚未框选';
    editorPreviewCanvas.hidden = true;
    editorPreviewEmpty.hidden = false;
    return;
  }

  const scale = Math.min(1, 600 / width, 280 / height);
  editorPreviewCanvas.width = Math.max(1, Math.round(width * scale));
  editorPreviewCanvas.height = Math.max(1, Math.round(height * scale));
  editorPreviewCanvas.getContext('2d').drawImage(
    editorImage,
    selection.x0,
    selection.y0,
    width,
    height,
    0,
    0,
    editorPreviewCanvas.width,
    editorPreviewCanvas.height,
  );
  editorSelectionSize.textContent = `${width} × ${height}px`;
  editorPreviewCanvas.hidden = false;
  editorPreviewEmpty.hidden = true;
}

function resetEditorSelection() {
  editorSelection = null;
  dragStart = null;
  drawEditorSelection();
  updateCropPreview();
  editorError.textContent = '';
}

function editorPoint(event) {
  const rect = editorCanvas.getBoundingClientRect();
  const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
  return {
    x: Math.round(x * editorImage.naturalWidth / rect.width),
    y: Math.round(y * editorImage.naturalHeight / rect.height),
  };
}

function openPanelEditor(jobId, recordId, resultRoot, preferredLabel = '', focusUnderstanding = false) {
  const result = resultSessions.get(jobId);
  const record = result?.records?.find(item => item.record_id === recordId);
  if (!record) return;
  activeResult = result;
  activeResultRoot = resultRoot;
  activeEditorRecord = record;
  editorSourceName.textContent = record.source;
  editorLabelOptions.innerHTML = (record.editable_labels || [])
    .map(label => `<option value="${escapeHtml(label)}"></option>`)
    .join('');
  editorLabel.value = preferredLabel || record.panel_targets?.[0]?.label || record.editable_labels?.[0] || 'A';
  editorError.textContent = '';
  resetEditorSelection();
  renderCurrentTarget();
  editorImage.src = record.whole_image_url;
  editorDialog.showModal();
  refreshIcons();
  if (focusUnderstanding) window.setTimeout(() => panelUnderstanding.focus(), 0);
}

function nextPendingRecordId(result, currentRecordId) {
  const document = result.documents?.find(item => item.record_ids?.includes(currentRecordId));
  const recordIds = document?.record_ids?.length
    ? document.record_ids
    : (result.records || []).map(record => record.record_id);
  const currentIndex = recordIds.indexOf(currentRecordId);
  for (let index = currentIndex + 1; index < recordIds.length; index += 1) {
    const candidate = result.records?.find(record => record.record_id === recordIds[index]);
    const verified = candidate?.review_status === 'verified' && candidate?.annotation_complete;
    if (candidate && !verified && candidate.review_status !== 'ambiguous') return candidate.record_id;
  }
  return currentRecordId;
}

function focusFigureRecord(resultRoot, recordId) {
  if (!recordId) return;
  const card = resultRoot.querySelector(`[data-record-id="${recordId}"]`);
  const carousel = card?.closest('[data-figure-carousel]');
  if (!card || !carousel) return;
  const slides = Array.from(carousel.querySelectorAll('.figure-result-card'));
  const index = slides.indexOf(card);
  if (index >= 0) scrollFigureCarousel(carousel, index, 'auto');
}

function rerenderResultRoot(resultRoot, result, focusRecordId = '') {
  if (!resultRoot || !result) return;
  resultRoot.innerHTML = renderResult(result);
  initializeFigureCarousels(resultRoot);
  focusFigureRecord(resultRoot, focusRecordId);
  refreshIcons();
}

async function updateFigureReview(jobId, recordId, resultRoot, status, annotationMode = 'panel_boxes') {
  const result = resultSessions.get(jobId);
  const record = result?.records?.find(item => item.record_id === recordId);
  if (!record) return;
  const message = status === 'verified' && annotationMode === 'no_split'
    ? '确认这张完整 Figure 不需要划分子图吗？\n\n自动候选框会保留在历史中，但不会进入真值；该样本会生成空 YOLO 标签，作为无子图目标样本。'
    : status === 'verified'
      ? '确认这张完整 Figure 的所有一级面板均已检查、补齐并删除错误框吗？\n\n确认后它会进入未划分训练素材库。'
      : '确认将这张 Figure 标记为存在语义歧义、暂不进入训练数据吗？';
  if (!window.confirm(message)) return;
  try {
    const response = await fetch(appUrl(`/api/jobs/${jobId}/figures/${recordId}/review`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, annotation_mode: annotationMode, reviewer: '', confirmed: true }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '整图审核状态保存失败');
    const panelCount = payload.annotation_mode === 'no_split' ? 0 : (payload.panel_targets?.length || 0);
    Object.assign(record, payload, { panel_count: panelCount });
    const focusRecordId = nextPendingRecordId(result, recordId);
    rerenderResultRoot(resultRoot, result, focusRecordId);
    updateNavigationJobFromResult(result);
    collectionStatus.textContent = payload.ready_for_training
      ? '该 Figure 已同步为可训练样本；完成一批后点击“整理训练数据”刷新总清单。'
      : '审核状态已保存；该 Figure 暂不进入训练数据。';
    collectionStatus.classList.remove('error');
  } catch (error) {
    window.alert(error.message);
  }
}

async function deleteCurrentPanel() {
  const target = selectedPanelTarget();
  if (!activeResult || !activeEditorRecord || !target) return;
  if (!window.confirm(`确认删除面板 ${target.label} 吗？\n\n历史图片不会被删除，之后仍可用同一标签重新框选。`)) return;
  deletePanelButton.disabled = true;
  editorError.textContent = '';
  try {
    const response = await fetch(
      appUrl(`/api/jobs/${activeResult.job_id}/figures/${activeEditorRecord.record_id}/panels`),
      {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label: target.label,
          expected_current_version: target.current_version,
          confirmed: true,
        }),
      },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '面板删除失败');
    Object.assign(activeEditorRecord, payload, { panel_count: payload.panel_targets?.length || 0 });
    editorLabel.value = activeEditorRecord.panel_targets?.[0]?.label || target.label;
    renderCurrentTarget();
    rerenderResultRoot(activeResultRoot, activeResult, activeEditorRecord.record_id);
    updateNavigationJobFromResult(activeResult);
    editorError.style.color = '#146b5c';
    editorError.textContent = `面板 ${target.label} 已从当前标注中移除，历史版本仍保留。`;
    collectionStatus.textContent = '标注已更新；完成一批后点击“整理训练数据”刷新总清单。';
  } catch (error) {
    editorError.style.color = '';
    editorError.textContent = error.message;
    renderCurrentTarget();
  }
}

function updateGalleryPanel(result) {
  if (!activeResultRoot) return;
  const card = activeResultRoot.querySelector(`[data-record-id="${activeEditorRecord.record_id}"]`);
  const panelGrid = card?.querySelector('.panel-grid');
  if (!panelGrid) return;
  const existing = panelGrid.querySelector(`[data-panel-label="${result.label}"]`);
  if (existing) {
    existing.outerHTML = renderPanelTarget(result, activeResult.job_id, activeEditorRecord.record_id);
  } else {
    panelGrid.querySelector('.panel-empty')?.remove();
    panelGrid.insertAdjacentHTML(
      'beforeend',
      renderPanelTarget(result, activeResult.job_id, activeEditorRecord.record_id),
    );
  }
}

async function savePanelUnderstanding() {
  const label = normalizedEditorLabel();
  const target = selectedPanelTarget();
  if (!activeResult || !activeEditorRecord || !target || !label) return;
  const text = panelUnderstanding.value.trim();
  const nextVersion = (target.understanding_current_version || 0) + 1;
  const action = text ? '保存当前图片理解' : '将图片理解清空';
  const confirmed = window.confirm(
    `确认${action}为面板 ${label} 的理解 v${nextVersion} 吗？\n\n之前的文字版本会全部保留。`,
  );
  if (!confirmed) return;

  saveUnderstandingButton.disabled = true;
  editorError.textContent = '';
  try {
    const response = await fetch(
      appUrl(`/api/jobs/${activeResult.job_id}/figures/${activeEditorRecord.record_id}/understanding-versions`),
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          text,
          expected_current_version: target.understanding_current_version || 0,
          confirmed: true,
        }),
      },
    );
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '图片理解保存失败');
    const index = activeEditorRecord.panel_targets.findIndex(item => item.label === label);
    activeEditorRecord.panel_targets[index] = result;
    renderCurrentTarget();
    updateGalleryPanel(result);
    editorError.style.color = '#146b5c';
    editorError.textContent = `面板 ${label} 的图片理解 v${result.understanding_current_version} 已保存。`;
  } catch (error) {
    editorError.style.color = '';
    editorError.textContent = error.message;
    updateUnderstandingSaveState();
  }
}

async function confirmManualCrop() {
  const label = normalizedEditorLabel();
  if (!activeResult || !activeEditorRecord || !editorSelection || !label) return;
  const target = selectedPanelTarget();
  const nextVersion = Math.max(...(target?.versions || []).map(item => item.version), 0) + 1;
  const confirmed = window.confirm(
    `确认用当前框选替换面板 ${label} 吗？\n\n系统将保存为 v${nextVersion}，已有版本不会删除。`,
  );
  if (!confirmed) return;

  confirmCropButton.disabled = true;
  resetSelectionButton.disabled = true;
  editorError.textContent = '';
  try {
    const response = await fetch(
      appUrl(`/api/jobs/${activeResult.job_id}/figures/${activeEditorRecord.record_id}/panel-versions`),
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          label,
          ...editorSelection,
          expected_current_version: target?.current_version ?? null,
          confirmed: true,
        }),
      },
    );
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '保存失败');

    const existingIndex = activeEditorRecord.panel_targets.findIndex(item => item.label === label);
    if (existingIndex >= 0) activeEditorRecord.panel_targets[existingIndex] = result;
    else activeEditorRecord.panel_targets.push(result);
    if (!activeEditorRecord.editable_labels.includes(label)) {
      activeEditorRecord.editable_labels.push(label);
      activeEditorRecord.editable_labels.sort();
    }
    editorLabelOptions.innerHTML = activeEditorRecord.editable_labels
      .map(item => `<option value="${escapeHtml(item)}"></option>`)
      .join('');
    activeEditorRecord.review_status = 'proposed';
    activeEditorRecord.annotation_mode = 'panel_boxes';
    activeEditorRecord.annotation_complete = false;
    activeEditorRecord.panel_count = activeEditorRecord.panel_targets.length;
    updateNavigationJobFromResult(activeResult);
    renderCurrentTarget();
    updateGalleryPanel(result);
    const card = activeResultRoot?.querySelector(`[data-record-id="${activeEditorRecord.record_id}"]`);
    if (card) {
      card.querySelector('.figure-status').textContent = recordStatus(activeEditorRecord);
      card.querySelector('.figure-status').classList.add('needs-review');
      card.querySelector('.figure-status').classList.remove('verified');
      card.querySelector('.edit-panel-button').textContent = '继续框选修正';
    }
    resetEditorSelection();
    editorError.textContent = `面板 ${label} 的 v${result.current_version} 已保存并设为当前版本。`;
    editorError.style.color = '#146b5c';
    collectionStatus.textContent = '人工框已同步到训练素材库；整图确认完成后才会生成 YOLO 真值标签。';
  } catch (error) {
    editorError.style.color = '';
    editorError.textContent = error.message;
    updateCropPreview();
  } finally {
    resetSelectionButton.disabled = false;
  }
}

setSidebarCollapsed(false);
sidebarToggleButton.addEventListener('click', () => {
  setSidebarCollapsed(!document.body.classList.contains('sidebar-collapsed'));
});

attachButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', updateSelection);
removeFileButton.addEventListener('click', () => { fileInput.value = ''; updateSelection(); });
messages.addEventListener('click', event => {
  const openJobButton = event.target.closest('.open-job-button');
  if (openJobButton) {
    openJobResult(openJobButton.dataset.jobId);
    return;
  }
  const sourcePageButton = event.target.closest('.source-page-compare-button');
  if (sourcePageButton) {
    openPdfReview(sourcePageButton.dataset.jobId, sourcePageButton.dataset.recordId);
    return;
  }
  const sourcePdfButton = event.target.closest('.source-pdf-button');
  if (sourcePdfButton) {
    openSourcePdf(sourcePdfButton.dataset.jobId, sourcePdfButton.dataset.recordId);
    return;
  }
  const verifyButton = event.target.closest('.verify-figure-button');
  if (verifyButton) {
    updateFigureReview(
      verifyButton.dataset.jobId,
      verifyButton.dataset.recordId,
      verifyButton.closest('.message-body'),
      'verified',
      'panel_boxes',
    );
    return;
  }
  const noSplitButton = event.target.closest('.no-split-figure-button');
  if (noSplitButton) {
    updateFigureReview(
      noSplitButton.dataset.jobId,
      noSplitButton.dataset.recordId,
      noSplitButton.closest('.message-body'),
      'verified',
      'no_split',
    );
    return;
  }
  const ambiguousButton = event.target.closest('.ambiguous-figure-button');
  if (ambiguousButton) {
    updateFigureReview(
      ambiguousButton.dataset.jobId,
      ambiguousButton.dataset.recordId,
      ambiguousButton.closest('.message-body'),
      'ambiguous',
    );
    return;
  }
  const understandingButton = event.target.closest('.understanding-button');
  if (understandingButton) {
    openPanelEditor(
      understandingButton.dataset.jobId,
      understandingButton.dataset.recordId,
      understandingButton.closest('.message-body'),
      understandingButton.dataset.panelLabel,
      true,
    );
    return;
  }
  const button = event.target.closest('.edit-panel-button');
  if (!button) return;
  openPanelEditor(button.dataset.jobId, button.dataset.recordId, button.closest('.message-body'));
});
closeEditorButton.addEventListener('click', () => editorDialog.close());
closeFolderBrowserButton.addEventListener('click', () => folderBrowserDialog.close());
browseFolderButton.addEventListener('click', openFolderBrowser);
folderUpButton.addEventListener('click', () => {
  if (folderBrowserState?.parent) navigateServerFolder(folderBrowserState.parent);
});
folderRootList.addEventListener('click', event => {
  const button = event.target.closest('[data-folder-path]');
  if (button && !button.disabled) navigateServerFolder(button.dataset.folderPath);
});
folderDirectoryList.addEventListener('click', event => {
  const button = event.target.closest('[data-folder-path]');
  if (button) navigateServerFolder(button.dataset.folderPath);
});
chooseFolderButton.addEventListener('click', () => {
  if (!folderBrowserState?.current) return;
  folderInput.value = folderBrowserState.current;
  folderBrowserDialog.close();
  setBatchStatus(`已选择服务器目录：${folderBrowserState.current}`);
});
resetSelectionButton.addEventListener('click', resetEditorSelection);
deletePanelButton.addEventListener('click', deleteCurrentPanel);
confirmCropButton.addEventListener('click', confirmManualCrop);
saveUnderstandingButton.addEventListener('click', savePanelUnderstanding);
panelUnderstanding.addEventListener('input', updateUnderstandingSaveState);
folderStartButton.addEventListener('click', startFolderBatch);
refreshBatchesButton.addEventListener('click', () => loadBatchHistory(true));
historyRefreshButton?.addEventListener('click', () => loadBatchHistory(true));
rebuildCollectionButton.addEventListener('click', rebuildTrainingCollection);
batchHistory.addEventListener('toggle', event => {
  const details = event.target.closest?.('.analysis-batch');
  if (details?.open) loadNavigationBatch(details);
}, true);
batchHistory.addEventListener('click', event => {
  const button = event.target.closest('[data-navigation-job-id]');
  if (!button) return;
  openJobResult(button.dataset.navigationJobId, true);
});
batchItems.addEventListener('click', event => {
  const button = event.target.closest('.open-job-button');
  if (button) openJobResult(button.dataset.jobId);
});
editorLabel.addEventListener('input', () => {
  editorLabel.value = normalizedEditorLabel();
  editorError.textContent = '';
  editorError.style.color = '';
  renderCurrentTarget();
  updateCropPreview();
});
editorImage.addEventListener('load', () => {
  syncEditorCanvas();
  updateCropPreview();
});
editorCanvas.addEventListener('pointerdown', event => {
  if (!editorImage.naturalWidth || !editorImage.naturalHeight) return;
  event.preventDefault();
  editorCanvas.setPointerCapture(event.pointerId);
  dragStart = editorPoint(event);
  editorSelection = { x0: dragStart.x, y0: dragStart.y, x1: dragStart.x, y1: dragStart.y };
  drawEditorSelection();
});
editorCanvas.addEventListener('pointermove', event => {
  if (!dragStart) return;
  event.preventDefault();
  const point = editorPoint(event);
  editorSelection = {
    x0: Math.min(dragStart.x, point.x),
    y0: Math.min(dragStart.y, point.y),
    x1: Math.max(dragStart.x, point.x),
    y1: Math.max(dragStart.y, point.y),
  };
  drawEditorSelection();
  editorSelectionSize.textContent = `${editorSelection.x1 - editorSelection.x0} × ${editorSelection.y1 - editorSelection.y0}px`;
});
editorCanvas.addEventListener('pointerup', event => {
  if (!dragStart) return;
  event.preventDefault();
  dragStart = null;
  updateCropPreview();
});
editorCanvas.addEventListener('pointercancel', () => {
  dragStart = null;
  updateCropPreview();
});
window.addEventListener('resize', () => {
  if (editorDialog.open) syncEditorCanvas();
  document.querySelectorAll('[data-figure-carousel]').forEach(carousel => {
    scrollFigureCarousel(carousel, Number(carousel.dataset.currentIndex || 0), 'auto');
  });
});
document.querySelector('#newChat').addEventListener('click', () => {
  if (editorDialog.open) editorDialog.close();
  messages.innerHTML = initialMessage;
  fileInput.value = '';
  prompt.value = '';
  activeResult = null;
  activeResultRoot = null;
  activeEditorRecord = null;
  resultSessions.clear();
  batchHistory.querySelectorAll('.analysis-job-item.active').forEach(item => item.classList.remove('active'));
  updateSelection();
  scrollToBottom();
});

composer.addEventListener('submit', async (event) => {
  event.preventDefault();
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  const note = prompt.value.trim();
  appendMessage('user', `<div class="file-line">${icon('files')}<span>${files.length === 1 ? escapeHtml(files[0].name) : `批量上传 ${files.length} 篇 PDF`}</span></div>${note ? `<p>${escapeHtml(note)}</p>` : ''}`);
  const form = new FormData();
  files.forEach(file => form.append('files', file, file.name));
  form.append('concurrency', String(Number(concurrencyInput.value || 1)));
  sendButton.disabled = true;
  attachButton.disabled = true;
  setBatchStatus(`正在上传 ${files.length} 篇 PDF…`);

  try {
    const response = await fetch(appUrl('/api/batches/upload'), { method: 'POST', body: form });
    const state = await response.json();
    if (!response.ok) throw new Error(state.detail || '批次创建失败');
    renderBatchState(state);
    startBatchPolling(state.batch_id);
    loadBatchHistory();
  } catch (error) {
    setBatchStatus(error.message, true);
  } finally {
    fileInput.value = '';
    prompt.value = '';
    attachButton.disabled = false;
    updateSelection();
    refreshIcons();
    scrollToBottom();
  }
});

fetch(appUrl('/api/health'))
  .then(response => { if (!response.ok) throw new Error(); return response.json(); })
  .then(data => {
    serviceStatus.textContent = `工具已就绪 · 300 DPI · 最多并发 ${data.max_workers}`;
    serviceStatus.classList.add('online');
    concurrencyInput.max = String(data.max_workers || 32);
    concurrencyInput.value = String(data.default_workers || 4);
  })
  .catch(() => { serviceStatus.textContent = '服务未连接'; });

loadBatchHistory();
loadTrainingCollection();

window.addEventListener('DOMContentLoaded', refreshIcons);
