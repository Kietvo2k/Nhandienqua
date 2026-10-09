/** Trang "Thử model" (#/lab): chạy model ONNX thật trên ảnh / video / webcam ngay trong trình duyệt. */
import { heatCanvas } from './heatmap'
import { FruitLab, type Finding, type InspectResult, type ModelInfo } from './runtime'
import { Smoother } from './track'
import './lab.css'

const VI: Record<string, string> = { mango: 'Xoài', banana: 'Chuối', dragon_fruit: 'Thanh long', jackfruit: 'Mít', sweet_potato: 'Khoai lang' }

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  return e
}

function modelCard(title: string, sub: string, info: ModelInfo | null, provider: string, missing: string) {
  const c = el('section', `lab-model ${info ? 'ok' : 'off'}`)
  c.append(el('h3', '', title), el('p', 'muted', sub))
  if (info) {
    c.append(el('p', 'mono', `${info.file.split('/').pop()} · ${(info.bytes / 1e6).toFixed(1)} MB · ${info.imgsz.join('×')}`))
    c.append(el('p', 'mono', `${info.names.length} lớp · ONNX Runtime Web (${provider.toUpperCase()})`))
  } else {
    c.append(el('p', 'warn', missing))
  }
  return c
}

export async function renderLab(root: HTMLElement) {
  document.body.classList.add('lab-mode')
  document.title = 'Thử model · Nhandienqua'
  root.replaceChildren()
  const page = el('div', 'lab')
  const head = el('header', 'lab-head')
  const back = el('a', 'btn ghost', '← Mô phỏng')
  back.href = '#/'
  head.append(el('h1', '', 'Thử model nhận diện quả'), el('span', 'muted', 'Model ONNX thật (train trên Colab) chạy trong trình duyệt'), back)
  const status = el('output', 'lab-status mono', 'Đang khởi động…')
  head.appendChild(status)
  page.appendChild(head)

  const models = el('div', 'lab-models')
  const controls = el('div', 'lab-controls')
  const stage = el('div', 'lab-stage')
  const canvas = el('canvas', 'lab-canvas')
  const video = el('video')
  video.muted = true
  video.playsInline = true
  video.loop = true
  video.hidden = true
  const empty = el('p', 'lab-empty muted', 'Chọn ảnh mẫu bên dưới, hoặc thả ảnh / video của bạn vào đây.')
  stage.append(canvas, empty, video)
  const side = el('aside', 'lab-side')
  const timing = el('div', 'lab-timing mono')
  const table = el('table', 'lab-table')
  table.innerHTML = '<thead><tr><th>#</th><th>Quả</th><th>Tin cậy</th><th>P(hư)</th><th>Quyết định</th></tr></thead><tbody></tbody>'
  side.append(el('h2', '', 'Kết quả'), timing, table)
  const gallery = el('div', 'lab-gallery')
  page.append(models, controls, el('div', 'lab-main'), gallery)
  page.querySelector('.lab-main')!.append(stage, side)
  root.appendChild(page)

  // ---- điều khiển
  const fileIn = el('input')
  fileIn.type = 'file'
  fileIn.accept = 'image/*,video/*'
  fileIn.hidden = true
  const pick = el('button', 'btn', 'Chọn ảnh / video…')
  pick.type = 'button'
  pick.onclick = () => fileIn.click()
  const cam = el('button', 'btn ghost', 'Bật webcam')
  cam.type = 'button'
  const stop = el('button', 'btn ghost', 'Dừng video')
  stop.type = 'button'
  stop.hidden = true
  const slider = (label: string, value: number) => {
    const wrap = el('label', 'lab-slider')
    const input = el('input')
    input.type = 'range'
    input.min = '0.05'
    input.max = '0.95'
    input.step = '0.05'
    input.value = String(value)
    const out = el('output', 'mono', value.toFixed(2))
    input.oninput = () => (out.textContent = Number(input.value).toFixed(2))
    wrap.append(el('span', '', label), input, out)
    return { wrap, input }
  }
  const sConf = slider('Ngưỡng phát hiện', 0.35)
  const sDef = slider('Ngưỡng hư', 0.5)
  const heatSeg = el('div', 'lab-seg')
  heatSeg.setAttribute('role', 'group')
  heatSeg.setAttribute('aria-label', 'Bản đồ nhiệt')
  let heatMode: 'off' | 'det' | 'defect' = 'off'
  const heatBtns = (
    [['off', 'Tắt'], ['det', 'Phát hiện'], ['defect', 'Hư hỏng']] as const
  ).map(([k, label]) => {
    const b = el('button', k === 'off' ? 'on' : '', label)
    b.type = 'button'
    b.onclick = () => {
      heatMode = k
      heatBtns.forEach((x) => x.classList.toggle('on', x === b))
      legend.hidden = k === 'off'
      legend.textContent = k === 'det'
        ? 'Phát hiện: chỗ model "thấy quả" (đỏ = chắc chắn có quả).'
        : 'Hư hỏng: vùng trên quả khiến model nghĩ là HƯ (đỏ = bằng chứng mạnh; quả tốt gần như không tô).'
      redraw()
    }
    return b
  })
  heatSeg.append(el('span', 'muted', 'Bản đồ nhiệt'), ...heatBtns)
  const smoothLbl = el('label', 'lab-check')
  const smoothIn = el('input')
  smoothIn.type = 'checkbox'
  smoothIn.checked = true
  smoothLbl.append(smoothIn, el('span', '', 'Làm mượt theo thời gian (video/webcam)'))
  const legend = el('p', 'lab-legend muted')
  legend.hidden = true
  controls.append(pick, cam, stop, fileIn, sConf.wrap, sDef.wrap, heatSeg, smoothLbl, legend)

  let lab: FruitLab
  try {
    lab = await FruitLab.load((s) => (status.textContent = s))
  } catch (e) {
    status.textContent = `Lỗi nạp model: ${(e as Error).message}. Chạy: python vision/scripts/export_web.py`
    return
  }
  const m = lab.manifest
  models.append(
    modelCard('1 · Phát hiện quả', 'YOLO26n — 5 loại quả', m.detect, lab.provider, 'Chưa có — chạy export_web.py'),
    modelCard('2 · Tốt / hư', 'YOLO26n-cls — 10 lớp (quả × tốt/hư)', m.quality, lab.provider, 'Chưa có — chạy export_web.py'),
    modelCard('3 · Mức sấy', 'YOLO26n-cls — chưa đạt / đạt / quá khô', m.drying, lab.provider,
      'Chưa train: cần gán nhãn ảnh trong vision/data/drying_unlabeled/ (xem HANDOFF.md)'),
  )
  status.textContent = `Sẵn sàng · ${lab.provider.toUpperCase()}`

  const g = canvas.getContext('2d')!
  type Shown = Finding & { id?: number; partial?: boolean }
  let last: { src: CanvasImageSource; W: number; H: number; r: InspectResult; shown: Shown[] } | null = null
  const redraw = () => last && draw(last.src, last.W, last.H, last.r, last.shown)
  const draw = (src: CanvasImageSource, W: number, H: number, r: InspectResult, shown: Shown[] = r.findings) => {
    last = { src, W, H, r, shown }
    canvas.width = W
    canvas.height = H
    g.drawImage(src, 0, 0, W, H)
    g.imageSmoothingEnabled = true
    if (heatMode === 'det') {
      const { heat, lb, stride } = r.detHeat
      const hc = heatCanvas(heat.data, heat.w, heat.h, (v) => Math.min(0.7, v * 0.9))
      g.drawImage(hc, lb.padX / stride, lb.padY / stride, lb.w / stride, lb.h / stride, 0, 0, W, H)
    }
    if (heatMode === 'defect') {
      for (const f of shown) {
        if (!f.cam || f.partial) continue
        // kiểu Grad-CAM: tô cả dải màu trên vùng quả (xanh = ít, đỏ = bằng chứng "hư" mạnh); căn bậc hai để
        // vùng phụ vẫn thấy; độ đậm cả lớp theo P(hư) → quả chắc là tốt gần như không bị tô
        const m = f.cam.map
        const hi = Math.max(1e-6, ...m)
        const vals = Float32Array.from(m, (v) => Math.sqrt(Math.max(0, v) / hi))
        const k = Math.min(1, 0.1 + f.pDefect)
        const hc = heatCanvas(vals, f.cam.w, f.cam.h, (v) => (0.25 + 0.4 * v) * k)
        const [x1, y1, x2, y2] = f.box
        g.save()
        g.beginPath()
        g.rect(x1, y1, x2 - x1, y2 - y1)
        g.clip()
        g.drawImage(hc, ...f.cam.region)
        g.restore()
      }
    }
    const lw = Math.max(2, W / 400)
    const fs = Math.max(12, Math.round(W / 55))
    shown.forEach((f, i) => {
      const [x1, y1, x2, y2] = f.box
      const color = f.partial ? '#9aa4ad' : f.action === 'reject' ? '#ff5d4d' : '#38d27a'
      g.strokeStyle = color
      g.lineWidth = lw
      g.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const verdict = f.partial ? 'đang vào' : f.action === 'reject' ? 'HƯ' : 'TỐT'
      const tag = `${f.id ?? i + 1}. ${VI[f.fruit] ?? f.fruit} ${verdict} ${Math.round(f.detConf * 100)}%`
      g.font = `600 ${fs}px "Segoe UI", system-ui, sans-serif`
      const tw = g.measureText(tag).width + 8
      g.fillStyle = color
      g.fillRect(x1, Math.max(0, y1 - fs - 6), tw, fs + 6)
      g.fillStyle = '#0b0f14'
      g.fillText(tag, x1 + 4, Math.max(fs, y1 - 5))
    })
    empty.hidden = true
  }
  const report = (r: InspectResult, fps?: number, shown: Shown[] = r.findings) => {
    const body = table.tBodies[0]
    body.replaceChildren(
      ...shown.map((f: Shown, i) => {
        const tr = el('tr')
        tr.append(
          el('td', 'mono', String(f.id ?? i + 1)),
          el('td', '', VI[f.fruit] ?? f.fruit),
          el('td', 'mono', `${(f.detConf * 100).toFixed(0)}%`),
          el('td', 'mono', f.pDefect.toFixed(2)),
          f.partial ? el('td', 'muted', 'Đang vào khung') : el('td', f.action === 'reject' ? 'bad' : 'ok', f.action === 'reject' ? 'Loại (hư)' : 'Nhận (tốt)'),
        )
        return tr
      }),
    )
    if (!shown.length) body.appendChild(el('tr')).appendChild(el('td', 'muted', 'Không phát hiện quả nào')).setAttribute('colspan', '5')
    timing.textContent =
      `Tiền xử lý ${r.ms.pre.toFixed(1)} ms · phát hiện ${r.ms.detect.toFixed(1)} ms · tốt/hư ${r.ms.quality.toFixed(1)} ms` +
      ` (${r.findings.length} quả) · tổng ${r.ms.total.toFixed(1)} ms${fps ? ` · ${fps.toFixed(1)} FPS` : ''}`
  }

  // ---- ảnh
  const runImage = async (url: string) => {
    stopVideo()
    const img = new Image()
    img.src = url
    await img.decode()
    const r = await lab.inspect(img, img.naturalWidth, img.naturalHeight, Number(sConf.input.value), Number(sDef.input.value))
    draw(img, img.naturalWidth, img.naturalHeight, r)
    report(r)
    lastImage = url
  }
  let lastImage: string | null = null
  sConf.input.onchange = sDef.input.onchange = () => lastImage && !video.srcObject && video.hidden && runImage(lastImage)

  // ---- video / webcam
  let loopId = 0
  const stopVideo = () => {
    loopId++
    video.pause()
    const s = video.srcObject as MediaStream | null
    s?.getTracks().forEach((t) => t.stop())
    video.srcObject = null
    stop.hidden = true
  }
  const smoother = new Smoother()
  const runVideo = async () => {
    smoother.reset()
    const id = ++loopId
    stop.hidden = false
    try {
      await video.play()
    } catch (e) {
      // trình duyệt chặn phát (tab bị ẩn để tiết kiệm pin, hoặc chưa có thao tác người dùng)
      status.textContent = `Không phát được video: ${(e as Error).message.split('.')[0]}. Mở tab rồi bấm lại.`
      stop.hidden = true
      return
    }
    let fps = 0
    while (id === loopId) {
      const t = performance.now()
      const r = await lab.inspect(video, video.videoWidth, video.videoHeight, Number(sConf.input.value), Number(sDef.input.value))
      if (id !== loopId) break
      const shown = smoothIn.checked ? smoother.update(r.findings, video.videoWidth, video.videoHeight) : r.findings
      draw(video, video.videoWidth, video.videoHeight, r, shown)
      fps = fps ? 0.9 * fps + 100 / (performance.now() - t) : 1000 / (performance.now() - t)
      report(r, fps, shown)
      await new Promise((res) => requestAnimationFrame(res))
    }
  }
  stop.onclick = stopVideo
  cam.onclick = async () => {
    stopVideo()
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
      runVideo()
    } catch (e) {
      status.textContent = `Không mở được webcam: ${(e as Error).message}`
    }
  }
  const openFile = (f: File) => {
    const url = URL.createObjectURL(f)
    if (f.type.startsWith('video/')) {
      stopVideo()
      video.src = url
      runVideo()
    } else runImage(url)
  }
  fileIn.onchange = () => fileIn.files?.[0] && openFile(fileIn.files[0])
  stage.ondragover = (e) => e.preventDefault()
  stage.ondrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files?.[0]
    if (f) openFile(f)
  }

  // ---- mẫu tập test
  try {
    const s = (await (await fetch('/lab-samples/samples.json')).json()) as { images: string[]; video: string | null }
    gallery.append(el('h2', '', 'Mẫu từ tập TEST (model chưa thấy khi train)'))
    const row = el('div', 'lab-thumbs')
    for (const name of s.images) {
      const b = el('button', 'lab-thumb')
      b.type = 'button'
      b.title = name
      const im = el('img')
      im.src = `/lab-samples/${name}`
      im.alt = name.startsWith('that') ? 'Ảnh thật' : 'Ảnh tổng hợp băng chuyền'
      im.loading = 'lazy'
      b.append(im, el('span', '', name.startsWith('that') ? 'thật' : 'băng chuyền'))
      b.onclick = () => runImage(`/lab-samples/${name}`)
      row.appendChild(b)
    }
    if (s.video) {
      const b = el('button', 'lab-thumb video', '▶ Video băng chuyền thử')
      b.type = 'button'
      b.onclick = () => {
        stopVideo()
        video.src = `/lab-samples/${s.video}`
        runVideo()
      }
      row.appendChild(b)
    }
    gallery.appendChild(row)
  } catch {
    /* chưa có mẫu */
  }
  Object.assign(window, { __lab: { lab, runImage } })
}
