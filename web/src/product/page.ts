/**
 * Trang sản phẩm mở từ QR: #/p/<dữ liệu mã hoá>.
 * Dữ liệu từ URL là KHÔNG tin cậy → decodeProduct kiểm tra cấu trúc, chỉ gán bằng textContent.
 */
import { FRUITS } from '../sim/params'
import { decodeProduct, type ProductInfo } from '../sim/trace'
import './page.css'

const hex = (n: number) => `#${n.toString(16).padStart(6, '0')}`

/** Hình minh hoạ túi sản phẩm (SVG tĩnh, chỉ phụ thuộc loại quả hợp lệ). */
function pouchSvg(info: ProductInfo): SVGSVGElement {
  const c = hex(FRUITS[info.fruit].color)
  const ns = 'http://www.w3.org/2000/svg'
  const svg = document.createElementNS(ns, 'svg')
  svg.setAttribute('viewBox', '0 0 200 240')
  svg.setAttribute('role', 'img')
  svg.setAttribute('aria-label', `Hình ảnh sản phẩm: ${info.name}`)
  const el = (tag: string, attrs: Record<string, string>) => {
    const e = document.createElementNS(ns, tag)
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v)
    svg.appendChild(e)
    return e
  }
  el('path', { d: 'M40 30 h120 l10 190 q0 10 -10 10 h-120 q-10 0 -10 -10 z', fill: '#f4efe4', stroke: '#c9bfa8', 'stroke-width': '2' })
  el('rect', { x: '40', y: '22', width: '120', height: '14', rx: '3', fill: '#d9cfb8' })
  el('rect', { x: '58', y: '70', width: '84', height: '84', rx: '12', fill: '#ffffff', stroke: '#d9cfb8' })
  for (let i = 0; i < 6; i++) {
    el('ellipse', { cx: String(78 + (i % 3) * 22), cy: String(96 + Math.floor(i / 3) * 30), rx: '15', ry: '9', fill: c, opacity: '0.9', transform: `rotate(${(i * 37) % 60 - 30} ${78 + (i % 3) * 22} ${96 + Math.floor(i / 3) * 30})` })
  }
  const t = el('text', { x: '100', y: '182', 'text-anchor': 'middle', 'font-size': '15', 'font-weight': '700', fill: '#3b2f1e' })
  t.textContent = info.name
  const m = el('text', { x: '100', y: '204', 'text-anchor': 'middle', 'font-size': '12', fill: '#6b5d45' })
  m.textContent = `${info.massG} g`
  return svg
}

const fmt = (iso: string) => {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('vi-VN', { hour12: false })
}

function h<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string, cls?: string) {
  const e = document.createElement(tag)
  if (text !== undefined) e.textContent = text
  if (cls) e.className = cls
  return e
}

export function renderProductPage(root: HTMLElement, encoded: string) {
  document.body.classList.add('product-mode')
  root.replaceChildren()
  const info = decodeProduct(encoded)
  const page = h('main', undefined, 'pp')
  root.appendChild(page)
  if (!info) {
    page.append(h('h1', 'Không đọc được mã QR'), h('p', 'Dữ liệu trong đường dẫn bị hỏng hoặc không đúng định dạng.', 'pp-muted'))
    return
  }
  document.title = `${info.name} · ${info.trace}`

  const head = h('header', undefined, 'pp-head')
  const fig = h('figure', undefined, 'pp-fig')
  fig.appendChild(pouchSvg(info))
  const title = h('div')
  title.append(h('p', 'Thông tin truy xuất nguồn gốc', 'pp-kicker'), h('h1', info.name), h('p', `Mã lô ${info.trace}`, 'pp-mono'))
  head.append(fig, title)
  page.appendChild(head)

  // 7 thông tin tối thiểu — khoản 5 Điều 6 Thông tư 02/2024/TT-BKHCN
  const req = h('section', undefined, 'pp-card')
  req.appendChild(h('h2', 'Thông tin sản phẩm'))
  const dl = h('dl', undefined, 'pp-dl')
  const [y, mo, d] = info.expiry.split('-')
  const rows: [string, string, string?][] = [
    ['Tên sản phẩm', info.name],
    ['Đơn vị sản xuất', info.producer],
    ['Địa chỉ', info.address],
    ['Thời gian sản xuất', fmt(info.producedAt), 'mono'],
    ['Số sê-ri', info.serial, 'mono'],
    ['Hạn sử dụng', `${d}/${mo}/${y}`, 'mono'],
    ['Khối lượng tịnh', `${info.massG} g`, 'mono'],
  ]
  for (const [k, v, cls] of rows) dl.append(h('dt', k), h('dd', v, cls))
  req.appendChild(dl)
  page.appendChild(req)

  const steps = h('section', undefined, 'pp-card')
  steps.appendChild(h('h2', 'Hành trình sản xuất'))
  const ol = h('ol', undefined, 'pp-steps')
  for (const s of info.steps) {
    const li = h('li')
    li.append(h('b', s.name), h('time', fmt(s.at)))
    ol.appendChild(li)
  }
  steps.appendChild(ol)
  page.appendChild(steps)

  const qc = h('section', undefined, 'pp-card')
  qc.appendChild(h('h2', 'Kết quả kiểm nghiệm lô'))
  const grid = h('div', undefined, 'pp-qc')
  const stat = (label: string, value: string) => {
    const d = h('div')
    d.append(h('span', label), h('b', value))
    return d
  }
  grid.append(
    stat('Mẫu đạt', `${info.qc.passed}/${info.qc.samples}`),
    stat('Độ ẩm TB', `${info.qc.moisture}%`),
    stat('Hoạt độ nước aw', String(info.qc.aw)),
  )
  qc.appendChild(grid)
  page.appendChild(qc)

  const vid = h('section', undefined, 'pp-card')
  vid.appendChild(h('h2', 'Video sản phẩm'))
  const video = document.createElement('video')
  video.controls = true
  video.muted = true
  video.playsInline = true
  video.loop = true
  video.src = `/videos/${encodeURIComponent(info.trace)}.webm`
  const miss = h('p', 'Chưa có video cho lô này (video được ghi khi lô vào máy đóng gói trên máy chạy mô phỏng).', 'pp-muted')
  miss.hidden = true
  video.addEventListener('error', () => {
    video.hidden = true
    miss.hidden = false
  })
  video.addEventListener('loadeddata', () => video.play().catch(() => {}))
  vid.append(video, miss)
  page.appendChild(vid)

  page.appendChild(h('p', 'Trang minh hoạ của đồ án mô phỏng — không phải sản phẩm thương mại.', 'pp-muted pp-foot'))
}
