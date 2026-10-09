/**
 * Một WebGLRenderer, nhiều khung nhìn (scissor/viewport): khung chính + 8 ô camera CCTV.
 * Không đọc ngược pixel → nhẹ. Lớp phủ 2D vẽ bbox / nhãn AI GIẢ LẬP.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { FactorySim } from '../sim/engine'
import { FRUITS } from '../sim/params'
import type { Actors } from '../scene/actors'
import { CAMERAS, type CamDef } from '../scene/layout'

interface Tile {
  def: CamDef
  cam: THREE.PerspectiveCamera
  el: HTMLElement
  overlay: HTMLCanvasElement
}

export type FrameHook = (def: CamDef, src: HTMLCanvasElement, rect: { x: number; y: number; w: number; h: number }) => void

const fmtClock = (d: Date) => d.toLocaleTimeString('vi-VN', { hour12: false })

export class Views {
  readonly renderer: THREE.WebGLRenderer
  readonly mainCam: THREE.PerspectiveCamera
  readonly controls: OrbitControls
  private tiles: Tile[] = []
  private mainOverlay: HTMLCanvasElement
  focused: CamDef | null = null
  private clock = ''
  private clockSec = -1
  /** gọi ngay sau khi vẽ một ô camera (dùng để ghi video) */
  onTile: FrameHook | null = null

  private canvas: HTMLCanvasElement
  private mainEl: HTMLElement

  constructor(canvas: HTMLCanvasElement, mainEl: HTMLElement, tileHost: HTMLElement, mainOverlay: HTMLCanvasElement) {
    this.canvas = canvas
    this.mainEl = mainEl
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' })
    this.renderer.setClearColor(0x000000, 0)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
    this.renderer.shadowMap.enabled = true
    // r18x: PCFSoftShadowMap đã bị gỡ (cảnh báo console) — PCFShadowMap nay đã lọc mềm
    this.renderer.shadowMap.type = THREE.PCFShadowMap
    this.renderer.shadowMap.autoUpdate = false
    this.renderer.setScissorTest(true)
    this.mainOverlay = mainOverlay

    this.mainCam = new THREE.PerspectiveCamera(45, 1, 0.1, 300)
    this.mainCam.layers.enable(1) // thấy mô hình CCTV
    this.mainCam.position.set(-6, 15, 19)
    this.controls = new OrbitControls(this.mainCam, mainEl)
    this.controls.target.set(4, 0, -3)
    this.controls.enableDamping = true
    this.controls.maxPolarAngle = Math.PI * 0.48
    this.controls.minDistance = 3
    this.controls.maxDistance = 70

    for (const def of CAMERAS) {
      const cam = new THREE.PerspectiveCamera(def.fov, 16 / 9, 0.05, 120)
      cam.position.set(...def.pos)
      cam.lookAt(...def.look)
      const el = document.createElement('button')
      el.className = 'cam-tile'
      el.type = 'button'
      el.title = `Phóng to ${def.id} — ${def.name}`
      el.setAttribute('aria-label', `${def.id} ${def.name}`)
      const overlay = document.createElement('canvas')
      overlay.className = 'cam-overlay'
      el.appendChild(overlay)
      el.addEventListener('click', () => this.focus(this.focused?.id === def.id ? null : def))
      tileHost.appendChild(el)
      this.tiles.push({ def, cam, el, overlay })
    }
  }

  focus(def: CamDef | null) {
    this.focused = def
    this.controls.enabled = !def
    for (const t of this.tiles) t.el.classList.toggle('active', t.def.id === def?.id)
    this.mainEl.dispatchEvent(new CustomEvent('focus-change', { detail: def }))
  }

  render(scene: THREE.Scene, sim: FactorySim, actors: Actors) {
    const r = this.renderer
    const W = window.innerWidth
    const H = window.innerHeight
    const size = r.getSize(new THREE.Vector2())
    if (size.x !== W || size.y !== H) r.setSize(W, H, false)
    this.controls.update()
    r.shadowMap.needsUpdate = true
    // định dạng giờ (Intl) tốn kém → chỉ tính lại khi sang giây mới
    const sec = Math.floor((sim.state.startDate.getTime() + sim.state.time * 1000) / 1000)
    if (sec !== this.clockSec) {
      this.clockSec = sec
      this.clock = fmtClock(sim.dateAt(sim.state.time))
    }
    // cập nhật ma trận cảnh một lần/khung thay vì 9 lần (mỗi lần render một khung nhìn)
    scene.matrixWorldAutoUpdate = true
    scene.updateMatrixWorld()
    scene.matrixWorldAutoUpdate = false

    // khung chính
    const mr = this.mainEl.getBoundingClientRect()
    const focusedTile = this.focused && this.tiles.find((t) => t.def.id === this.focused!.id)
    const mainCam = focusedTile ? focusedTile.cam : this.mainCam
    this.drawView(scene, mainCam, mr, H)
    this.drawOverlay(this.mainOverlay, mr, focusedTile ? focusedTile.def : null, mainCam, sim, actors, true)

    // các ô camera
    for (const t of this.tiles) {
      const rect = t.el.getBoundingClientRect()
      if (rect.width < 4 || rect.bottom < 0 || rect.top > H) continue
      this.drawView(scene, t.cam, rect, H)
      this.onTile?.(t.def, this.canvas, { x: rect.left, y: rect.top, w: rect.width, h: rect.height })
      this.drawOverlay(t.overlay, rect, t.def, t.cam, sim, actors, false)
    }
  }

  private drawView(scene: THREE.Scene, cam: THREE.PerspectiveCamera, rect: DOMRect, H: number) {
    const w = Math.max(1, rect.width)
    const h = Math.max(1, rect.height)
    cam.aspect = w / h
    cam.updateProjectionMatrix()
    const y = H - rect.bottom
    this.renderer.setViewport(rect.left, y, w, h)
    this.renderer.setScissor(rect.left, y, w, h)
    this.renderer.render(scene, cam)
  }

  private drawOverlay(c: HTMLCanvasElement, rect: DOMRect, def: CamDef | null, cam: THREE.PerspectiveCamera, sim: FactorySim, actors: Actors, big: boolean) {
    const dpr = Math.min(window.devicePixelRatio, 2)
    const w = Math.round(rect.width * dpr)
    const h = Math.round(rect.height * dpr)
    if (c.width !== w || c.height !== h) {
      c.width = w
      c.height = h
    }
    // ô không có lớp phủ động chỉ đổi khi đồng hồ nhảy giây → bỏ qua vẽ lại (tiết kiệm ~0.5 ms/khung)
    const clock = this.clock
    const clockKey = `${w}x${h}|${def?.id ?? '-'}|${big}|${clock}`
    if (!def?.overlay && c.dataset.key === clockKey) return
    c.dataset.key = clockKey
    const g = c.getContext('2d')!
    g.clearRect(0, 0, w, h)
    if (!def) return
    g.save()
    g.scale(dpr, dpr)
    const W = rect.width
    const fs = big ? 15 : 10
    // thanh tiêu đề kiểu CCTV
    g.fillStyle = 'rgba(8,12,16,0.62)'
    g.fillRect(0, 0, W, fs + 10)
    g.font = `600 ${fs}px Consolas, "Cascadia Mono", monospace`
    const cw = g.measureText(clock).width
    g.fillStyle = '#ff453a'
    g.beginPath()
    g.arc(W - cw - 16, fs / 2 + 5, fs * 0.32, 0, Math.PI * 2)
    g.fill()
    g.fillStyle = '#e6edf3'
    g.fillText(clock, W - cw - 6, fs + 3)
    // tên camera cắt bớt cho vừa phần còn lại
    let title = big ? `${def.id} · ${def.name}` : def.id
    const room = W - cw - 34
    if (!big && g.measureText(`${def.id} · ${def.name}`).width <= room) title = `${def.id} · ${def.name}`
    while (g.measureText(title).width > room && title.length > 4) title = title.slice(0, -2) + '…'
    g.fillText(title, 6, fs + 3)

    if (def.overlay === 'sort') this.drawSort(g, rect, cam, sim, actors, fs, big)
    if (def.overlay === 'dry') this.drawDry(g, rect, cam, sim, actors, fs, big)
    if (def.overlay === 'dryer') this.drawDryer(g, rect, sim, fs)
    g.restore()
  }

  /** Chiếu khối cầu (tâm, bán kính) lên ô → hình chữ nhật pixel. */
  private project(p: THREE.Vector3, radius: number, cam: THREE.PerspectiveCamera, rect: DOMRect) {
    const v = p.clone().project(cam)
    if (v.z > 1 || Math.abs(v.x) > 1.2 || Math.abs(v.y) > 1.2) return null
    const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0).multiplyScalar(radius)
    const e = p.clone().add(right).project(cam)
    const x = (v.x * 0.5 + 0.5) * rect.width
    const y = (-v.y * 0.5 + 0.5) * rect.height
    const rpx = Math.abs(e.x - v.x) * 0.5 * rect.width
    return { x, y, r: rpx }
  }

  private box(g: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, text: string, fs: number, dashed = false) {
    g.strokeStyle = color
    g.lineWidth = fs > 12 ? 2 : 1.3
    g.setLineDash(dashed ? [4, 3] : [])
    g.strokeRect(x - r, y - r, 2 * r, 2 * r)
    g.setLineDash([])
    g.font = `600 ${fs - 1}px Consolas, monospace`
    const tw = g.measureText(text).width + 6
    g.fillStyle = color
    g.fillRect(x - r, y - r - fs - 2, tw, fs + 2)
    g.fillStyle = '#0b0f14'
    g.fillText(text, x - r + 3, y - r - 4)
  }

  private drawSort(g: CanvasRenderingContext2D, rect: DOMRect, cam: THREE.PerspectiveCamera, sim: FactorySim, actors: Actors, fs: number, big: boolean) {
    for (const it of sim.state.items) {
      if (it.rejected) continue
      const p = actors.itemPos.get(it.id)
      if (!p) continue
      const rad = it.fruit === 'jackfruit' ? 0.3 : 0.12
      const q = this.project(p.clone().setY(p.y + rad * 0.5), rad, cam, rect)
      if (!q) continue
      const vi = big ? `${FRUITS[it.fruit].vi} ` : ''
      const conf = it.ai ? ` ${Math.round(it.ai.conf * 100)}%` : ''
      if (!it.ai) this.box(g, q.x, q.y, q.r, '#8b949e', big ? `${vi}· đang quét` : '…', fs, true)
      else if (it.ai.defect) this.box(g, q.x, q.y, q.r, '#ff6b5b', big ? `${vi}HƯ · ${it.ai.label}${conf}` : `HƯ${conf}`, fs)
      else this.box(g, q.x, q.y, q.r, '#3fd47a', `${vi}TỐT${conf}`, fs)
    }
    this.caption(g, rect, 'YOLO phát hiện → CNN tốt/hư (giả lập)', fs)
  }

  private drawDry(g: CanvasRenderingContext2D, rect: DOMRect, cam: THREE.PerspectiveCamera, sim: FactorySim, actors: Actors, fs: number, big: boolean) {
    const text = { under: 'CHƯA ĐẠT', ok: 'ĐẠT', over: 'QUÁ KHÔ' } as const
    const color = { under: '#ffb020', ok: '#3fd47a', over: '#ff6b5b' } as const
    for (const u of sim.state.units) {
      const p = actors.unitPos.get(u.id)
      if (!p) continue
      const q = this.project(p.clone().setY(p.y + 0.02), 0.14, cam, rect)
      if (!q) continue
      if (!u.ai) this.box(g, q.x, q.y, q.r, '#8b949e', big ? 'đang đo' : '…', fs, true)
      else this.box(g, q.x, q.y, q.r, color[u.ai.label], big ? `${text[u.ai.label]} ${u.ai.est.toFixed(1)}%` : `${u.ai.est.toFixed(0)}%`, fs)
    }
    this.caption(g, rect, 'Ước lượng độ ẩm từ ảnh (giả lập)', fs)
  }

  private drawDryer(g: CanvasRenderingContext2D, rect: DOMRect, sim: FactorySim, fs: number) {
    const d = sim.state.dryer
    const b = d.batchId ? sim.batch(d.batchId) : null
    const line = b ? `${d.tempC.toFixed(0)}°C · ${d.mode === 'main' ? `ẩm ${d.moisture.toFixed(1)}%` : 'sấy bổ sung'} · ${Math.round((d.t / d.duration) * 100)}%` : 'Lò trống — chờ lô'
    this.caption(g, rect, line, fs)
  }

  private caption(g: CanvasRenderingContext2D, rect: DOMRect, text: string, fs: number) {
    g.font = `${fs}px Consolas, monospace`
    // cắt bớt cho vừa ô nhỏ thay vì tràn khỏi mép
    const room = rect.width - 18
    while (g.measureText(text).width > room && text.length > 4) text = text.slice(0, -2) + '…'
    const tw = g.measureText(text).width + 10
    g.fillStyle = 'rgba(8,12,16,0.62)'
    g.fillRect(4, rect.height - fs - 10, tw, fs + 6)
    g.fillStyle = '#c9d1d9'
    g.fillText(text, 9, rect.height - 8)
  }
}
