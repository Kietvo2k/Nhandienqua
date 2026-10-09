/** Ghép engine mô phỏng + cảnh 3D + camera + giao diện, chạy vòng lặp khung hình. */
import * as THREE from 'three'
import { Views } from './cameras/views'
import { ClipRecorder } from './product/video'
import { Actors } from './scene/actors'
import { buildFactory } from './scene/factory'
import { CAMERAS } from './scene/layout'
import { FactorySim } from './sim/engine'
import { Dashboard } from './ui/dashboard'
import { openQr } from './ui/qr'

const SUBSTEP = 0.05

export function startFactoryApp() {
  const $ = <T extends HTMLElement>(s: string) => document.querySelector<T>(s)!
  const seed = Number(new URLSearchParams(location.search).get('seed') ?? Date.now() % 100000)
  const sim = new FactorySim({}, seed)
  const scene = new THREE.Scene()
  const handles = buildFactory(scene)
  const actors = new Actors(scene, handles)
  const stage = $('#stage')
  document.body.classList.add('factory-mode')
  // kéo chuột trái để xoay cảnh không được khởi tạo bôi chọn văn bản (kể cả khi kéo ra ngoài cột phải)
  for (const el of [stage, $('#cams'), $('#line-strip')]) el.addEventListener('selectstart', (e) => e.preventDefault())
  stage.addEventListener('pointerdown', (e) => {
    if (e.button === 0) window.getSelection()?.removeAllRanges()
  })
  const views = new Views($<HTMLCanvasElement>('#gl'), stage, $('#cams'), $<HTMLCanvasElement>('#stage-overlay'))
  const recorder = new ClipRecorder(() => views.renderer.getPixelRatio())
  views.onTile = recorder.hook

  const focusCam = (id: string | null) => views.focus(id ? CAMERAS.find((c) => c.id === id) ?? null : null)
  const qrDlg = $<HTMLDialogElement>('#qr-dlg')
  const dash = new Dashboard((id) => focusCam(views.focused?.id === id ? null : id), (b) => openQr(qrDlg, sim, b, recorder), sim)

  stage.addEventListener('focus-change', (e) => {
    const def = (e as CustomEvent).detail
    $('#stage-title').textContent = def ? `${def.id} · ${def.name}` : 'Toàn cảnh nhà xưởng'
    $('#back-overview').hidden = !def
    stage.classList.toggle('focused', !!def)
  })
  $('#back-overview').addEventListener('click', () => focusCam(null))
  window.addEventListener('keydown', (e) => e.key === 'Escape' && !document.querySelector('dialog[open]') && focusCam(null))

  let speed = 1
  for (const b of document.querySelectorAll<HTMLButtonElement>('.speed button')) {
    b.addEventListener('click', () => {
      speed = Number(b.dataset.speed)
      document.querySelectorAll('.speed button').forEach((x) => {
        x.classList.toggle('on', x === b)
        x.setAttribute('aria-pressed', String(x === b))
      })
    })
  }

  let last = performance.now()
  let uiTimer = 0
  let frames = 0
  let fpsTimer = 0
  const loop = (now: number) => {
    const dtReal = Math.min(0.1, (now - last) / 1000)
    last = now
    const dtSim = dtReal * speed
    const n = Math.min(1200, Math.ceil(dtSim / SUBSTEP))
    for (let i = 0; i < n; i++) sim.step(dtSim / n)

    // ghi video lô khi lô bắt đầu vào máy đóng gói
    for (const b of sim.state.batches) if (b.steps.pack !== undefined) recorder.request(b.traceCode, 'CAM-07', 6)
    recorder.tick(now)

    actors.update(sim, dtSim, dtReal)
    views.render(scene, sim, actors)

    uiTimer -= dtReal
    if (uiTimer <= 0) {
      uiTimer = 0.2
      dash.update()
    }
    frames++
    fpsTimer += dtReal
    if (fpsTimer >= 1) {
      $('#fps').textContent = `${Math.round(frames / fpsTimer)} fps`
      frames = 0
      fpsTimer = 0
    }
    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
  // để kiểm thử/gỡ lỗi từ console
  /** Đo hiệu năng đồng bộ (không phụ thuộc rAF khi tab bị ẩn): trả ms/khung. */
  const bench = (frames = 60, simSpeed = speed) => {
    const gl = views.renderer.getContext()
    const t0 = performance.now()
    for (let i = 0; i < frames; i++) {
      const dt = (1 / 60) * simSpeed
      const n = Math.max(1, Math.ceil(dt / SUBSTEP))
      for (let k = 0; k < n; k++) sim.step(dt / n)
      actors.update(sim, dt, 1 / 60)
      views.render(scene, sim, actors)
      gl.finish()
    }
    return (performance.now() - t0) / frames
  }
  Object.assign(window, { __ndq: { sim, views, recorder, scene, actors, bench, setSpeed: (x: number) => (speed = x) } })
}
