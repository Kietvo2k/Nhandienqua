/**
 * Ghi clip "video sản phẩm" của từng lô từ ô camera máy đóng gói (canvas.captureStream + MediaRecorder).
 * Clip giữ trong bộ nhớ (blob URL) và gửi lên dev server → web/public/videos/<mã lô>.webm
 * để điện thoại quét QR cũng xem được.
 */
import type { FrameHook } from '../cameras/views'

const W = 640
const H = 360

interface Active {
  trace: string
  cam: string
  until: number
  rec: MediaRecorder
  chunks: Blob[]
}

export class ClipRecorder {
  readonly clips = new Map<string, string>()
  private queue: { trace: string; cam: string; seconds: number }[] = []
  private active: Active | null = null
  private canvas = document.createElement('canvas')
  private ctx = this.canvas.getContext('2d')!
  private started = new Set<string>()

  private pixelRatio: () => number

  constructor(pixelRatio: () => number) {
    this.pixelRatio = pixelRatio
    this.canvas.width = W
    this.canvas.height = H
  }

  get supported() {
    return typeof MediaRecorder !== 'undefined' && 'captureStream' in this.canvas
  }

  request(trace: string, cam: string, seconds = 6) {
    if (!this.supported || this.started.has(trace)) return
    this.started.add(trace)
    this.queue.push({ trace, cam, seconds })
  }

  /** Gọi mỗi khung hình (thời gian thực). */
  tick(now: number) {
    if (!this.active && this.queue.length) this.begin(this.queue.shift()!, now)
    if (this.active && now >= this.active.until) {
      this.active.rec.stop()
      this.active = null
    }
  }

  readonly hook: FrameHook = (def, src, r) => {
    const a = this.active
    if (!a || def.id !== a.cam) return
    const pr = this.pixelRatio()
    this.ctx.drawImage(src, r.x * pr, r.y * pr, r.w * pr, r.h * pr, 0, 0, W, H)
    this.ctx.fillStyle = 'rgba(8,12,16,0.65)'
    this.ctx.fillRect(0, H - 30, W, 30)
    this.ctx.fillStyle = '#e6edf3'
    this.ctx.font = '600 16px Consolas, monospace'
    this.ctx.fillText(`${a.trace} · ${def.id} ${def.name}`, 10, H - 10)
  }

  private begin(job: { trace: string; cam: string; seconds: number }, now: number) {
    const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m))
    const stream = this.canvas.captureStream(30)
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 1_500_000 } : undefined)
    const a: Active = { ...job, until: now + job.seconds * 1000, rec, chunks: [] }
    rec.ondataavailable = (e) => e.data.size && a.chunks.push(e.data)
    rec.onstop = () => {
      const blob = new Blob(a.chunks, { type: 'video/webm' })
      this.clips.set(a.trace, URL.createObjectURL(blob))
      fetch(`/api/video/${encodeURIComponent(a.trace)}`, { method: 'POST', body: blob }).catch(() => {})
      document.dispatchEvent(new CustomEvent('clip-ready', { detail: a.trace }))
    }
    rec.start(500)
    this.active = a
  }
}
