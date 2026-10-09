/**
 * Làm mượt kết quả theo thời gian cho video/webcam ở trang thử (giống vision/nhandienqua/voting.py — EmaSmoother):
 * tracker ghép theo chồng lấn → mỗi quả một ID; EMA P(hư) + vùng trễ; loại quả theo đa số;
 * quả chạm mép khung = "đang vào" (ảnh cắt thiếu) → không quyết định, không tính vào EMA.
 */
import type { Finding } from './runtime'
import { iou, type Box } from './yolo'

export interface Tracked extends Finding {
  id: number
  partial: boolean
}

interface Track {
  box: Box
  ema: number | null
  bad: boolean | null
  votes: Map<string, number>
  age: number
}

export interface SmootherOpts {
  alpha: number
  threshold: number
  band: number
  maxAge: number
}

export class Smoother {
  private o: SmootherOpts
  private tracks = new Map<number, Track>()
  private nextId = 1

  constructor(opts: Partial<SmootherOpts> = {}) {
    this.o = { alpha: 0.3, threshold: 0.5, band: 0.1, maxAge: 15, ...opts }
  }

  reset() {
    this.tracks.clear()
  }

  /** Cùng quả nếu chồng lấn đủ, hoặc tâm gần nhau (băng chạy nhanh / FPS thấp làm IoU nhỏ). */
  private score(a: Box, b: Box): number {
    const s = iou(a, b)
    if (s > 0.1) return s
    const d = Math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    const r = 0.5 * Math.max(a[2] - a[0], a[3] - a[1])
    return d < r ? 0.1 * (1 - d / r) : 0
  }

  update(findings: Finding[], W: number, H: number): Tracked[] {
    const pairs: [number, number, number][] = []
    findings.forEach((f, i) => this.tracks.forEach((t, id) => {
      const s = this.score(t.box, f.box)
      if (s > 0) pairs.push([s, i, id])
    }))
    pairs.sort((a, b) => b[0] - a[0])
    const ofFinding = new Map<number, number>()
    const used = new Set<number>()
    for (const [, i, id] of pairs) {
      if (ofFinding.has(i) || used.has(id)) continue
      ofFinding.set(i, id)
      used.add(id)
    }
    const out = findings.map((f, i) => {
      let id = ofFinding.get(i)
      if (id === undefined) {
        id = this.nextId++
        this.tracks.set(id, { box: f.box, ema: null, bad: null, votes: new Map(), age: 0 })
      }
      const t = this.tracks.get(id)!
      t.box = f.box
      t.age = 0
      t.votes.set(f.fruit, (t.votes.get(f.fruit) ?? 0) + 1)
      const fruit = [...t.votes].sort((a, b) => b[1] - a[1])[0][0]
      const [x1, y1, x2, y2] = f.box
      const partial = x1 <= 1 || y1 <= 1 || x2 >= W - 1 || y2 >= H - 1
      if (partial) return { ...f, fruit, id, partial }
      const p = t.ema === null ? f.pDefect : this.o.alpha * f.pDefect + (1 - this.o.alpha) * t.ema
      t.ema = p
      const th = this.o.threshold
      const bad = t.bad === null ? p >= th : t.bad ? p >= th - this.o.band : p >= th + this.o.band
      t.bad = bad
      return { ...f, fruit, id, partial, pDefect: p, quality: bad ? 'defect' : 'good', action: bad ? 'reject' : 'accept' } as Tracked
    })
    for (const [id, t] of this.tracks) {
      if (used.has(id) || out.some((o) => o.id === id)) continue
      if (++t.age > this.o.maxAge) this.tracks.delete(id)
    }
    return out
  }
}
