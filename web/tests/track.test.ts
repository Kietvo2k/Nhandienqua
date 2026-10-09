import { describe, expect, it } from 'vitest'
import { Smoother } from '../src/lab/track'
import type { Finding } from '../src/lab/runtime'

const f = (box: [number, number, number, number], p: number, fruit = 'mango'): Finding => ({
  fruit, box, detConf: 0.9, quality: p >= 0.5 ? 'defect' : 'good', pDefect: p, action: p >= 0.5 ? 'reject' : 'accept',
})

describe('Smoother (tracker IoU + EMA + vùng trễ, giống vision/voting.py)', () => {
  it('giữ ID theo chồng lấn và làm mượt quả lưng chừng', () => {
    const s = new Smoother({ alpha: 0.3, band: 0.1 })
    const raw = [0.6, 0.45, 0.55, 0.48, 0.58, 0.46]
    const out = raw.map((p, i) => s.update([f([100 + 5 * i, 100, 160 + 5 * i, 160], p)], 640, 480)[0])
    expect(new Set(out.map((o) => o.id)).size).toBe(1)
    const acts = out.map((o) => o.action)
    expect(acts.every((a) => a === acts[0])).toBe(true)
  })

  it('quả mới tách ID riêng; quả cũ biến mất lâu thì bị xoá', () => {
    const s = new Smoother({ maxAge: 2 })
    const a = s.update([f([0, 0, 50, 50], 0.1)], 640, 480)[0].id
    const b = s.update([f([0, 0, 50, 50], 0.1), f([300, 300, 360, 360], 0.9)], 640, 480)[1].id
    expect(b).not.toBe(a)
    for (let i = 0; i < 4; i++) s.update([], 640, 480)
    expect(s.update([f([0, 0, 50, 50], 0.1)], 640, 480)[0].id).not.toBe(a)
  })

  it('quả chạm mép khung: đánh dấu "đang vào", không quyết định, không làm lệch EMA', () => {
    const s = new Smoother({ alpha: 0.5 })
    const e = s.update([f([0, 100, 40, 160], 0.95)], 640, 480)[0]
    expect(e.partial).toBe(true)
    const g = s.update([f([30, 100, 90, 160], 0.1)], 640, 480)[0]
    expect(g.partial).toBe(false)
    expect(g.pDefect).toBeCloseTo(0.1) // khung chạm mép không được tính vào EMA
    expect(g.action).toBe('accept')
  })

  it('giữ loại quả theo đa số khi detector lỡ đổi nhãn một khung', () => {
    const s = new Smoother()
    s.update([f([100, 100, 160, 160], 0.1, 'mango')], 640, 480)
    s.update([f([102, 100, 162, 160], 0.1, 'mango')], 640, 480)
    const o = s.update([f([104, 100, 164, 160], 0.1, 'banana')], 640, 480)[0]
    expect(o.fruit).toBe('mango')
  })
})
