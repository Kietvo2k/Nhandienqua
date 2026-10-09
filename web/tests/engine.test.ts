import { describe, expect, it } from 'vitest'
import { FactorySim } from '../src/sim/engine'
import { DEFAULT_LINE, FRUITS } from '../src/sim/params'

/** Chạy mô phỏng tới khi `done()` đúng hoặc hết thời gian. */
function runUntil(sim: FactorySim, done: () => boolean, maxS = 20000, dt = 0.1) {
  for (let t = 0; t < maxS && !done(); t += dt) sim.step(dt)
}

describe('FactorySim — một xe xoài', () => {
  const sim = new FactorySim({ ...DEFAULT_LINE, truckLoad: 2000, schedule: ['mango'], maxTrucks: 1 }, 7)
  runUntil(sim, () => sim.state.batches[0]?.status === 'done')
  const b = sim.state.batches[0]

  it('lô hoàn tất', () => {
    expect(b.status).toBe('done')
  })

  it('mọi quả đều được kiểm tra: nhận + loại = tổng', () => {
    expect(b.inspected).toBe(2000)
    expect(b.accepted + b.rejected).toBe(2000)
  })

  it('tỉ lệ loại khớp tham số AI giả lập (±3%)', () => {
    const p = FRUITS.mango.defectRate
    const expected = p * DEFAULT_LINE.aiRecall + (1 - p) * DEFAULT_LINE.aiFalseReject
    expect(b.rejected / 2000).toBeGreaterThan(expected - 0.03)
    expect(b.rejected / 2000).toBeLessThan(expected + 0.03)
  })

  it('ma trận nhầm lẫn cộng đủ', () => {
    const c = sim.state.kpi.sortConfusion
    expect(c.tp + c.fp + c.tn + c.fn).toBe(2000)
  })

  it('cân bằng khối lượng: khối lượng sấy = khối lượng nhận × hiệu suất', () => {
    expect(b.acceptedMassKg).toBeCloseTo(b.accepted * FRUITS.mango.massKg, 6)
    expect(b.driedMassKg).toBeCloseTo(b.acceptedMassKg * FRUITS.mango.yield, 6)
  })

  it('độ ẩm trong lò giảm dần về mục tiêu', () => {
    const curve = b.moistureCurve
    expect(curve.length).toBeGreaterThan(5)
    for (let i = 1; i < curve.length; i++) expect(curve[i].m).toBeLessThanOrEqual(curve[i - 1].m)
    expect(curve[0].m).toBeCloseTo(FRUITS.mango.moistureIn, 0)
    expect(curve[curve.length - 1].m).toBeLessThan(FRUITS.mango.moistureTarget + 1)
  })

  it('mỗi đơn vị sau sấy hoặc thành túi, hoặc mẫu QC, hoặc bị loại vì quá khô', () => {
    expect(b.units).toBeGreaterThan(0)
    expect(b.bags + b.qcTaken + b.overDried).toBe(b.units)
  })

  it('lấy mẫu QC theo ISO 2859-1 trên cỡ lô', () => {
    expect(b.qcTaken).toBe(b.qcTarget)
    expect(b.qcResults.length).toBe(b.qcTaken)
  })

  it('túi có số sê-ri gắn mã truy xuất của lô và đã dán nhãn', () => {
    const bags = sim.state.bags.filter((x) => x.batchId === b.id)
    expect(bags.length).toBe(b.bags)
    bags.forEach((bag) => expect(bag.serial.startsWith(b.traceCode)).toBe(true))
    expect(new Set(bags.map((x) => x.serial)).size).toBe(bags.length)
    expect(bags.every((x) => x.labeled)).toBe(true)
  })

  it('băng chuyền trống khi xong', () => {
    expect(sim.state.items.length).toBe(0)
    expect(sim.state.units.length).toBe(0)
  })
})

describe('FactorySim — nhiều xe, nhiều loại quả', () => {
  it('xử lý tuần tự các xe theo lịch, giữ tất định theo seed', () => {
    const cfg = { ...DEFAULT_LINE, truckLoad: 60, schedule: ['banana', 'jackfruit'] as const, maxTrucks: 2 }
    const a = new FactorySim({ ...cfg, schedule: [...cfg.schedule] }, 3)
    const b = new FactorySim({ ...cfg, schedule: [...cfg.schedule] }, 3)
    runUntil(a, () => a.state.batches.length === 2 && a.state.batches.every((x) => x.status === 'done'))
    runUntil(b, () => b.state.batches.length === 2 && b.state.batches.every((x) => x.status === 'done'))
    expect(a.state.batches.map((x) => x.fruit)).toEqual(['banana', 'jackfruit'])
    expect(a.state.batches.every((x) => x.status === 'done')).toBe(true)
    expect(a.state.batches.map((x) => x.rejected)).toEqual(b.state.batches.map((x) => x.rejected))
  })

  it('phát hiệu ứng chuyển tiếp cho cảnh 3D (qc, pack, …) và có thể rút hết', () => {
    const sim = new FactorySim({ ...DEFAULT_LINE, truckLoad: 40, schedule: ['mango'], maxTrucks: 1 }, 2)
    const seen = new Set<string>()
    runUntil(sim, () => {
      sim.drainFx().forEach((f) => seen.add(f.kind))
      return sim.state.batches[0]?.status === 'done'
    })
    expect(seen.has('qc')).toBe(true)
    expect(seen.has('pack')).toBe(true)
    expect(sim.drainFx()).toEqual([])
  })

  it('ghi log sự kiện theo thứ tự công đoạn', () => {
    const sim = new FactorySim({ ...DEFAULT_LINE, truckLoad: 30, schedule: ['sweet_potato'], maxTrucks: 1 }, 5)
    runUntil(sim, () => sim.state.batches[0]?.status === 'done')
    const kinds = sim.state.events.map((e) => e.kind)
    const order = ['truck_arrived', 'dryer_start', 'dryer_done', 'qc_sample', 'bag_labeled']
    const idx = order.map((k) => kinds.indexOf(k))
    idx.forEach((i) => expect(i).toBeGreaterThanOrEqual(0))
    expect([...idx].sort((x, y) => x - y)).toEqual(idx)
  })
})
