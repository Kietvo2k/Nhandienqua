import { describe, expect, it } from 'vitest'
import { FactorySim } from '../src/sim/engine'
import { DEFAULT_LINE } from '../src/sim/params'
import { buildProductInfo, PRODUCER } from '../src/sim/product'
import { decodeProduct, encodeProduct } from '../src/sim/trace'

describe('buildProductInfo', () => {
  const start = new Date('2026-10-08T01:00:00.000Z')
  const sim = new FactorySim({ ...DEFAULT_LINE, truckLoad: 40, schedule: ['mango'], maxTrucks: 1 }, 11, start)
  for (let t = 0; t < 20000 && sim.state.batches[0]?.status !== 'done'; t += 0.1) sim.step(0.1)
  const b = sim.state.batches[0]
  const bag = sim.state.bags.find((x) => x.batchId === b.id)!

  it('đủ 7 trường bắt buộc (TT 02/2024/TT-BKHCN)', () => {
    const p = buildProductInfo(sim, b, bag)
    expect(p.name).toBe('Xoài sấy dẻo')
    expect(p.fruit).toBe('mango') // → hình ảnh SP
    expect(p.producer).toBe(PRODUCER.name)
    expect(p.address).toBe(PRODUCER.address)
    const packMs = Math.round(sim.dateAt(b.steps.pack!).getTime() / 1000) * 1000 // QR lưu theo giây
    expect(p.producedAt).toBe(new Date(packMs).toISOString())
    expect(p.serial).toBe(bag.serial)
    expect(p.expiry).toBe('2027-04-08') // 6 tháng sau ngày đóng gói
  })

  it('có các công đoạn theo thứ tự thời gian và kết quả QC', () => {
    const p = buildProductInfo(sim, b, bag)
    expect(p.steps.map((s) => s.name)).toEqual([
      'Nhận hàng', 'Phân loại AI', 'Sấy', 'Đánh giá mức sấy', 'Kiểm nghiệm', 'Đóng gói', 'Dán nhãn QR',
    ])
    const times = p.steps.map((s) => Date.parse(s.at))
    expect([...times].sort((a, c) => a - c)).toEqual(times)
    expect(p.qc.samples).toBe(b.qcTaken)
    expect(p.qc.passed).toBeLessThanOrEqual(p.qc.samples)
  })

  it('mã hoá được và đủ ngắn để in QR', () => {
    const s = encodeProduct(buildProductInfo(sim, b, bag))
    expect(decodeProduct(s)).toEqual(buildProductInfo(sim, b, bag)) // khứ hồi đầy đủ với lô thật
    expect(s.length).toBeLessThan(200)
  })
})
