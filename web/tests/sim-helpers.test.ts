import { describe, expect, it } from 'vitest'
import { mulberry32, normal } from '../src/sim/rng'
import { sampleSize } from '../src/sim/qc'
import { PRODUCER, addMonthsIso, decodeProduct, encodeProduct, makeTraceCode, type ProductInfo } from '../src/sim/trace'

describe('rng', () => {
  it('cùng seed cho cùng chuỗi số', () => {
    const a = mulberry32(42)
    const b = mulberry32(42)
    const xs = Array.from({ length: 5 }, () => a())
    expect(Array.from({ length: 5 }, () => b())).toEqual(xs)
    xs.forEach((x) => expect(x).toBeGreaterThanOrEqual(0))
    xs.forEach((x) => expect(x).toBeLessThan(1))
  })

  it('normal có trung bình xấp xỉ mean', () => {
    const r = mulberry32(1)
    let sum = 0
    for (let i = 0; i < 5000; i++) sum += normal(r, 10, 2)
    expect(sum / 5000).toBeCloseTo(10, 1)
  })
})

describe('sampleSize (ISO 2859-1, mức kiểm tra chung II)', () => {
  it.each([
    [1, 1],
    [2, 2],
    [8, 2],
    [9, 3],
    [25, 5],
    [26, 8],
    [90, 13],
    [150, 20],
    [500, 50],
    [1200, 80],
    [3200, 125],
  ])('lô %i → cỡ mẫu %i', (lot, n) => {
    expect(sampleSize(lot)).toBe(n)
  })

  it('không lấy nhiều hơn cỡ lô', () => {
    expect(sampleSize(1)).toBeLessThanOrEqual(1)
    expect(sampleSize(0)).toBe(0)
  })
})

describe('trace', () => {
  it('mã truy xuất có loại quả, ngày, số lô', () => {
    expect(makeTraceCode('mango', new Date(2026, 9, 8), 7)).toBe('NDQ-XOAI-20261008-007')
    expect(makeTraceCode('dragon_fruit', new Date(2026, 0, 2), 12)).toBe('NDQ-TLONG-20260102-012')
  })

  const info: ProductInfo = {
    v: 1,
    name: 'Xoài sấy dẻo',
    fruit: 'mango',
    trace: 'NDQ-XOAI-20261008-007',
    serial: 'NDQ-XOAI-20261008-007-0003',
    producer: PRODUCER.name,
    address: PRODUCER.address,
    producedAt: '2026-10-08T09:00:00.000Z',
    expiry: addMonthsIso(new Date('2026-10-08T09:00:00.000Z'), 6),
    massG: 200,
    steps: [
      { name: 'Nhận hàng', at: '2026-10-08T08:00:00.000Z' },
      { name: 'Phân loại AI', at: '2026-10-08T08:00:11.000Z' },
      { name: 'Sấy', at: '2026-10-08T08:02:00.000Z' },
      { name: 'Đóng gói', at: '2026-10-08T09:00:00.000Z' },
    ],
    qc: { samples: 5, passed: 5, moisture: 17.2, aw: 0.6 },
  }

  it('encode/decode khứ hồi', () => {
    const s = encodeProduct(info)
    expect(s).toMatch(/^[A-Za-z0-9_-]+$/) // an toàn cho URL
    expect(decodeProduct(s)).toEqual(info)
  })

  it('chuỗi mã hoá gọn để QR thưa, dễ quét (< 200 ký tự)', () => {
    expect(encodeProduct(info).length).toBeLessThan(200)
  })

  it('decode từ chối dữ liệu hỏng hoặc sai cấu trúc', () => {
    const b64 = (x: unknown) => btoa(JSON.stringify(x)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
    expect(decodeProduct('khong-phai-base64!!')).toBeNull()
    expect(decodeProduct(b64({ v: 1 }))).toBeNull()
    expect(decodeProduct(b64([2, 'chuoi-la', 'X', 1, 0, 200, [], 1, 1, 170, 60]))).toBeNull() // loại quả không hợp lệ
    expect(decodeProduct(b64([3, 0, 'NDQ-XOAI-20261008-007', 3, 0, 200, [], 1, 1, 170, 60]))).toBeNull() // sai phiên bản
    expect(decodeProduct(b64([2, 0, '<script>', 3, 0, 200, [], 1, 1, 170, 60]))).toBeNull() // mã lô sai định dạng
  })
})
