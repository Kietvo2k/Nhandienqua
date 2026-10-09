import { describe, expect, it } from 'vitest'
import { defectMap, detHeatmap, jet } from '../src/lab/heatmap'

describe('detHeatmap — điểm lớp cao nhất mỗi anchor, gộp các lưới về lưới mịn nhất', () => {
  it('khớp vị trí anchor (hàng trước) và lấy max giữa các tầng', () => {
    // 2 lớp, lưới 2×2 + 1×1 → N = 5 (giống test Python)
    const n = 5
    const out = new Float32Array(6 * n)
    out[4 * n + 1] = 0.9 // anchor (y=0,x=1) lưới 2×2, lớp 0
    out[5 * n + 4] = 0.3 // anchor của lưới 1×1, lớp 1 → phủ cả ảnh
    const h = detHeatmap(out, 2, n, [[2, 2], [1, 1]])
    expect(h.w).toBe(2)
    expect(h.h).toBe(2)
    expect(Array.from(h.data).map((v) => +v.toFixed(3))).toEqual([0.3, 0.9, 0.3, 0.3])
  })
})

describe('defectMap — bằng chứng "hư" = TB lớp hư − lớp tốt của đúng loại quả', () => {
  const names = ['banana__defect', 'banana__good', 'mango__defect', 'mango__good']
  it('chọn đúng kênh và bỏ lớp của quả khác', () => {
    const hw = 4
    const cams = new Float32Array(4 * hw).fill(0)
    cams.set([100, 100, 100, 100], 0) // banana__defect — phải bỏ qua
    cams.set([3, 0, 0, 0], 2 * hw) // mango__defect
    cams.set([1, 1, 1, 1], 3 * hw) // mango__good
    expect(Array.from(defectMap(cams, names, 'mango', 2, 2)!)).toEqual([2, -1, -1, -1])
  })
  it('quả không có lớp tốt/hư → null', () => {
    expect(defectMap(new Float32Array(16), names, 'jackfruit', 2, 2)).toBeNull()
  })
})

describe('jet colormap', () => {
  it('0 → xanh dương, 1 → đỏ, 0.5 → xanh lá-vàng', () => {
    expect(jet(0)).toEqual([0, 0, 128])
    expect(jet(1)).toEqual([128, 0, 0])
    const [r, g, b] = jet(0.5)
    expect(g).toBeGreaterThan(200)
    expect(r).toBeGreaterThan(100)
    expect(b).toBeLessThan(150)
  })
})
