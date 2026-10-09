import { describe, expect, it } from 'vitest'
import { classifyCrop, decideQuality, decodeYolo, iou, letterbox, nms, padBox } from '../src/lab/yolo'

describe('letterbox (giống Ultralytics: giữ tỉ lệ, pad giữa)', () => {
  it('ảnh ngang 1280×720 → 640×360, pad trên/dưới 140', () => {
    expect(letterbox(1280, 720, 640)).toEqual({ scale: 0.5, padX: 0, padY: 140, w: 640, h: 360 })
  })
  it('ảnh dọc 300×600 → 320×640, pad trái/phải 160', () => {
    expect(letterbox(300, 600, 640)).toEqual({ scale: 640 / 600, padX: 160, padY: 0, w: 320, h: 640 })
  })
})

describe('decodeYolo — đầu ra (1, 4+nc, N) dạng channel-major', () => {
  // 2 lớp, 3 anchor: [cx, cy, w, h, s0, s1] × N
  const N = 3
  const rows = [
    [100, 300, 500], // cx
    [100, 300, 500], // cy
    [40, 40, 40], // w
    [20, 20, 20], // h
    [0.9, 0.1, 0.2], // điểm lớp 0
    [0.05, 0.8, 0.1], // điểm lớp 1
  ]
  const out = Float32Array.from(rows.flat())
  const lb = { scale: 0.5, padX: 0, padY: 140, w: 640, h: 360 }

  it('lọc theo ngưỡng, chọn lớp cao nhất, đổi về toạ độ ảnh gốc', () => {
    const d = decodeYolo(out, 2, N, 0.5, lb)
    expect(d).toHaveLength(2)
    expect(d[0]).toMatchObject({ cls: 0, conf: expect.closeTo(0.9, 5) })
    // cx=100,cy=100,w=40,h=20 → x1=80,y1=90 trên khung 640; bỏ pad 140 ở y, chia scale 0.5
    expect(d[0].box).toEqual([160, -100, 240, -60].map((v) => expect.closeTo(v, 5)))
    expect(d[1]).toMatchObject({ cls: 1, conf: expect.closeTo(0.8, 5) })
  })
})

describe('nms theo lớp', () => {
  it('iou', () => {
    expect(iou([0, 0, 10, 10], [5, 0, 15, 10])).toBeCloseTo(50 / 150)
    expect(iou([0, 0, 10, 10], [20, 20, 30, 30])).toBe(0)
  })
  it('bỏ hộp trùng cùng lớp, giữ hộp trùng khác lớp', () => {
    const dets = [
      { box: [0, 0, 10, 10] as [number, number, number, number], cls: 0, conf: 0.9 },
      { box: [1, 0, 11, 10] as [number, number, number, number], cls: 0, conf: 0.8 },
      { box: [1, 0, 11, 10] as [number, number, number, number], cls: 1, conf: 0.7 },
      { box: [50, 50, 60, 60] as [number, number, number, number], cls: 0, conf: 0.6 },
    ]
    const keep = nms(dets, 0.7)
    expect(keep.map((d) => d.conf)).toEqual([0.9, 0.7, 0.6])
  })
})

describe('padBox / classifyCrop', () => {
  it('nới 10% và kẹp trong ảnh (giống crop_with_pad của Python)', () => {
    expect(padBox([190, 90, 210, 120], 200, 100, 0.1)).toEqual([188, 87, 200, 100])
  })
  it('resize cạnh ngắn rồi cắt giữa (Ultralytics classify_transforms, crop_fraction=1)', () => {
    // crop 100×50 → cạnh ngắn 50 → 224, cạnh dài 448 → cắt giữa từ 112
    expect(classifyCrop(100, 50, 224)).toEqual({ scale: 224 / 50, sx: 112, sy: 0 })
  })
})

describe('decideQuality — chỉ xét lớp của loại quả mà detector báo', () => {
  const names = ['banana__defect', 'banana__good', 'mango__defect', 'mango__good']
  it('chuẩn hoá trong nhóm lớp của quả', () => {
    const r = decideQuality(Float32Array.from([0.7, 0.1, 0.05, 0.15]), names, 'mango', 0.5)
    expect(r.label).toBe('good')
    expect(r.pDefect).toBeCloseTo(0.25)
  })
  it('quả chưa có lớp chất lượng → hư (an toàn)', () => {
    expect(decideQuality(Float32Array.from([0.5, 0.5, 0, 0]), names, 'jackfruit', 0.5)).toEqual({ label: 'defect', pDefect: 1 })
  })
})
