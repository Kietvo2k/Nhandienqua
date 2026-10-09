/**
 * Bản đồ nhiệt (giống vision/nhandienqua/explain.py):
 * - phát hiện: điểm lớp cao nhất của mỗi anchor trên các lưới stride 8/16/32, gộp về lưới mịn nhất;
 * - hư hỏng: CAM của model tốt/hư = TB các lớp hư − lớp tốt của đúng loại quả (dương = nghiêng về hư).
 */
export interface Heat {
  data: Float32Array
  w: number
  h: number
}

/** Lấy mẫu song tuyến (căn tâm ô như cv2.INTER_LINEAR). */
function sample(src: Float32Array, sw: number, sh: number, x: number, y: number): number {
  const fx = Math.min(Math.max(x, 0), sw - 1)
  const fy = Math.min(Math.max(y, 0), sh - 1)
  const x0 = Math.floor(fx)
  const y0 = Math.floor(fy)
  const x1 = Math.min(x0 + 1, sw - 1)
  const y1 = Math.min(y0 + 1, sh - 1)
  const tx = fx - x0
  const ty = fy - y0
  const a = src[y0 * sw + x0] * (1 - tx) + src[y0 * sw + x1] * tx
  const b = src[y1 * sw + x0] * (1 - tx) + src[y1 * sw + x1] * tx
  return a * (1 - ty) + b * ty
}

export function detHeatmap(out: Float32Array, nc: number, n: number, grids: [number, number][]): Heat {
  const [h0, w0] = grids[0]
  const data = new Float32Array(h0 * w0)
  let k = 0
  for (const [gh, gw] of grids) {
    const g = new Float32Array(gh * gw)
    for (let i = 0; i < gh * gw; i++) {
      let m = 0
      for (let c = 0; c < nc; c++) m = Math.max(m, out[(4 + c) * n + k + i])
      g[i] = m
    }
    k += gh * gw
    for (let y = 0; y < h0; y++) {
      for (let x = 0; x < w0; x++) {
        const v = gh === h0 && gw === w0 ? g[y * gw + x] : sample(g, gw, gh, ((x + 0.5) * gw) / w0 - 0.5, ((y + 0.5) * gh) / h0 - 0.5)
        data[y * w0 + x] = Math.max(data[y * w0 + x], v)
      }
    }
  }
  return { data, w: w0, h: h0 }
}

export function defectMap(cams: Float32Array, names: string[], fruit: string, h: number, w: number): Float32Array | null {
  const bad = names.flatMap((n, i) => (n.startsWith(`${fruit}__`) && !n.endsWith('__good') ? [i] : []))
  const good = names.indexOf(`${fruit}__good`)
  if (!bad.length || good < 0) return null
  const hw = h * w
  const m = new Float32Array(hw)
  for (let p = 0; p < hw; p++) {
    let s = 0
    for (const i of bad) s += cams[i * hw + p]
    m[p] = s / bad.length - cams[good * hw + p]
  }
  return m
}

/** Bảng màu JET (0 xanh dương → 1 đỏ). */
export function jet(v: number): [number, number, number] {
  const c = (x: number) => Math.round(255 * Math.min(1, Math.max(0, x)))
  return [c(1.5 - Math.abs(4 * v - 3)), c(1.5 - Math.abs(4 * v - 2)), c(1.5 - Math.abs(4 * v - 1))]
}

/** Vẽ bản đồ (giá trị đã chuẩn hoá 0..1) thành canvas nhỏ; trình duyệt nội suy khi phóng lên. */
export function heatCanvas(values: Float32Array, w: number, h: number, alpha: (v: number) => number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const img = c.getContext('2d')!.createImageData(w, h)
  for (let i = 0; i < w * h; i++) {
    const v = values[i]
    const [r, g, b] = jet(v)
    img.data.set([r, g, b, Math.round(255 * alpha(v))], i * 4)
  }
  c.getContext('2d')!.putImageData(img, 0, 0)
  return c
}
