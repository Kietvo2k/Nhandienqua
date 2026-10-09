/**
 * Tiền/hậu xử lý cho model ONNX xuất từ Ultralytics — logic thuần, test bằng vitest.
 * Khớp với Python: letterbox (Ultralytics), crop_with_pad + decide_quality (vision/nhandienqua/pipeline.py).
 */
export type Box = [number, number, number, number] // x1, y1, x2, y2 (pixel ảnh gốc)

export interface Letterbox {
  scale: number
  padX: number
  padY: number
  w: number
  h: number
}

export interface Det {
  box: Box
  cls: number
  conf: number
}

/** Giữ tỉ lệ, thu ảnh vào khung `size`×`size`, pad đều hai bên (màu 114 khi vẽ). */
export function letterbox(srcW: number, srcH: number, size: number): Letterbox {
  const scale = Math.min(size / srcW, size / srcH)
  const w = Math.round(srcW * scale)
  const h = Math.round(srcH * scale)
  // làm tròn như Ultralytics: round(d - 0.1); kẹp ≥ 0 (tránh -0)
  return { scale, padX: Math.max(0, Math.round((size - w) / 2 - 0.1)), padY: Math.max(0, Math.round((size - h) / 2 - 0.1)), w, h }
}

/** Đầu ra YOLO (1, 4+nc, N) channel-major: [cx, cy, w, h, điểm lớp…] → hộp trên ảnh gốc. */
export function decodeYolo(out: Float32Array, nc: number, n: number, confTh: number, lb: Letterbox): Det[] {
  const dets: Det[] = []
  for (let i = 0; i < n; i++) {
    let best = -1
    let conf = confTh
    for (let c = 0; c < nc; c++) {
      const s = out[(4 + c) * n + i]
      if (s > conf) {
        conf = s
        best = c
      }
    }
    if (best < 0) continue
    const cx = out[i]
    const cy = out[n + i]
    const w = out[2 * n + i]
    const h = out[3 * n + i]
    const un = (v: number, pad: number) => (v - pad) / lb.scale
    dets.push({
      box: [un(cx - w / 2, lb.padX), un(cy - h / 2, lb.padY), un(cx + w / 2, lb.padX), un(cy + h / 2, lb.padY)],
      cls: best,
      conf,
    })
  }
  return dets
}

export function iou(a: Box, b: Box): number {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const inter = iw * ih
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return ua > 0 ? inter / ua : 0
}

/** NMS theo từng lớp (như Ultralytics agnostic=False), ngưỡng IoU mặc định 0.7. */
export function nms(dets: Det[], iouTh = 0.7, maxDet = 300): Det[] {
  const sorted = [...dets].sort((a, b) => b.conf - a.conf)
  const keep: Det[] = []
  for (const d of sorted) {
    if (keep.length >= maxDet) break
    if (keep.every((k) => k.cls !== d.cls || iou(k.box, d.box) <= iouTh)) keep.push(d)
  }
  return keep
}

/** Nới hộp `pad` mỗi phía rồi kẹp trong ảnh, làm tròn xuống như int() của Python. */
export function padBox(box: Box, imgW: number, imgH: number, pad = 0.1): Box {
  const [x1, y1, x2, y2] = box.map(Math.trunc)
  const px = Math.trunc((x2 - x1) * pad)
  const py = Math.trunc((y2 - y1) * pad)
  return [Math.max(0, x1 - px), Math.max(0, y1 - py), Math.min(imgW, x2 + px), Math.min(imgH, y2 + py)]
}

/** Ultralytics classify_transforms (crop_fraction=1): resize cạnh ngắn = size, cắt giữa size×size. */
export function classifyCrop(cropW: number, cropH: number, size: number) {
  const scale = size / Math.min(cropW, cropH)
  return { scale, sx: Math.round((cropW * scale - size) / 2), sy: Math.round((cropH * scale - size) / 2) }
}

/** Chỉ xét lớp `<fruit>__*`; p_defect = tổng lớp không phải `__good` / tổng nhóm. */
export function decideQuality(probs: Float32Array, names: string[], fruit: string, th: number) {
  let total = 0
  let bad = 0
  names.forEach((n, i) => {
    if (!n.startsWith(`${fruit}__`)) return
    total += probs[i]
    if (!n.endsWith('__good')) bad += probs[i]
  })
  if (total <= 0) return { label: 'defect' as const, pDefect: 1 }
  const pDefect = bad / total
  return { label: pDefect >= th ? ('defect' as const) : ('good' as const), pDefect }
}
