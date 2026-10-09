/**
 * Chạy model ONNX (xuất từ vision/) ngay trong trình duyệt bằng onnxruntime-web: WebGPU nếu có, không thì WASM.
 * Pipeline giống Python: phát hiện → nới hộp 10% → phân loại tốt/hư (chỉ xét lớp của loại quả đó).
 */
import * as ort from 'onnxruntime-web/webgpu'
import { defectMap, detHeatmap, type Heat } from './heatmap'
import { type Box, classifyCrop, decideQuality, decodeYolo, type Letterbox, letterbox, nms, padBox } from './yolo'

if (!globalThis.crossOriginIsolated) ort.env.wasm.numThreads = 1

export interface ModelInfo {
  file: string
  task: 'detect' | 'classify'
  names: string[]
  imgsz: [number, number]
  bytes: number
  /** có đầu ra CAM (h, w) — bản đồ nhiệt hư hỏng */
  cam?: [number, number]
}
export interface Manifest {
  detect: ModelInfo | null
  quality: ModelInfo | null
  drying: ModelInfo | null
}

export interface Finding {
  fruit: string
  box: Box
  detConf: number
  quality: 'good' | 'defect'
  pDefect: number
  action: 'accept' | 'reject'
  /** bằng chứng "hư" theo vị trí (CAM) + vùng ảnh gốc tương ứng đầu vào model tốt/hư */
  cam?: { map: Float32Array; h: number; w: number; region: [number, number, number, number] }
}
export interface InspectResult {
  findings: Finding[]
  /** bản đồ nhiệt phát hiện trên khung letterbox (stride 8) */
  detHeat: { heat: Heat; lb: Letterbox; stride: number }
  ms: { pre: number; detect: number; quality: number; total: number }
}

interface Loaded {
  info: ModelInfo
  session: ort.InferenceSession
}

const now = () => performance.now()

export class FruitLab {
  provider = 'wasm'
  readonly manifest: Manifest
  private det: Loaded | null = null
  private qual: Loaded | null = null
  private canvas = document.createElement('canvas')
  private ctx = this.canvas.getContext('2d', { willReadFrequently: true })!

  private constructor(manifest: Manifest) {
    this.manifest = manifest
  }

  static async load(onStatus: (s: string) => void): Promise<FruitLab> {
    const manifest = (await (await fetch('/models/models.json')).json()) as Manifest
    const lab = new FruitLab(manifest)
    const providers: string[] = 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm']
    const open = async (info: ModelInfo | null) => {
      if (!info) return null
      onStatus(`Đang nạp ${info.file}…`)
      for (const p of providers) {
        try {
          const session = await ort.InferenceSession.create(`/${info.file}`, { executionProviders: [p] })
          lab.provider = p
          return { info, session }
        } catch (e) {
          console.warn(`Không mở được ${info.file} với ${p}`, e)
        }
      }
      throw new Error(`Không nạp được ${info.file}`)
    }
    lab.det = await open(manifest.detect)
    lab.qual = await open(manifest.quality)
    onStatus('Sẵn sàng')
    return lab
  }

  /** Ảnh → tensor (1,3,h,w) RGB/255; `draw` vẽ nội dung vào canvas kích thước w×h. */
  private toTensor(w: number, h: number, draw: (g: CanvasRenderingContext2D) => void) {
    this.canvas.width = w
    this.canvas.height = h
    draw(this.ctx)
    const px = this.ctx.getImageData(0, 0, w, h).data
    const t = new Float32Array(3 * w * h)
    for (let i = 0, j = 0; i < w * h; i++, j += 4) {
      t[i] = px[j] / 255
      t[w * h + i] = px[j + 1] / 255
      t[2 * w * h + i] = px[j + 2] / 255
    }
    return new ort.Tensor('float32', t, [1, 3, h, w])
  }

  async inspect(src: CanvasImageSource, W: number, H: number, detConf = 0.35, defectTh = 0.5): Promise<InspectResult> {
    if (!this.det || !this.qual) throw new Error('Chưa nạp đủ model phát hiện + tốt/hư')
    const t0 = now()
    const [, size] = this.det.info.imgsz
    const lb = letterbox(W, H, size)
    const x = this.toTensor(size, size, (g) => {
      g.fillStyle = 'rgb(114,114,114)'
      g.fillRect(0, 0, size, size)
      g.imageSmoothingQuality = 'high'
      g.drawImage(src, lb.padX, lb.padY, lb.w, lb.h)
    })
    const t1 = now()
    const out = await this.det.session.run({ [this.det.session.inputNames[0]]: x })
    const y = out[this.det.session.outputNames[0]]
    const [, ch, n] = y.dims as number[]
    const raw = y.data as Float32Array
    const dets = nms(decodeYolo(raw, ch - 4, n, detConf, lb))
    const grids = [8, 16, 32].map((st) => [size / st, size / st] as [number, number])
    const detHeat = { heat: detHeatmap(raw, ch - 4, n, grids), lb, stride: 8 }
    const t2 = now()

    const findings: Finding[] = []
    const qs = this.qual.info.imgsz[0]
    for (const d of dets) {
      const box = d.box.map((v, i) => Math.min(i % 2 ? H : W, Math.max(0, v))) as Box
      const [x1, y1, x2, y2] = padBox(box, W, H, 0.1)
      const fruit = this.det.info.names[d.cls]
      let quality: 'good' | 'defect' = 'defect'
      let pDefect = 1
      let cam: Finding['cam']
      if (x2 - x1 >= 2 && y2 - y1 >= 2) {
        const c = classifyCrop(x2 - x1, y2 - y1, qs)
        const region: [number, number, number, number] = [x1 + c.sx / c.scale, y1 + c.sy / c.scale, qs / c.scale, qs / c.scale]
        const xt = this.toTensor(qs, qs, (g) => g.drawImage(src, ...region, 0, 0, qs, qs))
        const o = await this.qual.session.run({ [this.qual.session.inputNames[0]]: xt })
        const names = this.qual.session.outputNames
        const r = decideQuality(o[names[0]].data as Float32Array, this.qual.info.names, fruit, defectTh)
        quality = r.label
        pDefect = r.pDefect
        const ci = this.qual.info.cam
        if (ci && names[1]) {
          const map = defectMap(o[names[1]].data as Float32Array, this.qual.info.names, fruit, ci[0], ci[1])
          if (map) cam = { map, h: ci[0], w: ci[1], region }
        }
      }
      findings.push({ fruit, box, detConf: d.conf, quality, pDefect, action: quality === 'defect' ? 'reject' : 'accept', cam })
    }
    const t3 = now()
    return { findings, detHeat, ms: { pre: t1 - t0, detect: t2 - t1, quality: t3 - t2, total: t3 - t0 } }
  }
}
