import { FRUITS, type FruitType } from './params'

/** Mã viết tắt trong mã truy xuất. */
const FRUIT_CODE: Record<FruitType, string> = {
  mango: 'XOAI',
  banana: 'CHUOI',
  dragon_fruit: 'TLONG',
  jackfruit: 'MIT',
  sweet_potato: 'KLANG',
}

const pad = (n: number, w: number) => String(n).padStart(w, '0')

/** NDQ-<QUẢ>-<yyyymmdd>-<số lô 3 chữ số> */
export function makeTraceCode(fruit: FruitType, date: Date, batchSeq: number): string {
  const ymd = `${date.getFullYear()}${pad(date.getMonth() + 1, 2)}${pad(date.getDate(), 2)}`
  return `NDQ-${FRUIT_CODE[fruit]}-${ymd}-${pad(batchSeq, 3)}`
}

/** Đơn vị sản xuất giả lập (đồ án) — hiển thị trên trang sản phẩm. */
export const PRODUCER = {
  name: 'Xưởng sấy mô phỏng Nhandienqua',
  address: 'Mô phỏng đồ án — không phải cơ sở sản xuất thật',
}

/** Thứ tự cố định các công đoạn in trên trang SP. */
export const STEP_NAMES = ['Nhận hàng', 'Phân loại AI', 'Sấy', 'Đánh giá mức sấy', 'Kiểm nghiệm', 'Đóng gói', 'Dán nhãn QR'] as const

/**
 * Thông tin trang sản phẩm. Bảy trường tối thiểu theo khoản 5 Điều 6 Thông tư 02/2024/TT-BKHCN:
 * tên SP (name), hình ảnh (theo `fruit`), tên đơn vị (producer), địa chỉ (address),
 * thời gian sản xuất (producedAt + steps), nhãn hiệu/số sê-ri (serial), hạn sử dụng (expiry).
 */
export interface ProductInfo {
  v: 1
  name: string
  fruit: FruitType
  trace: string
  serial: string
  producer: string
  address: string
  producedAt: string
  expiry: string
  massG: number
  steps: { name: string; at: string }[]
  qc: { samples: number; passed: number; moisture: number; aw: number }
}

const pad2 = (n: number) => String(n).padStart(2, '0')

/** Ngày hết hạn (yyyy-mm-dd, giờ địa phương) = ngày sản xuất + `months` tháng. */
export function addMonthsIso(d: Date, months: number): string {
  const x = new Date(d.getFullYear(), d.getMonth() + months, d.getDate())
  return `${x.getFullYear()}-${pad2(x.getMonth() + 1)}-${pad2(x.getDate())}`
}

function toBase64Url(bytes: Uint8Array): string {
  let bin = ''
  bytes.forEach((b) => (bin += String.fromCharCode(b)))
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

function fromBase64Url(s: string): Uint8Array {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/')
  const bin = atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4))
  return Uint8Array.from(bin, (c) => c.charCodeAt(0))
}

const FRUIT_LIST = Object.keys(FRUIT_CODE) as FruitType[]
const TRACE_RE = /^NDQ-[A-Z]+-\d{8}-\d{3}$/

/**
 * Mã hoá GỌN vào URL (QR thưa, dễ quét): chỉ giữ phần thay đổi theo túi.
 * [2, chỉ số quả, mã lô, số túi, t0 (giây epoch), khối lượng g, [độ lệch giây từng công đoạn | null],
 *  số mẫu, số mẫu đạt, độ ẩm×10, aw×100]
 * Tên SP, đơn vị, địa chỉ, hạn dùng được dựng lại theo quy tắc của dự án.
 */
export function encodeProduct(info: ProductInfo): string {
  const t0 = Math.round(Date.parse(info.producedAt) / 1000)
  const byName = new Map(info.steps.map((s) => [s.name, Math.round(Date.parse(s.at) / 1000) - t0]))
  const bagNo = Number(info.serial.slice(info.trace.length + 1))
  const tuple = [
    2, FRUIT_LIST.indexOf(info.fruit), info.trace, bagNo, t0, info.massG,
    STEP_NAMES.map((n) => byName.get(n) ?? null),
    info.qc.samples, info.qc.passed, Math.round(info.qc.moisture * 10), Math.round(info.qc.aw * 100),
  ]
  return toBase64Url(new TextEncoder().encode(JSON.stringify(tuple)))
}

const isInt = (x: unknown): x is number => Number.isInteger(x)

/** Giải mã + kiểm tra cấu trúc. Dữ liệu từ URL là không tin cậy → sai thì trả null. */
export function decodeProduct(s: string): ProductInfo | null {
  if (!/^[A-Za-z0-9_-]{1,600}$/.test(s)) return null
  let t: unknown
  try {
    t = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(fromBase64Url(s)))
  } catch {
    return null
  }
  if (!Array.isArray(t) || t.length !== 11 || t[0] !== 2) return null
  const [, fi, trace, bagNo, t0, massG, offs, samples, passed, m10, aw100] = t
  if (!isInt(fi) || fi < 0 || fi >= FRUIT_LIST.length) return null
  if (typeof trace !== 'string' || !TRACE_RE.test(trace)) return null
  if (![bagNo, t0, massG, samples, passed, m10, aw100].every(isInt)) return null
  if (!Array.isArray(offs) || offs.length !== STEP_NAMES.length || !offs.every((x) => x === null || isInt(x))) return null
  const fruit = FRUIT_LIST[fi]
  const p = FRUITS[fruit]
  const at = (sec: number) => new Date((t0 + sec) * 1000).toISOString()
  return {
    v: 1,
    name: p.product,
    fruit,
    trace,
    serial: `${trace}-${String(bagNo).padStart(4, '0')}`,
    producer: PRODUCER.name,
    address: PRODUCER.address,
    producedAt: at(0),
    expiry: addMonthsIso(new Date(t0 * 1000), p.shelfLifeMonths),
    massG,
    steps: STEP_NAMES.flatMap((name, i) => (offs[i] === null ? [] : [{ name, at: at(offs[i]) }])),
    qc: { samples, passed, moisture: m10 / 10, aw: aw100 / 100 },
  }
}
