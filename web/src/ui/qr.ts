import QRCode from 'qrcode'
import type { Batch, FactorySim } from '../sim/engine'
import { buildProductInfo } from '../sim/product'
import { encodeProduct } from '../sim/trace'
import type { ClipRecorder } from '../product/video'
import { pickLanUrl } from './lan'

let lanBase: string | null | undefined

/** Địa chỉ gốc để in vào QR: ưu tiên IP LAN do dev server báo (điện thoại mở được). */
async function baseUrl(): Promise<{ base: string; lan: boolean }> {
  if (lanBase === undefined) {
    try {
      const r = await fetch('/api/lan')
      const j = (await r.json()) as { network?: string[] }
      lanBase = pickLanUrl(j.network ?? [])
    } catch {
      lanBase = null
    }
  }
  return lanBase ? { base: lanBase, lan: true } : { base: `${location.origin}/`, lan: false }
}

export async function productUrl(sim: FactorySim, b: Batch): Promise<{ url: string; lan: boolean; serial: string } | null> {
  const bag = [...sim.state.bags].reverse().find((x) => x.batchId === b.id && x.labeled)
  if (!bag) return null
  const { base, lan } = await baseUrl()
  return { url: `${base}#/p/${encodeProduct(buildProductInfo(sim, b, bag))}`, lan, serial: bag.serial }
}

export async function openQr(dlg: HTMLDialogElement, sim: FactorySim, b: Batch, rec: ClipRecorder) {
  const res = await productUrl(sim, b)
  if (!res) return
  const img = dlg.querySelector<HTMLImageElement>('#qr-img')!
  img.src = await QRCode.toDataURL(res.url, { errorCorrectionLevel: 'M', margin: 1, width: 440 })
  dlg.querySelector('#qr-serial')!.textContent = `Túi ${res.serial}`
  dlg.querySelector('#qr-note')!.textContent = res.lan
    ? 'Quét bằng điện thoại cùng mạng Wi-Fi để mở trang sản phẩm.'
    : 'Đang chạy ở localhost — điện thoại không mở được. Chạy "npm run dev" (đã bật --host) và mở bằng địa chỉ mạng LAN.'
  const link = dlg.querySelector<HTMLAnchorElement>('#qr-link')!
  link.href = res.url
  const v = dlg.querySelector<HTMLAnchorElement>('#qr-video')!
  const clip = rec.clips.get(b.traceCode)
  v.hidden = !clip
  if (clip) {
    v.href = clip
    v.download = `${b.traceCode}.webm`
  }
  dlg.showModal()
}
