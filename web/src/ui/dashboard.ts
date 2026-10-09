import type { Batch, BatchStatus, FactorySim } from '../sim/engine'
import { FRUITS } from '../sim/params'
import { CAMERAS } from '../scene/layout'

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!

const STATUS_VI: Record<BatchStatus, string> = {
  receiving: 'Đang nhận & phân loại',
  waiting_dryer: 'Chờ lò sấy',
  drying: 'Đang sấy',
  grading: 'Đánh giá · QC · đóng gói',
  done: 'Hoàn tất',
}

interface Chip {
  cam: string
  title: string
  text: (sim: FactorySim) => string
  state: (sim: FactorySim) => 'idle' | 'run' | 'busy'
}

const STATE_VI = { idle: 'Chờ', run: 'Chạy', busy: 'Bận' } as const

const pct = (a: number, b: number) => (b ? `${((100 * a) / b).toFixed(1)}%` : '0%')
const sum = (sim: FactorySim, f: (b: Batch) => number) => sim.state.batches.reduce((s, b) => s + f(b), 0)

const CHIPS: Chip[] = [
  {
    cam: 'CAM-01', title: 'Nhận hàng',
    text: (s) => { const t = s.state.trucks[0]; return t ? `Xe #${t.id} · còn ${t.remaining}` : 'Chờ xe' },
    state: (s) => (s.state.trucks[0]?.phase === 'unloading' ? 'run' : s.state.trucks.length ? 'busy' : 'idle'),
  },
  {
    cam: 'CAM-02', title: 'AI phân loại',
    text: (s) => `${sum(s, (b) => b.accepted)} nhận · ${sum(s, (b) => b.rejected)} loại`,
    state: (s) => (s.state.items.length ? 'run' : 'idle'),
  },
  {
    cam: 'CAM-03', title: 'Thùng rác',
    text: (s) => `${s.state.kpi.trashCount} đã loại`,
    state: (s) => (s.state.items.some((i) => i.rejected) ? 'busy' : 'idle'),
  },
  {
    cam: 'CAM-04', title: 'Lò sấy',
    text: (s) => { const d = s.state.dryer; return d.batchId ? `${d.tempC.toFixed(0)}°C · ${Math.round((100 * d.t) / d.duration)}%` : 'Trống' },
    state: (s) => (s.state.dryer.batchId ? 'busy' : 'idle'),
  },
  {
    cam: 'CAM-05', title: 'AI mức sấy',
    text: (s) => `${sum(s, (b) => b.redried)} sấy lại · ${sum(s, (b) => b.overDried)} quá khô`,
    state: (s) => (s.state.units.length ? 'run' : 'idle'),
  },
  {
    cam: 'CAM-06', title: 'Kiểm nghiệm',
    text: (s) => `${sum(s, (b) => b.qcResults.filter((r) => r.pass).length)}/${sum(s, (b) => b.qcTaken)} mẫu đạt`,
    state: (s) => (s.state.batches.some((b) => b.status === 'grading' && b.qcTaken < b.qcTarget) ? 'run' : 'idle'),
  },
  {
    cam: 'CAM-07', title: 'Đóng gói',
    text: (s) => `${sum(s, (b) => b.bagsLabeled)} túi có QR`,
    state: (s) => (s.state.bags.some((b) => b.s < s.cfg.belt3Length) ? 'run' : 'idle'),
  },
]

export class Dashboard {
  private lastEventCount = -1
  private lastEventTime = -1
  private chipEls: HTMLButtonElement[] = []

  private sim: FactorySim

  constructor(onCam: (camId: string) => void, onQr: (b: Batch) => void, sim: FactorySim) {
    this.sim = sim
    const strip = $('#line-strip')
    for (const c of CHIPS) {
      const li = document.createElement('li')
      const btn = document.createElement('button')
      btn.type = 'button'
      btn.title = `Xem ${c.cam} — ${CAMERAS.find((x) => x.id === c.cam)?.name ?? ''}`
      // tiêu đề + nhãn trạng thái bằng chữ (không chỉ dựa vào màu viền) + số liệu
      const em = document.createElement('em')
      em.setAttribute('aria-hidden', 'true')
      btn.append(em, Object.assign(document.createElement('b'), { textContent: c.title }), document.createElement('span'))
      btn.addEventListener('click', () => onCam(c.cam))
      li.appendChild(btn)
      strip.appendChild(li)
      this.chipEls.push(btn)
    }
    const dlg = $<HTMLDialogElement>('#batches-dlg')
    $('#open-batches').addEventListener('click', () => {
      this.batchesSig = ''
      this.renderBatches(onQr)
      dlg.showModal()
    })
    for (const d of document.querySelectorAll<HTMLDialogElement>('dialog')) {
      d.querySelector('[data-close]')?.addEventListener('click', () => d.close())
      d.addEventListener('click', (e) => e.target === d && d.close())
    }
    this.onQr = onQr
  }

  private onQr: (b: Batch) => void

  update() {
    const sim = this.sim
    const s = sim.state
    $('#clock').textContent = sim.dateAt(s.time).toLocaleTimeString('vi-VN', { hour12: false })
    const cur = [...s.batches].reverse().find((b) => b.status !== 'done')
    $('#now-batch').textContent = cur
      ? `Lô ${cur.traceCode} · ${FRUITS[cur.fruit].vi} · ${STATUS_VI[cur.status]}`
      : s.batches.length ? 'Tất cả lô đã hoàn tất' : 'Chờ xe tải đầu tiên…'

    const recv = sum(sim, (b) => b.inspected)
    const rej = sum(sim, (b) => b.rejected)
    $('#k-recv').textContent = String(recv)
    $('#k-rej').textContent = String(rej)
    $('#k-rej-pct').textContent = pct(rej, recv)
    $('#k-bags').textContent = String(sum(sim, (b) => b.bagsLabeled))
    $('#k-qc').textContent = `${sum(sim, (b) => b.qcResults.filter((r) => r.pass).length)}/${sum(sim, (b) => b.qcTaken)}`
    const c = s.kpi.sortConfusion
    const n = c.tp + c.fp + c.tn + c.fn
    $('#k-ai').textContent = n ? pct(c.tp + c.tn, n) : '—'
    $('#k-ai-sub').textContent = n ? `độ chính xác · ${c.fn} bỏ sót · ${c.fp} loại nhầm` : 'độ chính xác'

    CHIPS.forEach((chip, i) => {
      const btn = this.chipEls[i]
      const st = chip.state(sim)
      if (btn.dataset.state !== st) {
        btn.dataset.state = st
        btn.title = `${chip.title} — ${STATE_VI[st]} · bấm để xem ${chip.cam}`
      }
      const text = chip.text(sim)
      const span = btn.querySelector('span')!
      if (span.textContent !== text) span.textContent = text
      btn.setAttribute('aria-label', `${chip.title}: ${STATE_VI[st]} — ${text}`)
    })

    const last = s.events[s.events.length - 1]
    if (s.events.length !== this.lastEventCount || (last && last.t !== this.lastEventTime)) {
      this.lastEventCount = s.events.length
      this.lastEventTime = last?.t ?? -1
      const ol = $('#log')
      ol.replaceChildren(
        ...s.events.slice(-60).reverse().map((e) => {
          const li = document.createElement('li')
          li.dataset.kind = e.kind
          const t = document.createElement('time')
          t.textContent = sim.dateAt(e.t).toLocaleTimeString('vi-VN', { hour12: false })
          const p = document.createElement('p')
          p.textContent = e.text
          li.append(t, p)
          return li
        }),
      )
    }
    const dlg = $<HTMLDialogElement>('#batches-dlg')
    if (dlg.open) this.renderBatches(this.onQr)
  }

  private batchesSig = ''

  private renderBatches(onQr: (b: Batch) => void) {
    // chỉ dựng lại khi dữ liệu đổi — tránh thay nút ngay lúc người dùng đang bấm
    const sig = this.sim.state.batches.map((b) => `${b.status}|${b.accepted}|${b.rejected}|${b.bagsLabeled}|${b.qcTaken}`).join(';')
    if (sig === this.batchesSig) return
    this.batchesSig = sig
    const body = $('#batches-body')
    if (!this.sim.state.batches.length) {
      const tr = document.createElement('tr')
      tr.className = 'empty-row'
      tr.appendChild(Object.assign(document.createElement('td'), { colSpan: 7, textContent: 'Chưa có lô nào — chờ xe tải đầu tiên vào dock.' }))
      body.replaceChildren(tr)
      return
    }
    body.replaceChildren(
      ...[...this.sim.state.batches].reverse().map((b) => {
        const tr = document.createElement('tr')
        const td = (text: string, cls = '') => Object.assign(document.createElement('td'), { textContent: text, className: cls })
        const pass = b.qcResults.filter((r) => r.pass).length
        const status = document.createElement('td')
        status.appendChild(Object.assign(document.createElement('span'), { className: 'pill', textContent: STATUS_VI[b.status] })).dataset.status = b.status
        tr.append(
          td(b.traceCode, 'mono'),
          td(FRUITS[b.fruit].vi),
          status,
          td(`${b.accepted} / ${b.rejected}`, 'mono num'),
          td(String(b.bagsLabeled), 'mono num'),
          td(`${pass}/${b.qcTaken}`, 'mono num'),
        )
        const act = document.createElement('td')
        if (b.bagsLabeled > 0) {
          const btn = Object.assign(document.createElement('button'), { type: 'button', className: 'btn ghost', textContent: 'Xem QR' })
          btn.setAttribute('aria-label', `Xem mã QR lô ${b.traceCode}`)
          btn.addEventListener('click', () => onQr(b))
          act.appendChild(btn)
        }
        tr.appendChild(act)
        return tr
      }),
    )
  }
}
