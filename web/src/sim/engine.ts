/**
 * Engine mô phỏng dây chuyền sấy — TypeScript thuần, KHÔNG phụ thuộc Three.js.
 *
 * Tuyến: xe tải → băng 1 (camera AI phân loại, cửa gạt ra thùng rác) → sơ chế → lò sấy
 *        → băng 2 (camera AI mức sấy, trạm lấy mẫu QC, máy đóng gói) → băng 3 (máy dán nhãn QR).
 * Mọi kết quả AI ở đây là GIẢ LẬP theo tham số (recall, tỉ lệ loại nhầm…).
 */
import { DEFAULT_LINE, FRUITS, dryingLabel, type FruitType, type LineParams } from './params'
import { sampleSize } from './qc'
import { mulberry32, normal, type Rng } from './rng'
import { makeTraceCode } from './trace'

export type BatchStatus = 'receiving' | 'waiting_dryer' | 'drying' | 'grading' | 'done'
export type StepName = 'receive' | 'sort' | 'dry' | 'grade' | 'qc' | 'pack' | 'label' | 'done'

export interface Truck {
  id: number
  batchId: number
  fruit: FruitType
  phase: 'arriving' | 'unloading' | 'leaving'
  t: number
  remaining: number
}

export interface Item {
  id: number
  batchId: number
  fruit: FruitType
  defect: boolean
  defectType: string | null
  s: number
  ai: { defect: boolean; conf: number; label: string } | null
  rejected: boolean
  trashT: number
}

export interface Unit {
  id: number
  batchId: number
  fruit: FruitType
  moisture: number
  s: number
  ai: { label: 'under' | 'ok' | 'over'; est: number; conf: number } | null
  pass: number
}

export interface Bag {
  serial: string
  batchId: number
  fruit: FruitType
  massKg: number
  s: number
  labeled: boolean
  packedAt: number
}

export interface QcResult {
  moisture: number
  aw: number
  pass: boolean
}

export interface Batch {
  id: number
  seq: number
  fruit: FruitType
  traceCode: string
  status: BatchStatus
  total: number
  unloaded: number
  inspected: number
  accepted: number
  rejected: number
  acceptedMassKg: number
  driedMassKg: number
  moistureCurve: { t: number; m: number }[]
  units: number
  unitsResolved: number
  okAtQc: number
  redried: number
  overDried: number
  qcTarget: number
  qcTaken: number
  qcResults: QcResult[]
  bags: number
  bagsLabeled: number
  steps: Partial<Record<StepName, number>>
}

export interface Dryer {
  batchId: number | null
  mode: 'main' | 'redry'
  t: number
  duration: number
  tempC: number
  moisture: number
  redryUnits: Unit[]
}

/** Hiệu ứng chuyển tiếp cho cảnh 3D (cảnh tự rút qua drainFx). */
export interface Fx {
  kind: 'qc' | 'pack' | 'redry' | 'over' | 'to_dryer'
  fruit: FruitType
  s: number
  moisture?: number
}

export interface SimEvent {
  t: number
  kind: string
  batchId: number
  text: string
}

export interface SimState {
  time: number
  startDate: Date
  trucks: Truck[]
  items: Item[]
  units: Unit[]
  bags: Bag[]
  batches: Batch[]
  dryer: Dryer
  outQueue: Unit[]
  redryQueue: Unit[]
  events: SimEvent[]
  kpi: {
    sortConfusion: { tp: number; fp: number; tn: number; fn: number }
    trashCount: number
  }
}

const MAX_EVENTS = 400

export class FactorySim {
  readonly cfg: LineParams
  readonly state: SimState
  private rng: Rng
  private nextId = 1
  private trucksSpawned = 0
  private truckCooldown = 0
  private fx: Fx[] = []

  constructor(cfg: Partial<LineParams> = {}, seed = 1, startDate = new Date()) {
    this.cfg = { ...DEFAULT_LINE, ...cfg }
    this.rng = mulberry32(seed)
    this.state = {
      time: 0,
      startDate,
      trucks: [],
      items: [],
      units: [],
      bags: [],
      batches: [],
      dryer: { batchId: null, mode: 'main', t: 0, duration: 0, tempC: 25, moisture: 0, redryUnits: [] },
      outQueue: [],
      redryQueue: [],
      events: [],
      kpi: { sortConfusion: { tp: 0, fp: 0, tn: 0, fn: 0 }, trashCount: 0 },
    }
  }

  /** Lấy và xoá hàng đợi hiệu ứng (giới hạn 200 để không phình nếu cảnh không rút). */
  drainFx(): Fx[] {
    return this.fx.splice(0)
  }

  private pushFx(f: Fx) {
    this.fx.push(f)
    if (this.fx.length > 200) this.fx.splice(0, this.fx.length - 200)
  }

  /** Thời điểm thực (Date) tương ứng giây mô phỏng `t`. */
  dateAt(t: number): Date {
    return new Date(this.state.startDate.getTime() + t * 1000)
  }

  batch(id: number): Batch {
    return this.state.batches[id - 1]
  }

  step(dt: number): void {
    const s = this.state
    s.time += dt
    this.stepTrucks(dt)
    this.stepBelt1(dt)
    this.stepDryer(dt)
    this.emitUnits()
    this.stepBelt2(dt)
    this.stepBelt3(dt)
    this.checkBatchesDone()
  }

  // ---------------------------------------------------------------- nhận hàng
  private stepTrucks(dt: number) {
    const { cfg, state: s } = this
    if (s.trucks.length === 0) {
      this.truckCooldown -= dt
      const more = cfg.maxTrucks === 0 || this.trucksSpawned < cfg.maxTrucks
      if (more && this.truckCooldown <= 0) this.spawnTruck()
    }
    for (const tr of s.trucks) {
      tr.t += dt
      if (tr.phase === 'arriving' && tr.t >= cfg.truckArriveS) {
        tr.phase = 'unloading'
        tr.t = 0
        const b = this.batch(tr.batchId)
        b.steps.receive = s.time
        this.log('truck_arrived', b.id, `Xe tải #${tr.id} chở ${b.total} quả ${FRUITS[b.fruit].vi.toLowerCase()} đã vào dock — lô ${b.traceCode}`)
      } else if (tr.phase === 'unloading') {
        const last = s.items[s.items.length - 1]
        const free = !last || last.s >= cfg.itemGap
        if (tr.remaining > 0 && free) {
          this.placeItem(tr)
        } else if (tr.remaining === 0) {
          tr.phase = 'leaving'
          tr.t = 0
        }
      } else if (tr.phase === 'leaving' && tr.t >= cfg.truckLeaveS) {
        s.trucks = s.trucks.filter((x) => x !== tr)
        this.truckCooldown = cfg.truckIntervalS
      }
    }
  }

  private spawnTruck() {
    const { cfg, state: s } = this
    const fruit = cfg.schedule[this.trucksSpawned % cfg.schedule.length]
    this.trucksSpawned++
    const seq = s.batches.length + 1
    const b: Batch = {
      id: seq,
      seq,
      fruit,
      traceCode: makeTraceCode(fruit, this.dateAt(s.time), seq),
      status: 'receiving',
      total: cfg.truckLoad,
      unloaded: 0,
      inspected: 0,
      accepted: 0,
      rejected: 0,
      acceptedMassKg: 0,
      driedMassKg: 0,
      moistureCurve: [],
      units: 0,
      unitsResolved: 0,
      okAtQc: 0,
      redried: 0,
      overDried: 0,
      qcTarget: 0,
      qcTaken: 0,
      qcResults: [],
      bags: 0,
      bagsLabeled: 0,
      steps: {},
    }
    s.batches.push(b)
    s.trucks.push({ id: this.trucksSpawned, batchId: b.id, fruit, phase: 'arriving', t: 0, remaining: cfg.truckLoad })
  }

  private placeItem(tr: Truck) {
    const p = FRUITS[tr.fruit]
    const defect = this.rng() < p.defectRate
    const defectType = defect ? p.defectTypes[Math.floor(this.rng() * p.defectTypes.length)] : null
    this.state.items.push({
      id: this.nextId++, batchId: tr.batchId, fruit: tr.fruit, defect, defectType,
      s: 0, ai: null, rejected: false, trashT: 0,
    })
    tr.remaining--
    this.batch(tr.batchId).unloaded++
  }

  // ---------------------------------------------------------------- băng 1: phân loại AI
  private stepBelt1(dt: number) {
    const { cfg, state: s } = this
    const keep: Item[] = []
    for (const it of s.items) {
      const b = this.batch(it.batchId)
      if (it.rejected) {
        it.trashT += dt
        if (it.trashT < 1.5) keep.push(it)
        continue
      }
      it.s += cfg.beltSpeed * dt
      if (!it.ai && it.s >= cfg.cameraAt) this.classify(it, b)
      if (it.ai?.defect && it.s >= cfg.gateAt) {
        it.rejected = true
        b.rejected++
        s.kpi.trashCount++
        keep.push(it)
        continue
      }
      if (it.s >= cfg.belt1Length) {
        b.accepted++
        b.acceptedMassKg += FRUITS[it.fruit].massKg
        this.pushFx({ kind: 'to_dryer', fruit: it.fruit, s: it.s })
        continue
      }
      keep.push(it)
    }
    s.items = keep
    for (const b of s.batches) {
      if (b.status === 'receiving' && b.unloaded === b.total && b.accepted + b.rejected === b.total) {
        b.status = 'waiting_dryer'
        this.log('sorted', b.id, `Lô ${b.traceCode}: nhận ${b.accepted} quả, loại ${b.rejected} quả ra thùng rác`)
      }
    }
  }

  private classify(it: Item, b: Batch) {
    const { cfg, rng, state: s } = this
    const saysDefect = it.defect ? rng() < cfg.aiRecall : rng() < cfg.aiFalseReject
    const correct = saysDefect === it.defect
    const conf = correct ? 0.8 + 0.19 * rng() : 0.5 + 0.25 * rng()
    const label = saysDefect ? (it.defectType ?? 'hư') : 'tốt'
    it.ai = { defect: saysDefect, conf, label }
    b.inspected++
    b.steps.sort ??= s.time
    const c = s.kpi.sortConfusion
    if (saysDefect && it.defect) c.tp++
    else if (saysDefect) c.fp++
    else if (it.defect) c.fn++
    else c.tn++
    if (saysDefect) {
      this.log('reject', b.id, `Camera AI: ${FRUITS[it.fruit].vi.toLowerCase()} #${it.id} — ${label} (${Math.round(conf * 100)}%) → loại`)
    }
  }

  // ---------------------------------------------------------------- lò sấy
  private stepDryer(dt: number) {
    const s = this.state
    const d = s.dryer
    if (d.batchId === null) {
      if (s.redryQueue.length) return this.startRedry()
      const next = s.batches.find((b) => b.status === 'waiting_dryer')
      if (next) this.startMain(next)
      else d.tempC += (25 - d.tempC) * Math.min(1, dt * 0.2)
      return
    }
    const b = this.batch(d.batchId)
    const p = FRUITS[b.fruit]
    d.t = Math.min(d.duration, d.t + dt)
    d.tempC += (p.dryTempC - d.tempC) * Math.min(1, dt * 0.5)
    if (d.mode === 'main') {
      // Đường cong sấy mũ: tới cuối chu kỳ còn cách mục tiêu 0,5 điểm %.
      const k = Math.log((p.moistureIn - p.moistureTarget) / 0.5) / d.duration
      d.moisture = p.moistureTarget + (p.moistureIn - p.moistureTarget) * Math.exp(-k * d.t)
      const last = b.moistureCurve[b.moistureCurve.length - 1]
      if (!last || d.t - last.t >= d.duration / 40 || d.t >= d.duration) b.moistureCurve.push({ t: d.t, m: d.moisture })
    }
    if (d.t >= d.duration) (d.mode === 'main' ? this.finishMain(b) : this.finishRedry(b))
  }

  private startMain(b: Batch) {
    const d = this.state.dryer
    const p = FRUITS[b.fruit]
    Object.assign(d, { batchId: b.id, mode: 'main', t: 0, duration: (p.dryHours * 3600) / this.cfg.dryerCompression, moisture: p.moistureIn })
    b.status = 'drying'
    b.steps.dry = this.state.time
    b.moistureCurve = [{ t: 0, m: p.moistureIn }]
    this.log('dryer_start', b.id, `Lò sấy: bắt đầu lô ${b.traceCode} — ${b.acceptedMassKg.toFixed(1)} kg, ${p.dryTempC}°C, ${p.dryHours} giờ (nén thời gian)`)
  }

  private finishMain(b: Batch) {
    const { cfg, rng, state: s } = this
    const p = FRUITS[b.fruit]
    b.driedMassKg = b.acceptedMassKg * p.yield
    b.units = Math.round(b.driedMassKg / cfg.bagKg)
    b.qcTarget = sampleSize(b.units)
    b.status = 'grading'
    for (let i = 0; i < b.units; i++) {
      const moisture = Math.max(5, normal(rng, p.moistureTarget, cfg.dryingNoiseSd))
      s.outQueue.push({ id: this.nextId++, batchId: b.id, fruit: b.fruit, moisture, s: 0, ai: null, pass: 0 })
    }
    s.dryer.batchId = null
    this.log('dryer_done', b.id, `Lò sấy: xong lô ${b.traceCode} — ${b.driedMassKg.toFixed(1)} kg thành phẩm (${b.units} phần 200 g), lấy mẫu QC ${b.qcTarget}`)
  }

  private startRedry() {
    const { state: s } = this
    const d = s.dryer
    const units = s.redryQueue.splice(0)
    const b = this.batch(units[0].batchId)
    const main = (FRUITS[b.fruit].dryHours * 3600) / this.cfg.dryerCompression
    Object.assign(d, { batchId: b.id, mode: 'redry', t: 0, duration: main * this.cfg.redryFraction, redryUnits: units })
    this.log('redry_start', b.id, `Lò sấy: sấy bổ sung ${units.length} phần chưa đạt của lô ${b.traceCode}`)
  }

  private finishRedry(b: Batch) {
    const { rng, state: s } = this
    const target = FRUITS[b.fruit].moistureTarget
    for (const u of s.dryer.redryUnits) {
      u.moisture = target + Math.min(1.5, normal(rng, 0, 0.6))
      u.s = 0
      u.ai = null
      u.pass++
      s.outQueue.push(u)
    }
    s.dryer.redryUnits = []
    s.dryer.batchId = null
  }

  private emitUnits() {
    const { cfg, state: s } = this
    if (!s.outQueue.length) return
    const last = s.units[s.units.length - 1]
    if (!last || last.s >= cfg.unitGap) s.units.push(s.outQueue.shift()!)
  }

  // ---------------------------------------------------------------- băng 2: mức sấy, QC, đóng gói
  private stepBelt2(dt: number) {
    const { cfg, rng, state: s } = this
    const keep: Unit[] = []
    for (const u of s.units) {
      const b = this.batch(u.batchId)
      const p = FRUITS[u.fruit]
      const before = u.s
      u.s += cfg.belt2Speed * dt
      if (!u.ai && u.s >= cfg.dryCamAt) {
        const est = u.moisture + normal(rng, 0, cfg.dryAiSd)
        const label = dryingLabel(est, p.moistureTarget)
        u.ai = { label, est, conf: 0.7 + 0.29 * rng() }
        b.steps.grade ??= s.time
        if (label === 'under') {
          b.redried++
          s.redryQueue.push(u)
          this.pushFx({ kind: 'redry', fruit: u.fruit, s: u.s, moisture: u.moisture })
          this.log('dry_under', b.id, `Camera mức sấy: phần #${u.id} độ ẩm ước ${est.toFixed(1)}% → sấy bổ sung`)
          continue
        }
        if (label === 'over') {
          b.overDried++
          b.unitsResolved++
          s.kpi.trashCount++
          this.pushFx({ kind: 'over', fruit: u.fruit, s: u.s, moisture: u.moisture })
          this.log('dry_over', b.id, `Camera mức sấy: phần #${u.id} độ ẩm ước ${est.toFixed(1)}% → quá khô, loại`)
          continue
        }
      }
      if (before < cfg.qcAt && u.s >= cfg.qcAt && this.takeQcSample(b)) {
        const measured = u.moisture + normal(rng, 0, 0.3)
        const aw = 0.25 + 0.02 * measured
        const pass = measured <= p.moistureTarget + 3 && aw <= 0.7
        b.qcTaken++
        b.unitsResolved++
        b.qcResults.push({ moisture: measured, aw, pass })
        b.steps.qc ??= s.time
        this.pushFx({ kind: 'qc', fruit: u.fruit, s: u.s, moisture: u.moisture })
        this.log('qc_sample', b.id, `Lấy mẫu QC ${b.qcTaken}/${b.qcTarget} lô ${b.traceCode}: ẩm ${measured.toFixed(1)}%, aw ${aw.toFixed(2)} → ${pass ? 'đạt' : 'KHÔNG đạt'}`)
        continue
      }
      if (u.s >= cfg.packAt) {
        b.bags++
        b.unitsResolved++
        b.steps.pack ??= s.time
        const serial = `${b.traceCode}-${String(b.bags).padStart(4, '0')}`
        this.pushFx({ kind: 'pack', fruit: u.fruit, s: u.s })
        s.bags.push({ serial, batchId: b.id, fruit: u.fruit, massKg: cfg.bagKg, s: 0, labeled: false, packedAt: s.time })
        continue
      }
      keep.push(u)
    }
    s.units = keep
  }

  /** Lấy mẫu rải đều theo lô; cuối lô thì lấy bù để đủ cỡ mẫu nếu còn đủ đơn vị. */
  private takeQcSample(b: Batch): boolean {
    const idx = b.okAtQc++
    const need = b.qcTarget - b.qcTaken
    if (need <= 0) return false
    const remaining = b.units - b.unitsResolved
    const every = Math.max(1, Math.floor(b.units / b.qcTarget))
    return idx % every === 0 || remaining <= need
  }

  // ---------------------------------------------------------------- băng 3: dán nhãn QR
  private stepBelt3(dt: number) {
    const { cfg, state: s } = this
    for (const bag of s.bags) {
      if (bag.s >= cfg.belt3Length) continue
      bag.s = Math.min(cfg.belt3Length, bag.s + cfg.belt3Speed * dt)
      if (!bag.labeled && bag.s >= cfg.labelAt) {
        bag.labeled = true
        const b = this.batch(bag.batchId)
        b.bagsLabeled++
        b.steps.label ??= s.time
        if (b.bagsLabeled === 1 || b.bagsLabeled % 25 === 0) {
          this.log('bag_labeled', b.id, `Dán nhãn QR túi ${bag.serial} (${b.bagsLabeled} túi)`)
        }
      }
    }
  }

  private checkBatchesDone() {
    const s = this.state
    for (const b of s.batches) {
      if (b.status !== 'grading') continue
      if (b.unitsResolved === b.units && b.bagsLabeled === b.bags) {
        b.status = 'done'
        b.steps.done = s.time
        const passed = b.qcResults.filter((r) => r.pass).length
        this.log('batch_done', b.id, `Hoàn tất lô ${b.traceCode}: ${b.bags} túi, QC đạt ${passed}/${b.qcTaken}`)
      }
    }
  }

  private log(kind: string, batchId: number, text: string) {
    const ev = this.state.events
    ev.push({ t: this.state.time, kind, batchId, text })
    if (ev.length > MAX_EVENTS) ev.splice(0, ev.length - MAX_EVENTS)
  }
}
