import type { Bag, Batch, FactorySim, StepName } from './engine'
import { FRUITS } from './params'
import { PRODUCER, STEP_NAMES, addMonthsIso, type ProductInfo } from './trace'

export { PRODUCER }

const STEP_KEYS: StepName[] = ['receive', 'sort', 'dry', 'grade', 'qc', 'pack', 'label']

const r1 = (x: number) => Math.round(x * 10) / 10
const r2 = (x: number) => Math.round(x * 100) / 100

/** Làm tròn tới giây (mã QR lưu theo giây). */
const toSec = (d: Date) => new Date(Math.round(d.getTime() / 1000) * 1000)

export function buildProductInfo(sim: FactorySim, b: Batch, bag: Bag): ProductInfo {
  const p = FRUITS[b.fruit]
  const packedAt = toSec(sim.dateAt(b.steps.pack ?? bag.packedAt))
  const n = b.qcResults.length
  const avg = (f: (r: (typeof b.qcResults)[number]) => number) => (n ? b.qcResults.reduce((s, r) => s + f(r), 0) / n : 0)
  return {
    v: 1,
    name: p.product,
    fruit: b.fruit,
    trace: b.traceCode,
    serial: bag.serial,
    producer: PRODUCER.name,
    address: PRODUCER.address,
    producedAt: packedAt.toISOString(),
    expiry: addMonthsIso(packedAt, p.shelfLifeMonths),
    massG: Math.round(bag.massKg * 1000),
    steps: STEP_KEYS.flatMap((k, i) => (b.steps[k] === undefined ? [] : [{ name: STEP_NAMES[i], at: toSec(sim.dateAt(b.steps[k]!)).toISOString() }])),
    qc: {
      samples: b.qcTaken,
      passed: b.qcResults.filter((r) => r.pass).length,
      moisture: r1(avg((r) => r.moisture)),
      aw: r2(avg((r) => r.aw)),
    },
  }
}
