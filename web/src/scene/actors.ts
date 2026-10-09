/** Đối tượng động: đọc SimState mỗi khung hình và cập nhật InstancedMesh / nhóm. */
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { FactorySim, Fx, Item } from '../sim/engine'
import { FRUITS, FRUIT_TYPES, dryingLabel, type FruitType } from '../sim/params'
import type { FactoryHandles, Pusher } from './factory'
import { bagGeometry, bagMaterial, fruitGeometry, fruitMaterials, productGeometry, productMaterial, productTrayGeometry } from './fruitModels'
import { BELT_Y, EJECT_OVER_S, EJECT_REDRY_S, FEED, L, SPOTS, belt1, belt2, belt3 } from './layout'

const CAP = 256
const PROD_CAP = 384
const tmpM = new THREE.Matrix4()
const tmpQ = new THREE.Quaternion()
const tmpS = new THREE.Vector3()
const tmpP = new THREE.Vector3()
const tmpV = new THREE.Vector3()
const tmpC = new THREE.Color()
const tmpC2 = new THREE.Color()
const ROT = new THREE.Euler()
const ONE = new THREE.Vector3(1, 1, 1)

/** Hash ổn định theo id → [0,1) để rải lệch/xoay. */
const h01 = (n: number) => (((n * 2654435761) >>> 0) % 10007) / 10007
const ease = (t: number) => t * t * (3 - 2 * t)
const clamp01 = (t: number) => Math.min(1, Math.max(0, t))

// ---- quả rơi từ phễu: rơi HOPPER_DROP m trong DROP_S m đầu băng 1
const DROP_S = 0.25
const HOPPER_DROP = 0.3
// ---- cửa gạt băng 1 (giây kể từ lúc bị gạt; engine xoá quả ở 1.5 s)
const T_PUSH = 0.35
const T_CHUTE = 0.75
const T_FALL = 1.05
const T_SINK = 1.45
const chute1Y = (z: number) => 0.8 + (z + 0.5) * (0.16 / 1.05)
// ---- cửa đẩy băng 2 (giây kể từ lúc tới điểm đẩy)
const E_PUSH = 0.3
const E_CHUTE = 0.55
const E_FALL = 0.85
const E_END = 1.1
// ---- lấy mẫu QC
const QC_DESC = 0.25
const QC_MOVE = 0.85
const QC_REST = 1.6
const QC_DUR = 3.6
// ---- túi rơi vào thùng cuối băng 3, thùng đầy chuyển sang pallet
const BAG_DROP = 0.5
const CARTON_MOVE = 1.2
const BAGS_PER_CARTON = 20
const PALLET_SLOTS = 16

function instanced(geo: THREE.BufferGeometry, mat: THREE.Material, cap = CAP) {
  const m = new THREE.InstancedMesh(geo, mat, cap)
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  m.count = 0
  m.castShadow = true
  m.setColorAt(0, new THREE.Color())
  return m
}

/** Đánh dấu cập nhật GPU và ẩn mesh rỗng (bớt draw call ở 9 khung nhìn). */
function commit(m: THREE.InstancedMesh) {
  m.visible = m.count > 0
  // tính lại hình cầu bao theo instance hiện tại → mỗi ô camera tự cull mesh nằm ngoài khung
  m.boundingSphere = null
  m.instanceMatrix.needsUpdate = true
  if (m.instanceColor) m.instanceColor.needsUpdate = true
}

/** Màu sản phẩm sau sấy theo độ ẩm (ẩm hơn → nhạt hơn, quá khô → nâu sẫm). */
function driedColor(f: FruitType, moisture: number, out: THREE.Color) {
  const p = FRUITS[f]
  out.set(p.color).lerp(tmpC2.set(0x9a5a1c), 0.35)
  const lab = dryingLabel(moisture, p.moistureTarget)
  if (lab === 'under') out.lerp(tmpC2.set(0xfff1c8), 0.35)
  if (lab === 'over') out.lerp(tmpC2.set(0x4a2a10), 0.55)
  return out
}

interface FruitSet {
  good: THREE.InstancedMesh
  bad: THREE.InstancedMesh
  /** nửa chiều dài (trục x) và nửa bề ngang (trục z) của mô hình */
  hx: number
  hz: number
  hy: number
}

interface ActiveFx extends Fx {
  t: number
  dur: number
  /** ô xếp trong xe khay sấy lại (redry) */
  slot: number
}

interface TruckRig {
  group: THREE.Group
  crates: THREE.InstancedMesh
  /** 6 bánh = 1 InstancedMesh (quay quanh trục riêng mỗi khung) */
  wheels: THREE.InstancedMesh
}

const WHEELS: [number, number][] = [[-3.4, -1.1], [-3.4, 1.1], [-0.8, -1.1], [-0.8, 1.1], [1.6, -1.1], [1.6, 1.1]]

/** Tô màu đỉnh cho cả hình (để gộp các phần khác màu vào một mesh). */
function tint(g: THREE.BufferGeometry, hex: number) {
  const c = new THREE.Color(hex)
  const n = g.attributes.position.count
  const a = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) a.set([c.r, c.g, c.b], i * 3)
  g.setAttribute('color', new THREE.BufferAttribute(a, 3))
  return g
}

function makeTruck(): TruckRig {
  const group = new THREE.Group()
  const b = (w: number, h: number, d: number, x: number, y: number, z: number, hex: number) => tint(new THREE.BoxGeometry(w, h, d).translate(x, y, z), hex)
  // thân xe gộp một mesh (màu đỉnh) thay vì ~10 mesh → bớt draw call ở 9 khung nhìn
  const PAINT = 0x2b5d8a
  const DARK = 0x1d2228
  const bodyGeo = mergeGeometries([
    b(1.9, 2.1, 2.3, -3.4, 1.55, 0, PAINT), // cabin
    b(0.05, 0.8, 1.9, -4.36, 1.95, 0, 0x9ec9e8), // kính
    b(0.12, 0.25, 2.3, -4.4, 0.6, 0, DARK), // cản trước
    b(5, 0.25, 2.4, 0, 0.75, 0, DARK), // sàn thùng
    b(5, 0.9, 0.06, 0, 1.3, -1.17, PAINT),
    b(5, 0.9, 0.06, 0, 1.3, 1.17, PAINT),
    b(0.06, 0.9, 2.4, -2.47, 1.3, 0, PAINT),
  ])!
  const body = new THREE.Mesh(bodyGeo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.25 }))
  body.castShadow = true
  group.add(body)
  const HUB = 0xb8bec6
  const wheelGeo = mergeGeometries([
    tint(new THREE.CylinderGeometry(0.45, 0.45, 0.3, 20).rotateX(Math.PI / 2), DARK),
    tint(new THREE.CylinderGeometry(0.2, 0.2, 0.32, 10).rotateX(Math.PI / 2), HUB),
    tint(new THREE.BoxGeometry(0.62, 0.07, 0.31), HUB),
    tint(new THREE.BoxGeometry(0.07, 0.62, 0.31), HUB),
  ])!
  const wheels = new THREE.InstancedMesh(wheelGeo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7 }), WHEELS.length)
  wheels.castShadow = true
  wheels.frustumCulled = false
  group.add(wheels)
  const crates = new THREE.InstancedMesh(new THREE.BoxGeometry(0.75, 0.45, 0.55), new THREE.MeshStandardMaterial({ roughness: 0.9 }), 24)
  crates.castShadow = true
  group.add(crates)
  return { group, crates, wheels }
}

export class Actors {
  private fruits = new Map<FruitType, FruitSet>()
  private prod = new Map<FruitType, THREE.InstancedMesh>()
  private prodTrays: THREE.InstancedMesh
  private bags: THREE.InstancedMesh
  private labels: THREE.InstancedMesh
  private trays: THREE.InstancedMesh
  private trayFill: THREE.InstancedMesh
  private cartons: THREE.InstancedMesh
  private trucks: TruckRig[] = []
  private fx: ActiveFx[] = []
  private panelTimer = 0
  private binFillY = 0.03
  private redrySlots = 0
  private prevDryer: number | null = null
  private trolleyAnim: { t: number; n: number; fruit: FruitType } | null = null
  /** vị trí 3D của quả theo id (dùng cho khung bbox camera) */
  readonly itemPos = new Map<number, THREE.Vector3>()
  readonly unitPos = new Map<number, THREE.Vector3>()

  private h: FactoryHandles

  constructor(scene: THREE.Scene, h: FactoryHandles) {
    this.h = h
    for (const f of FRUIT_TYPES) {
      const geo = fruitGeometry(f)
      const bb = geo.boundingBox!
      const { good, bad } = fruitMaterials(f)
      const set: FruitSet = {
        good: instanced(geo, good),
        bad: instanced(geo, bad),
        hx: (bb.max.x - bb.min.x) / 2,
        hz: (bb.max.z - bb.min.z) / 2,
        hy: bb.max.y - bb.min.y,
      }
      this.fruits.set(f, set)
      scene.add(set.good, set.bad)
      const p = instanced(productGeometry(f), productMaterial(f), PROD_CAP)
      this.prod.set(f, p)
      scene.add(p)
    }
    this.prodTrays = instanced(productTrayGeometry(), new THREE.MeshStandardMaterial({ color: 0xc3c9d0, roughness: 0.4, metalness: 0.6 }), PROD_CAP)
    this.bags = instanced(bagGeometry(), bagMaterial())
    const labelTex = qrLikeTexture()
    this.labels = instanced(new THREE.PlaneGeometry(0.11, 0.11).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ map: labelTex }))
    this.labels.castShadow = false
    this.trays = instanced(new THREE.BoxGeometry(0.6, 0.025, 0.88).translate(0, 0.0125, 0), new THREE.MeshStandardMaterial({ color: 0xc3c9d0, roughness: 0.4, metalness: 0.6 }), 32)
    this.trayFill = instanced(new THREE.BoxGeometry(0.54, 0.014, 0.82).translate(0, 0.032, 0), productMaterial('mango'), 32)
    this.trayFill.castShadow = false
    this.cartons = instanced(new THREE.BoxGeometry(0.5, 0.36, 0.42), h.cartonMat, PALLET_SLOTS + 2)
    for (const m of [this.prodTrays, this.bags, this.labels, this.trays, this.trayFill, this.cartons]) scene.add(m)
    for (let i = 0; i < 2; i++) {
      const t = makeTruck()
      t.group.visible = false
      scene.add(t.group)
      this.trucks.push(t)
    }
  }

  update(sim: FactorySim, dtSim: number, dtReal: number) {
    this.binFillY = 0.03 + Math.min(0.42, sim.state.kpi.trashCount * 0.003)
    this.updateFruits(sim)
    this.updateProducts(sim, dtSim)
    this.updateBags(sim)
    this.updateTrucks(sim)
    this.updateMachines(sim, dtSim, dtReal)
  }

  // ---------------------------------------------------------------- quả: băng cấp liệu → phễu → băng 1 → cửa gạt
  private updateFruits(sim: FactorySim) {
    const items = sim.state.items
    for (const s of this.fruits.values()) s.good.count = s.bad.count = 0
    this.itemPos.clear()
    for (const it of items) {
      const set = this.fruits.get(it.fruit)!
      const m = it.defect ? set.bad : set.good
      if (m.count >= CAP) continue
      // xoay quanh trục đứng giới hạn ±0.6 rad (quả dài nằm dọc băng) + lật đầu ngẫu nhiên
      const yaw = (h01(it.id * 7) - 0.5) * 1.2 + (h01(it.id * 13) < 0.5 ? 0 : Math.PI)
      const hz = Math.abs(set.hx * Math.sin(yaw)) + set.hz * Math.abs(Math.cos(yaw))
      const latMax = Math.max(0, 0.41 - hz)
      const lat = (h01(it.id) - 0.5) * 2 * latMax
      let roll = 0
      if (!it.rejected) {
        belt1(it.s, tmpP)
        // rơi từ miệng phễu xuống băng rồi dạt dần về làn của mình
        tmpP.z += lat * ease(clamp01(it.s / 0.7))
        if (it.s < DROP_S) {
          const q = it.s / DROP_S
          tmpP.y += HOPPER_DROP * (1 - q * q)
          roll = (1 - q) * 0.6 * (h01(it.id * 3) - 0.5)
        }
      } else {
        this.rejectPos(it, lat, set.hy, tmpP)
        roll = it.trashT > T_PUSH ? (it.trashT - T_PUSH) * 5 : 0
      }
      this.itemPos.set(it.id, tmpP.clone())
      ROT.set(roll, yaw, 0)
      tmpQ.setFromEuler(ROT)
      const k = 0.92 + 0.16 * h01(it.id * 3)
      m.setMatrixAt(m.count, tmpM.compose(tmpP, tmpQ, tmpS.set(k, k, k)))
      const b = 0.86 + 0.14 * h01(it.id * 11)
      m.setColorAt(m.count, tmpC.setRGB(b, b, b))
      m.count++
    }
    this.feederFruits(sim)
    for (const s of this.fruits.values()) {
      commit(s.good)
      commit(s.bad)
    }
  }

  /** Quả bị gạt: cửa gạt đẩy qua mép băng → trượt máng → rơi vào thùng → lún vào đống rác. */
  private rejectPos(it: Item, lat: number, hy: number, out: THREE.Vector3) {
    const t = it.trashT
    const gx = SPOTS.gate.x
    if (t < T_PUSH) {
      const z = lat + (-0.52 - lat) * ease(t / T_PUSH)
      const lift = clamp01((-0.4 - z) / 0.1) * 0.05
      return out.set(gx, BELT_Y + lift, z)
    }
    if (t < T_CHUTE) {
      const q = (t - T_PUSH) / (T_CHUTE - T_PUSH)
      const z = -0.52 + (-1.55 + 0.52) * q * q
      return out.set(gx, chute1Y(z) + 0.1 * (1 - q) ** 3, z)
    }
    const tx = gx + (h01(it.id * 5) - 0.5) * 0.4
    const tz = SPOTS.trash.z + (h01(it.id * 17) - 0.5) * 0.4
    if (t < T_FALL) {
      const q = (t - T_CHUTE) / (T_FALL - T_CHUTE)
      return out.set(gx + (tx - gx) * q, 0.64 - (0.64 - this.binFillY) * q * q, -1.55 + (tz + 1.55) * q)
    }
    const q = clamp01((t - T_FALL) / (T_SINK - T_FALL))
    return out.set(tx, this.binFillY - hy * q, tz)
  }

  /** Quả "sắp tới" trên băng cấp liệu: nhịp đặt quả của engine (itemGap) quy ra khoảng cách → liền mạch khi quả rơi vào phễu. */
  private feederFruits(sim: FactorySim) {
    const tr = sim.state.trucks.find((t) => t.phase === 'unloading')
    this.h.hopperPile.visible = !!tr && tr.remaining > 0
    if (!tr || tr.remaining <= 0) return
    const set = this.fruits.get(tr.fruit)!
    ;(this.h.hopperPile.material as THREE.MeshStandardMaterial).color.set(FRUITS[tr.fruit].color).multiplyScalar(0.8)
    const gap = sim.cfg.itemGap
    const items = sim.state.items
    const last = items[items.length - 1]
    const sl = last && last.batchId === tr.batchId ? Math.min(last.s, gap) : gap
    const HIDDEN = 0.55
    const { foot, head } = FEED
    const feedLen = foot.distanceTo(head)
    const dir = tmpV.copy(foot).sub(head).normalize()
    const pitch = Math.atan2(head.y - foot.y, head.x - foot.x)
    const placed = sim.batch(tr.batchId).total - tr.remaining
    for (let k = 1; k <= tr.remaining && set.good.count < CAP; k++) {
      const d = k * gap - sl - HIDDEN
      const key = tr.id * 4096 + placed + k
      const yaw = (h01(key * 7) - 0.5) * 0.5
      if (d < 0) {
        // rơi từ đầu băng vào phễu (khuất sau thành phễu)
        const q = -d / HIDDEN
        tmpP.set(head.x + 0.35 * q, head.y + 0.02 - 1.0 * q * q, 0)
        ROT.set(q * 1.5, yaw, 0)
      } else if (d <= feedLen) {
        tmpP.copy(head).addScaledVector(dir, d)
        tmpP.z = (h01(key) - 0.5) * Math.max(0, 0.5 - 2 * set.hz)
        ROT.set(0, yaw, pitch, 'ZYX') // xoay ngang trước, rồi nghiêng theo mặt băng
      } else break
      tmpQ.setFromEuler(ROT)
      ROT.order = 'XYZ'
      const m = set.good
      m.setMatrixAt(m.count, tmpM.compose(tmpP, tmpQ, ONE))
      m.setColorAt(m.count, tmpC.setRGB(0.95, 0.95, 0.95))
      m.count++
    }
  }

  // ---------------------------------------------------------------- phần sản phẩm trên băng 2 + hiệu ứng rẽ nhánh
  private drawProduct(f: FruitType, moisture: number, p: THREE.Vector3, yaw: number) {
    const m = this.prod.get(f)!
    if (m.count >= PROD_CAP) return
    tmpQ.setFromAxisAngle(THREE.Object3D.DEFAULT_UP, yaw)
    tmpM.compose(p, tmpQ, ONE)
    m.setMatrixAt(m.count, tmpM)
    m.setColorAt(m.count, driedColor(f, moisture, tmpC))
    m.count++
    const t = this.prodTrays
    t.setMatrixAt(t.count, tmpM)
    t.setColorAt(t.count, tmpC.setRGB(1, 1, 1))
    t.count++
  }

  private updateProducts(sim: FactorySim, dtSim: number) {
    for (const m of this.prod.values()) m.count = 0
    this.prodTrays.count = 0
    this.unitPos.clear()
    for (const u of sim.state.units) {
      belt2(u.s, tmpP)
      tmpP.z += (h01(u.id) - 0.5) * 0.12
      this.unitPos.set(u.id, tmpP.clone())
      this.drawProduct(u.fruit, u.moisture, tmpP, (h01(u.id * 5) - 0.5) * 0.5)
    }
    // hiệu ứng: quá khô → thùng loại, chưa đạt → xe khay sấy lại, mẫu QC → bàn mẫu → phòng kiểm nghiệm
    if (sim.state.redryQueue.length === 0) this.redrySlots = 0
    for (const f of sim.drainFx()) {
      if (f.kind === 'to_dryer' || f.kind === 'pack') continue // vào máy (khuất) — không cần vẽ
      const ej = f.kind === 'over' ? EJECT_OVER_S : EJECT_REDRY_S
      const dur = f.kind === 'qc' ? QC_DUR : Math.max(0, (ej - f.s) / L.belt2Speed) + (f.kind === 'over' ? E_END : E_FALL)
      this.fx.push({ ...f, t: 0, dur, slot: f.kind === 'redry' ? this.redrySlots++ : 0 })
    }
    this.fx = this.fx.filter((f) => (f.t += dtSim) < f.dur)
    let armZ = -6.0
    let headY = 1.36
    let kickOver = false
    let kickRedry = false
    const inFlight = new Set<number>()
    for (const f of this.fx) {
      const moist = f.moisture ?? FRUITS[f.fruit].moistureTarget
      if (f.kind === 'qc') {
        const r = this.qcPos(f, tmpP)
        if (r.arm) {
          armZ = tmpP.z
          headY = tmpP.y + 0.045
        }
      } else {
        const r = this.ejectPos(f, tmpP)
        if (r.pushing) f.kind === 'over' ? (kickOver = true) : (kickRedry = true)
        if (f.kind === 'redry') inFlight.add(f.slot)
      }
      this.drawProduct(f.fruit, moist, tmpP, 0)
    }
    // phần chờ sấy lại nằm trong xe khay (xếp 2 cột)
    const q = sim.state.redryQueue
    for (let i = 0; i < Math.min(q.length, 24); i++) {
      if (inFlight.has(i)) continue
      this.redrySlotPos(i, tmpP)
      this.drawProduct(q[i].fruit, q[i].moisture, tmpP, 0)
    }
    for (const m of this.prod.values()) commit(m)
    commit(this.prodTrays)
    // tay gắp QC
    const arm = this.h.qcArm
    arm.carriage.position.z += (armZ - arm.carriage.position.z) * 0.35
    const local = Math.min(-0.12, headY - arm.carriage.position.y)
    arm.head.position.y += (local - arm.head.position.y) * 0.35
    arm.rod.scale.y = -arm.head.position.y
    arm.rod.position.y = arm.head.position.y / 2
    this.kick = { over: kickOver, redry: kickRedry }
  }

  private kick = { over: false, redry: false }

  private redrySlotPos(i: number, out: THREE.Vector3) {
    const c = SPOTS.redryCart
    return out.set(c.x + ((i % 2) - 0.5) * 0.3, 0.25 + Math.floor(i / 2) * 0.026, c.z)
  }

  /** Đi tiếp trên băng tới cửa đẩy → bị đẩy qua mép → trượt máng → rơi vào thùng loại / xe khay. */
  private ejectPos(f: ActiveFx, out: THREE.Vector3): { pushing: boolean } {
    const over = f.kind === 'over'
    const ej = over ? EJECT_OVER_S : EJECT_REDRY_S
    const tBelt = Math.max(0, (ej - f.s) / L.belt2Speed)
    if (f.t < tBelt) {
      belt2(f.s + f.t * L.belt2Speed, out)
      return { pushing: f.t > tBelt - 0.05 }
    }
    const e = f.t - tBelt
    const dir = over ? 1 : -1
    const x = belt2(ej).x
    if (e < E_PUSH) {
      const zz = 0.5 * ease(e / E_PUSH)
      out.set(x, BELT_Y + clamp01((zz - 0.36) / 0.1) * 0.05, -6 + dir * zz)
      return { pushing: true }
    }
    const z0 = -6 + dir * 0.5
    const z1 = over ? -5.3 : -6.82
    const y1 = over ? 0.62 : 0.64
    if (e < E_CHUTE) {
      const q = (e - E_PUSH) / (E_CHUTE - E_PUSH)
      out.set(x, 0.8 + (y1 - 0.8) * q * q + 0.1 * (1 - q) ** 3, z0 + (z1 - z0) * q * q)
      return { pushing: false }
    }
    const target = over ? tmpV.set(SPOTS.trash2.x, 0.05, SPOTS.trash2.z) : this.redrySlotPos(f.slot, tmpV)
    if (e < E_FALL) {
      const q = (e - E_CHUTE) / (E_FALL - E_CHUTE)
      out.set(x + (target.x - x) * q, y1 - (y1 - target.y) * q * q, z1 + (target.z - z1) * q)
      return { pushing: false }
    }
    out.copy(target)
    out.y -= 0.03 * clamp01((e - E_FALL) / (E_END - E_FALL)) // lún vào đáy thùng loại
    return { pushing: false }
  }

  /** Mẫu QC: tay gắp hạ xuống → nhấc sang bàn mẫu → nằm chờ → trượt theo băng mẫu vào cửa chuyển phòng kiểm nghiệm. */
  private qcPos(f: ActiveFx, out: THREE.Vector3): { arm: boolean } {
    const t = f.t
    const x = SPOTS.qc.x
    const tz = SPOTS.qcTable.z
    if (t < QC_DESC) {
      belt2(f.s + t * L.belt2Speed * 0.5, out)
      return { arm: true }
    }
    const p0x = belt2(f.s + QC_DESC * L.belt2Speed * 0.5).x
    if (t < QC_MOVE) {
      const q = ease((t - QC_DESC) / (QC_MOVE - QC_DESC))
      out.set(p0x + (x - p0x) * q, BELT_Y + (0.82 - BELT_Y) * q + 0.32 * Math.sin(Math.PI * q), -6 + (tz + 6) * q)
      return { arm: true }
    }
    if (t < QC_REST) {
      out.set(x, 0.82, tz)
      return { arm: false }
    }
    const q = clamp01((t - QC_REST) / (QC_DUR - QC_REST))
    out.set(x, 0.82, tz + (-9.3 - tz) * q * q)
    return { arm: false }
  }

  // ---------------------------------------------------------------- túi trên băng 3 + nhãn QR → thùng carton → pallet
  private updateBags(sim: FactorySim) {
    const { bags, labels, cartons } = this
    const all = sim.state.bags
    const time = sim.state.time
    const travel = L.belt3Length / L.belt3Speed
    const st = SPOTS.cartonStation
    bags.count = labels.count = cartons.count = 0
    let i = all.length - 1
    for (; i >= 0 && bags.count < CAP - BAGS_PER_CARTON; i--) {
      const b = all[i]
      let tEnd = 0
      if (b.s < L.belt3Length) {
        belt3(b.s, tmpP)
      } else {
        tEnd = Math.max(0, time - (b.packedAt + travel))
        if (tEnd >= BAG_DROP) break // từ đây về trước: đã nằm yên trong thùng
        // trượt khỏi đầu băng, rơi vào ô của mình trong thùng
        const q = tEnd / BAG_DROP
        const slot = this.bagSlot(i % BAGS_PER_CARTON, tmpV)
        const p0 = belt3(L.belt3Length)
        tmpP.set(p0.x + (slot.x - p0.x) * q, BELT_Y + (slot.y - BELT_Y) * q * q, p0.z + (slot.z - p0.z) * q)
      }
      this.drawBag(b.fruit, tmpP, b.labeled, tEnd > 0 ? (tEnd / BAG_DROP) * 0.5 : 0)
    }
    const done = i + 1 // số túi đã nằm trong thùng (chỉ số 0..i)
    // túi đang có trong thùng đang đóng (thùng đầy được đóng nắp và chuyển sang pallet)
    const inCarton = done % BAGS_PER_CARTON
    for (let k = 0; k < inCarton; k++) {
      const b = all[done - inCarton + k]
      this.drawBag(b.fruit, this.bagSlot(k, tmpP), true, 0)
    }
    // thùng trên pallet; thùng vừa đầy bay từ trạm vô thùng sang ô của nó
    const full = Math.floor(done / BAGS_PER_CARTON)
    if (full > 0) {
      const lastFull = full - 1
      const doneAt = all[lastFull * BAGS_PER_CARTON + BAGS_PER_CARTON - 1].packedAt + travel + BAG_DROP
      const age = time - doneAt
      const first = Math.floor(lastFull / PALLET_SLOTS) * PALLET_SLOTS
      for (let c = first; c <= lastFull; c++) {
        this.cartonSlot(c % PALLET_SLOTS, tmpP)
        if (c === lastFull && age < CARTON_MOVE) {
          const q = ease(clamp01(age / CARTON_MOVE))
          const sx = st.x
          tmpP.set(sx + (tmpP.x - sx) * q, 0.44 + (tmpP.y - 0.44) * q + 0.45 * Math.sin(Math.PI * q), st.z + (tmpP.z - st.z) * q)
        }
        cartons.setMatrixAt(cartons.count++, tmpM.compose(tmpP, tmpQ.identity(), ONE))
      }
    }
    commit(bags)
    commit(labels)
    commit(cartons)
  }

  private drawBag(fruit: FruitType, p: THREE.Vector3, labeled: boolean, tilt: number) {
    const { bags, labels } = this
    if (bags.count >= CAP) return
    tmpQ.setFromAxisAngle(tmpV.set(0, 0, 1), tilt)
    bags.setMatrixAt(bags.count, tmpM.compose(p, tmpQ, ONE))
    bags.setColorAt(bags.count, tmpC.setRGB(1, 1, 1).lerp(tmpC2.set(FRUITS[fruit].color), 0.4))
    bags.count++
    if (labeled) {
      tmpS.set(0, 0.058, 0.06).applyQuaternion(tmpQ).add(p)
      labels.setMatrixAt(labels.count++, tmpM.compose(tmpS, tmpQ, ONE))
    }
  }

  /** Ô xếp túi trong thùng tại trạm vô thùng: 2 cột × 10 lớp. */
  private bagSlot(k: number, out: THREE.Vector3) {
    const st = SPOTS.cartonStation
    return out.set(st.x + ((k % 2) - 0.5) * 0.235, 0.27 + Math.floor(k / 2) * 0.034, st.z)
  }

  private cartonSlot(k: number, out: THREE.Vector3) {
    const p = SPOTS.pallet
    const layer = Math.floor(k / 4)
    const j = k % 4
    return out.set(p.x - 0.26 + (j % 2) * 0.52, 0.1125 + 0.18 + layer * 0.37, p.z - 0.22 + Math.floor(j / 2) * 0.44)
  }

  // ---------------------------------------------------------------- xe tải
  private updateTrucks(sim: FactorySim) {
    this.trucks.forEach((rig, i) => {
      const tr = sim.state.trucks[i]
      rig.group.visible = !!tr
      if (!tr) return
      const c = sim.cfg
      let x = SPOTS.truckDock.x
      if (tr.phase === 'arriving') x = SPOTS.truckFar.x + (SPOTS.truckDock.x - SPOTS.truckFar.x) * ease(Math.min(1, tr.t / c.truckArriveS))
      if (tr.phase === 'leaving') x = SPOTS.truckDock.x + (SPOTS.truckFar.x - SPOTS.truckDock.x) * ease(Math.min(1, tr.t / c.truckLeaveS))
      rig.group.position.set(x, 0, 0)
      // bánh lăn theo quãng đường (r = 0.45 m)
      tmpQ.setFromAxisAngle(tmpV.set(0, 0, 1), -(x - SPOTS.truckFar.x) / 0.45)
      WHEELS.forEach(([wx, wz], k) => rig.wheels.setMatrixAt(k, tmpM.compose(tmpP.set(wx, 0.45, wz), tmpQ, ONE)))
      rig.wheels.instanceMatrix.needsUpdate = true
      const n = Math.ceil((24 * tr.remaining) / sim.batch(tr.batchId).total)
      rig.crates.count = n
      for (let k = 0; k < n; k++) {
        const layer = Math.floor(k / 12)
        const j = k % 12
        tmpP.set(-1.9 + (j % 6) * 0.8, 1.11 + layer * 0.47, j < 6 ? -0.5 : 0.5)
        rig.crates.setMatrixAt(k, tmpM.compose(tmpP, tmpQ.identity(), ONE))
        rig.crates.setColorAt(k, tmpC.set(FRUITS[tr.fruit].color).lerp(tmpC2.set(0x6b4a2b), 0.35))
      }
      rig.crates.instanceMatrix.needsUpdate = true
      if (rig.crates.instanceColor) rig.crates.instanceColor.needsUpdate = true
    })
  }

  // ---------------------------------------------------------------- máy móc
  private drivePusher(p: Pusher, out: boolean, dtReal: number) {
    const target = out ? p.dir * p.reach : 0
    const z = (p.paddle.position.z += (target - p.paddle.position.z) * Math.min(1, dtReal * 18))
    // cần piston nối từ mặt thân xi-lanh (z = -dir·0.11) tới bàn đẩy
    const back = -p.dir * 0.11
    p.rod.scale.z = Math.max(0.01, Math.abs(z - back))
    p.rod.position.z = (z + back) / 2
  }

  private updateMachines(sim: FactorySim, dtSim: number, dtReal: number) {
    const s = sim.state
    const h = this.h
    for (const b of h.belts) b.tex.offset.x -= (b.speed * dtSim) / 1.2
    if (s.trucks.some((t) => t.phase === 'unloading' && t.remaining > 0)) h.feed.tex.offset.x -= (h.feed.speed * dtSim) / 1.2
    // cửa gạt / cửa đẩy
    this.drivePusher(h.pusher, s.items.some((it) => it.rejected && it.trashT < T_PUSH), dtReal)
    this.drivePusher(h.ejectOver, this.kick.over, dtReal)
    this.drivePusher(h.ejectRedry, this.kick.redry, dtReal)
    // thùng rác đầy dần
    h.binFill.position.y = this.binFillY - 0.02
    // đầu dán nhãn ấn xuống khi túi tới; hàm hàn ở cửa ra máy đóng gói kẹp khi túi vừa ra
    let labeling = false
    let sealing = false
    for (let i = s.bags.length - 1; i >= 0; i--) {
      const b = s.bags[i]
      if (b.s > L.labelAt + 0.2) break
      if (Math.abs(b.s - L.labelAt) < 0.12) labeling = true
      if (b.s < 0.12) sealing = true
    }
    h.labelHead.position.y += ((labeling ? 0.94 : 1.15) - h.labelHead.position.y) * Math.min(1, dtReal * 15)
    h.packHead.position.y += ((sealing ? 0.98 : 1.18) - h.packHead.position.y) * Math.min(1, dtReal * 15)
    this.updateTrays(sim, dtSim)
    // lò sấy: đèn + cửa kính + bảng
    const d = s.dryer
    const running = d.batchId !== null
    h.dryerLamp.color.set(running ? 0xff7a1a : 0x4caf50)
    h.dryerGlow.intensity = running ? 2.2 : 0
    h.dryerWindow.emissiveIntensity = running ? 0.85 : 0.05
    this.panelTimer -= dtReal
    if (this.panelTimer <= 0) {
      this.panelTimer = 0.25
      drawDryerPanel(h.dryerPanel.ctx, sim)
      h.dryerPanel.tex.needsUpdate = true
    }
  }

  /** Khay chờ sấy trên xe đẩy; khi lò bắt đầu lô, khay của lô trượt vào lò qua cửa tây. */
  private updateTrays(sim: FactorySim, dtSim: number) {
    const s = sim.state
    const d = s.dryer
    if (d.batchId !== this.prevDryer) {
      if (d.batchId !== null && d.mode === 'main') {
        const b = sim.batch(d.batchId)
        this.trolleyAnim = { t: 0, n: Math.min(16, Math.ceil(b.acceptedMassKg / 6)), fruit: b.fruit }
      }
      this.prevDryer = d.batchId
    }
    const { trays, trayFill } = this
    trays.count = trayFill.count = 0
    const put = (i: number, dx: number, fruit: FruitType) => {
      const r = SPOTS.trayRack
      tmpP.set(r.x + dx, 0.226 + Math.floor(i / 2) * 0.17, r.z + (i % 2 ? 0.48 : -0.48))
      tmpM.compose(tmpP, tmpQ.identity(), ONE)
      trays.setMatrixAt(trays.count, tmpM)
      trays.setColorAt(trays.count++, tmpC.setRGB(1, 1, 1))
      trayFill.setMatrixAt(trayFill.count, tmpM)
      trayFill.setColorAt(trayFill.count++, tmpC.set(FRUITS[fruit].color).lerp(tmpC2.set(0xfff4d6), 0.25))
    }
    const waiting = s.batches.filter((b) => b.status === 'waiting_dryer' || b.status === 'receiving')
    const kg = waiting.reduce((a, b) => a + b.acceptedMassKg, 0)
    const n = Math.min(16, Math.ceil(kg / 6))
    const fruit = waiting[0]?.fruit ?? 'mango'
    const anim = this.trolleyAnim
    let skip = 0
    if (anim) {
      anim.t += dtSim
      const q = ease(clamp01(anim.t / 2.2))
      for (let i = 0; i < anim.n; i++) put(i, 2.7 * q, anim.fruit)
      if (anim.t > 2.3) this.trolleyAnim = null
      else skip = anim.n // xe đang trống trong lúc đẩy khay vào lò
    }
    for (let i = skip; i < Math.min(16, n + skip) && i < 16; i++) put(i, 0, fruit)
    commit(trays)
    commit(trayFill)
  }
}

function drawDryerPanel(g: CanvasRenderingContext2D, sim: FactorySim) {
  const d = sim.state.dryer
  const b = d.batchId ? sim.batch(d.batchId) : null
  g.fillStyle = '#0d1117'
  g.fillRect(0, 0, 512, 256)
  g.font = '600 30px Consolas, "Cascadia Mono", monospace'
  g.fillStyle = b ? '#ff9f43' : '#4caf50'
  g.fillText(b ? (d.mode === 'main' ? 'ĐANG SẤY' : 'SẤY BỔ SUNG') : 'SẴN SÀNG', 20, 44)
  g.fillStyle = '#e6edf3'
  g.font = '28px Consolas, "Cascadia Mono", monospace'
  g.fillText(`Nhiệt độ  ${d.tempC.toFixed(1)} °C`, 20, 92)
  if (b) {
    if (d.mode === 'main') g.fillText(`Độ ẩm     ${d.moisture.toFixed(1)} %`, 20, 132)
    g.fillStyle = '#9fb3c8'
    g.font = '22px Consolas, "Cascadia Mono", monospace'
    g.fillText(b.traceCode, 20, 172)
    const p = d.t / d.duration
    g.fillStyle = '#30363d'
    g.fillRect(20, 196, 472, 26)
    g.fillStyle = '#ff9f43'
    g.fillRect(20, 196, 472 * p, 26)
    g.fillStyle = '#0d1117'
    g.font = '600 20px Consolas, monospace'
    g.fillText(`${Math.round(p * 100)}%`, 230, 216)
  }
}

function qrLikeTexture() {
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')!
  g.fillStyle = '#fff'
  g.fillRect(0, 0, 64, 64)
  g.fillStyle = '#000'
  let seed = 7
  for (let y = 4; y < 60; y += 4) {
    for (let x = 4; x < 60; x += 4) {
      seed = (seed * 16807) % 2147483647
      if (seed % 3 === 0) g.fillRect(x, y, 4, 4)
    }
  }
  for (const [x, y] of [[4, 4], [40, 4], [4, 40]]) {
    g.fillRect(x, y, 20, 20)
    g.fillStyle = '#fff'
    g.fillRect(x + 4, y + 4, 12, 12)
    g.fillStyle = '#000'
    g.fillRect(x + 7, y + 7, 6, 6)
  }
  const t = new THREE.CanvasTexture(c)
  t.magFilter = THREE.NearestFilter
  return t
}
