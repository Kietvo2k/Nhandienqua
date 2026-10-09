/** Hình học tĩnh của nhà xưởng (low-poly) + các tay cầm để cảnh động điều khiển. */
import * as THREE from 'three'
import { batchStatic } from './batch'
import { BELT_Y, CAMERAS, FEED, SPOTS, belt1, belt2, belt3, L } from './layout'

const mat = (color: number, rough = 0.8, metal = 0.1, side: THREE.Side = THREE.FrontSide) =>
  new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: metal, side })
const M = {
  steel: mat(0x8b949e, 0.45, 0.7),
  steelDS: mat(0x8b949e, 0.45, 0.7, THREE.DoubleSide),
  darkSteel: mat(0x3d434b, 0.5, 0.6),
  frame: mat(0x2f6f9f, 0.6, 0.3),
  machine: mat(0xd9dee4, 0.55, 0.2),
  machineDark: mat(0x59616b, 0.6, 0.3),
  yellow: mat(0xf2c230, 0.7, 0.1),
  bin: mat(0x2e7d32, 0.7, 0.1, THREE.DoubleSide),
  glass: new THREE.MeshStandardMaterial({ color: 0x9ec9e8, roughness: 0.1, metalness: 0, transparent: true, opacity: 0.25, depthWrite: false }),
  black: mat(0x16191d, 0.6, 0.2),
  rubber: mat(0x202428, 0.9, 0),
  wood: mat(0xa77b4f, 0.9, 0),
  crate: mat(0x3a6ea5, 0.8, 0),
  /** lỗ cửa máy: tối, không nhận sáng */
  hole: new THREE.MeshBasicMaterial({ color: 0x07090b }),
  led: new THREE.MeshBasicMaterial({ color: 0xff3b30 }),
  lamp: new THREE.MeshBasicMaterial({ color: 0xf6f8ff }),
  stripe: new THREE.MeshBasicMaterial({ color: 0xf2c230 }),
}

/** Nơi `box()` gắn mesh khi không chỉ định parent (trước đây mesh không parent bị bỏ rơi → thiếu chân, bàn, sàn dock…). */
let root: THREE.Object3D | null = null

function box(w: number, h: number, d: number, m: THREE.Material, x = 0, y = 0, z = 0, parent?: THREE.Object3D) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m)
  mesh.position.set(x, y, z)
  mesh.castShadow = mesh.receiveShadow = true
  ;(parent ?? root)?.add(mesh)
  return mesh
}

/** Lỗ cửa tối (mặt phẳng). rotY: 0 → nhìn +z, π/2 → +x, -π/2 → -x. */
function hole(w: number, h: number, x: number, y: number, z: number, rotY: number, parent: THREE.Object3D) {
  const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), M.hole)
  p.position.set(x, y, z)
  p.rotation.y = rotY
  parent.add(p)
  return p
}

/** Rèm dải cao su trước cửa máy (trải theo trục z, treo tại x). */
function curtain(width: number, top: number, bottom: number, x: number, z: number, parent: THREE.Object3D, n = 6) {
  const h = top - bottom
  const w = (width / n) * 0.86
  for (let i = 0; i < n; i++) box(0.012, h, w, M.rubber, x, bottom + h / 2, z - width / 2 + (i + 0.5) * (width / n), parent).castShadow = false
}

function canvasTexture(w: number, h: number, draw: (g: CanvasRenderingContext2D) => void) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d')!
  draw(g)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 4
  return { canvas: c, ctx: g, tex: t }
}

/** Biển tên trạm (sprite luôn quay về camera). Chỉ hiện ở khung toàn cảnh (layer 1), không che ô CCTV. */
export function makeSign(text: string, sub = '') {
  const { tex } = canvasTexture(512, 128, (g) => {
    g.fillStyle = '#1b2430'
    g.fillRect(0, 0, 512, 128)
    g.fillStyle = '#f2c230'
    g.fillRect(0, 0, 10, 128)
    g.fillStyle = '#ffffff'
    g.font = '600 46px "Segoe UI", system-ui, sans-serif'
    g.fillText(text, 30, sub ? 62 : 80)
    if (sub) {
      g.fillStyle = '#9fb3c8'
      g.font = '28px "Segoe UI", system-ui, sans-serif'
      g.fillText(sub, 30, 104)
    }
  })
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false }))
  s.scale.set(2.4, 0.6, 1)
  s.layers.set(1)
  return s
}

function beltTexture() {
  const { tex } = canvasTexture(256, 64, (g) => {
    g.fillStyle = '#23272c'
    g.fillRect(0, 0, 256, 64)
    g.strokeStyle = '#2f353c'
    g.lineWidth = 3
    for (let x = 0; x < 256; x += 16) {
      g.beginPath()
      g.moveTo(x, 0)
      g.lineTo(x, 64)
      g.stroke()
    }
  })
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  return tex
}

/** Thùng carton: nền kraft + băng keo + tem. */
function cartonTexture() {
  return canvasTexture(128, 128, (g) => {
    g.fillStyle = '#c49a6c'
    g.fillRect(0, 0, 128, 128)
    for (let i = 0; i < 300; i++) {
      g.fillStyle = `rgba(90,60,30,${Math.random() * 0.08})`
      g.fillRect(Math.random() * 128, Math.random() * 128, 2, 1)
    }
    g.fillStyle = '#d9b98f'
    g.fillRect(56, 0, 16, 128)
    g.fillStyle = '#f4f1ea'
    g.fillRect(14, 70, 32, 22)
    g.fillStyle = '#3b3b3b'
    g.fillRect(18, 75, 22, 3)
    g.fillRect(18, 82, 16, 3)
  }).tex
}

/** Nội thất lò nhìn qua cửa kính: giá khay có sản phẩm. Phát sáng cam khi lò chạy. */
function dryerInteriorTexture() {
  return canvasTexture(256, 128, (g) => {
    g.fillStyle = '#17120e'
    g.fillRect(0, 0, 256, 128)
    for (let r = 0; r < 4; r++) {
      const y = 14 + r * 29
      g.fillStyle = '#6b6f75'
      g.fillRect(6, y + 16, 244, 3)
      for (let x = 10; x < 246; x += 6) {
        g.fillStyle = `rgb(${150 + Math.random() * 60},${80 + Math.random() * 40},${30 + Math.random() * 20})`
        g.fillRect(x, y + 10 + Math.random() * 3, 4, 5)
      }
    }
  }).tex
}

export interface Belt {
  tex: THREE.Texture
  /** m/s; kết cấu luôn chạy theo trục +x cục bộ (đã quay hướng a → b) */
  speed: number
}

function makeBelt(scene: THREE.Object3D, a: THREE.Vector3, b: THREE.Vector3, speed: number, width = 0.9): Belt {
  const len = a.distanceTo(b)
  const mid = a.clone().add(b).multiplyScalar(0.5)
  const g = new THREE.Group()
  g.position.set(mid.x, 0, mid.z)
  g.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x)
  const tex = beltTexture()
  tex.repeat.set(len / 1.2, 1)
  const surf = new THREE.Mesh(new THREE.BoxGeometry(len, 0.05, width), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }))
  surf.position.y = BELT_Y - 0.03
  surf.receiveShadow = true
  g.add(surf)
  for (const side of [-1, 1]) box(len, 0.12, 0.06, M.frame, 0, BELT_Y - 0.02, side * (width / 2 + 0.03), g)
  for (let x = -len / 2 + 0.3; x <= len / 2; x += 2) {
    for (const side of [-1, 1]) box(0.06, BELT_Y - 0.06, 0.06, M.darkSteel, x, (BELT_Y - 0.06) / 2, side * (width / 2 - 0.05), g)
  }
  for (const end of [-1, 1]) {
    const roller = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, width, 14), M.steel)
    roller.rotation.x = Math.PI / 2
    roller.position.set((end * len) / 2, BELT_Y - 0.03, 0)
    g.add(roller)
  }
  scene.add(g)
  return { tex, speed }
}

/** Băng cấp liệu nghiêng có vách: mặt trên băng trùng đoạn FEED.foot → FEED.head. */
function makeFeeder(scene: THREE.Object3D): Belt {
  const { foot: a, head: b } = FEED
  const len = a.distanceTo(b)
  const g = new THREE.Group()
  g.position.copy(a).add(b).multiplyScalar(0.5)
  g.rotation.z = Math.atan2(b.y - a.y, b.x - a.x)
  const tex = beltTexture()
  tex.repeat.set(len / 1.2, 1)
  const surf = new THREE.Mesh(new THREE.BoxGeometry(len, 0.04, 0.55), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }))
  surf.position.y = -0.02
  surf.receiveShadow = true
  g.add(surf)
  for (const side of [-1, 1]) box(len, 0.16, 0.03, M.frame, 0, 0.05, side * 0.29, g)
  for (const end of [-1, 1]) {
    const roller = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.55, 12), M.steel)
    roller.rotation.x = Math.PI / 2
    roller.position.set((end * len) / 2, -0.03, 0)
    g.add(roller)
  }
  scene.add(g)
  // chân đỡ phía đầu (đứng trên sàn) + khối kê chân băng trên sàn dock
  const p = a.clone().lerp(b, 0.78)
  for (const side of [-1, 1]) box(0.06, p.y - 0.06, 0.06, M.darkSteel, p.x, (p.y - 0.06) / 2, side * 0.24)
  box(0.45, a.y - 0.06 - 0.85, 0.6, M.darkSteel, a.x + 0.15, (a.y - 0.06 + 0.85) / 2, 0)
  return { tex, speed: L.beltSpeed }
}

export interface Pusher {
  group: THREE.Group
  rod: THREE.Mesh
  paddle: THREE.Mesh
  /** hướng đẩy theo trục z (+1/-1) và tầm với (m) */
  dir: number
  reach: number
}

/** Xi-lanh đẩy đặt cạnh băng tại (x, z) — đẩy theo hướng dir trên trục z. */
function makePusher(scene: THREE.Object3D, x: number, z: number, dir: 1 | -1, reach: number, w = 0.42): Pusher {
  const g = new THREE.Group()
  g.userData.dynamic = true
  const y = BELT_Y + 0.13
  g.position.set(x, y, z)
  // thân xi-lanh đứng yên → để ngoài nhóm động cho được gộp tĩnh
  box(w + 0.08, 0.24, 0.32, M.machineDark, x, y, z - dir * 0.27)
  const rod = box(0.05, 0.05, 1, M.steel, 0, 0, 0, g)
  const paddle = box(w, 0.17, 0.04, M.yellow, 0, 0, 0, g)
  scene.add(g)
  box(0.08, y - 0.12, 0.08, M.darkSteel, x, (y - 0.12) / 2, z - dir * 0.27)
  return { group: g, rod, paddle, dir, reach }
}

/** Máng nghiêng từ mép băng xuống miệng thùng (theo trục z). */
function makeChute(x: number, z0: number, y0: number, z1: number, y1: number, w: number) {
  const len = Math.hypot(z1 - z0, y1 - y0)
  const g = new THREE.Group()
  g.position.set(x, (y0 + y1) / 2, (z0 + z1) / 2)
  // quay quanh x: z cục bộ → (0, -sinθ, cosθ); cần dy/dz = (y1-y0)/(z1-z0)
  g.rotation.x = -Math.atan((y1 - y0) / (z1 - z0))
  box(w, 0.025, len, M.steelDS, 0, -0.0125, 0, g)
  for (const side of [-1, 1]) box(0.025, 0.09, len, M.steel, (side * (w + 0.025)) / 2, 0.035, 0, g)
  root!.add(g)
  return g
}

function cctv(scene: THREE.Object3D, pos: THREE.Vector3Tuple, look: THREE.Vector3Tuple) {
  const g = new THREE.Group()
  g.position.set(...pos)
  g.lookAt(new THREE.Vector3(...look))
  box(0.14, 0.12, 0.26, M.machine, 0, 0, 0.05, g)
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.04, 16), M.black)
  lens.rotation.x = Math.PI / 2
  lens.position.z = 0.19
  g.add(lens)
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.015, 8, 8), M.led)
  led.position.set(0.05, 0.04, 0.18)
  g.add(led)
  // layer 1: chỉ khung toàn cảnh thấy; camera ô không thấy chính vỏ của nó
  g.traverse((o) => o.layers.set(1))
  scene.add(g)
}

function gantry(scene: THREE.Object3D, x: number, z: number, h: number, label: string) {
  for (const side of [-1, 1]) box(0.08, h, 0.08, M.yellow, x, h / 2, z + side * 0.75)
  box(0.1, 0.1, 1.6, M.yellow, x, h, z)
  // đèn LED chiếu sáng khu chụp
  const light = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.04, 1.2), M.lamp)
  light.position.set(x - 0.45, h - 0.05, z)
  scene.add(light)
  box(0.6, 0.06, 0.06, M.yellow, x - 0.3, h - 0.02, z + 0.62)
  box(0.6, 0.06, 0.06, M.yellow, x - 0.3, h - 0.02, z - 0.62)
  const s = makeSign(label)
  s.position.set(x, h + 0.55, z)
  scene.add(s)
}

export interface FactoryHandles {
  belts: Belt[]
  /** băng cấp liệu nghiêng — chỉ chạy khi xe đang dỡ hàng */
  feed: Belt
  pusher: Pusher
  ejectOver: Pusher
  ejectRedry: Pusher
  dryerPanel: { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; tex: THREE.Texture }
  dryerLamp: THREE.MeshBasicMaterial
  dryerGlow: THREE.PointLight
  dryerWindow: THREE.MeshStandardMaterial
  /** đống quả trong phễu (hiện khi đang dỡ hàng) */
  hopperPile: THREE.Mesh
  labelHead: THREE.Object3D
  packHead: THREE.Object3D
  binFill: THREE.Mesh
  qcArm: { carriage: THREE.Object3D; rod: THREE.Mesh; head: THREE.Object3D }
  cartonMat: THREE.Material
}

export function buildFactory(scene: THREE.Scene): FactoryHandles {
  root = scene
  // ---------------------------------------------------------------- nền & ánh sáng
  scene.background = new THREE.Color(0xdfe4ea)
  scene.fog = new THREE.Fog(0xdfe4ea, 60, 120)
  const floorTex = canvasTexture(512, 512, (g) => {
    g.fillStyle = '#b9bec4'
    g.fillRect(0, 0, 512, 512)
    for (let i = 0; i < 2500; i++) {
      g.fillStyle = `rgba(0,0,0,${Math.random() * 0.05})`
      g.fillRect(Math.random() * 512, Math.random() * 512, 2, 2)
    }
    g.strokeStyle = 'rgba(0,0,0,0.12)'
    g.strokeRect(0, 0, 512, 512)
  }).tex
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping
  floorTex.repeat.set(16, 8)
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(96, 48), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.95 }))
  floor.rotation.x = -Math.PI / 2
  floor.position.set(-8, 0, -4)
  floor.receiveShadow = true
  scene.add(floor)
  // vạch lối đi an toàn
  for (const z of [3.4, -3.2, -12.2]) {
    const line = new THREE.Mesh(new THREE.PlaneGeometry(44, 0.12), M.stripe)
    line.rotation.x = -Math.PI / 2
    line.position.set(5, 0.005, z)
    scene.add(line)
  }

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f96, 1.1))
  const sun = new THREE.DirectionalLight(0xffffff, 1.6)
  sun.position.set(-12, 26, 14)
  sun.castShadow = true
  sun.shadow.mapSize.set(2048, 2048)
  Object.assign(sun.shadow.camera, { left: -32, right: 32, top: 18, bottom: -18, near: 1, far: 70 })
  sun.shadow.bias = -0.0005
  sun.shadow.normalBias = 0.02
  scene.add(sun)

  // ---------------------------------------------------------------- băng chuyền
  const belts = [
    makeBelt(scene, belt1(-0.3), belt1(L.belt1Length + 0.3), L.beltSpeed),
    makeBelt(scene, belt2(-0.2), belt2(L.packAt + 0.4), L.belt2Speed, 0.8),
    makeBelt(scene, belt3(-0.3), belt3(L.belt3Length + 0.2), L.belt3Speed, 0.7),
  ]

  // ---------------------------------------------------------------- sàn dock + băng cấp liệu + phễu
  // sàn dock nằm ngay sau đuôi xe (đuôi xe ở x = -13.1 khi đỗ)
  box(1.5, 0.85, 3.2, M.machineDark, -12.35, 0.425, 0)
  box(0.05, 0.08, 3.2, M.yellow, -13.1, 0.81, 0)
  for (const z of [-1.0, 1.0]) box(0.12, 0.25, 0.3, M.rubber, -13.16, 0.55, z)
  // két nhựa rỗng xếp chồng trên sàn dock
  box(0.55, 0.32, 0.4, M.crate, -11.95, 1.01, 1.15)
  box(0.55, 0.32, 0.4, M.crate, -11.95, 1.33, 1.15)
  const feed = makeFeeder(scene)

  // phễu: miệng xả ngay trên đầu băng 1 (cạnh dưới y = 1.15), miệng trên y = 2.15
  const hopper = new THREE.Mesh(new THREE.CylinderGeometry(0.95, 0.45, 1.0, 4, 1, true), M.steelDS)
  hopper.rotation.y = Math.PI / 4
  hopper.position.set(SPOTS.hopper.x, 1.65, 0)
  hopper.castShadow = true
  scene.add(hopper)
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(0.97, 0.97, 0.05, 4, 1, true), M.darkSteel)
  rim.rotation.y = Math.PI / 4
  rim.position.set(SPOTS.hopper.x, 2.15, 0)
  scene.add(rim)
  for (const dx of [-0.64, 0.64]) for (const dz of [-0.64, 0.64]) box(0.06, 2.15, 0.06, M.darkSteel, SPOTS.hopper.x + dx, 1.075, dz)
  const hopperPile = new THREE.Mesh(new THREE.ConeGeometry(0.5, 0.22, 4), new THREE.MeshStandardMaterial({ roughness: 0.7 }))
  hopperPile.rotation.y = Math.PI / 4
  hopperPile.position.set(SPOTS.hopper.x, 1.72, 0)
  hopperPile.userData.dynamic = true
  hopperPile.visible = false
  scene.add(hopperPile)
  const dockSign = makeSign('Nhận hàng', 'Dock xe tải')
  dockSign.position.set(-14.5, 3.4, 2.6)
  scene.add(dockSign)

  // ---------------------------------------------------------------- trạm AI phân loại + cửa gạt + máng + thùng rác
  gantry(scene, SPOTS.sortCam.x, 0, 2.5, 'AI phân loại')
  const pusher = makePusher(scene, SPOTS.gate.x, 0.55, -1, 0.82)
  makeChute(SPOTS.gate.x, -0.5, 0.8, -1.55, 0.64, 0.55)
  for (const dx of [-0.26, 0.26]) box(0.04, 0.62, 0.04, M.darkSteel, SPOTS.gate.x + dx, 0.31, -1.5)
  const bin = new THREE.Group()
  bin.position.copy(SPOTS.trash)
  const binBody = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.47, 0.62, 24, 1, true), M.bin)
  binBody.position.y = 0.31
  binBody.castShadow = true
  bin.add(binBody)
  const binRim = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.022, 6, 24), M.bin)
  binRim.rotation.x = Math.PI / 2
  binRim.position.y = 0.62
  bin.add(binRim)
  const binBottom = new THREE.Mesh(new THREE.CircleGeometry(0.47, 24), M.bin)
  binBottom.rotation.x = -Math.PI / 2
  binBottom.position.y = 0.01
  bin.add(binBottom)
  scene.add(bin)
  const binFill = new THREE.Mesh(new THREE.CylinderGeometry(0.46, 0.46, 0.04, 20), mat(0x4d3b27, 0.95, 0))
  binFill.position.set(SPOTS.trash.x, 0.03, SPOTS.trash.z)
  binFill.userData.dynamic = true
  scene.add(binFill)
  const trashSign = makeSign('Thùng rác', 'Quả hư / bệnh')
  trashSign.position.set(SPOTS.trash.x, 2.0, SPOTS.trash.z - 0.4)
  scene.add(trashSign)

  // ---------------------------------------------------------------- sơ chế (mặt trước x = 11.6, có cửa + rèm)
  const prep = new THREE.Group()
  prep.position.copy(SPOTS.prep)
  box(2.6, 1.7, 1.6, M.machine, 0, 0.85, 0, prep)
  box(2.62, 0.25, 1.62, M.frame, 0, 1.6, 0, prep)
  box(0.9, 0.5, 0.05, M.black, -0.4, 1.1, 0.81, prep)
  hole(1.0, 0.44, -1.302, 1.04, 0, -Math.PI / 2, prep)
  curtain(0.96, 1.26, 0.9, -1.33, 0, prep)
  hole(0.8, 0.5, 1.302, 0.6, 0.2, Math.PI / 2, prep)
  scene.add(prep)
  const prepSign = makeSign('Sơ chế', 'Rửa · gọt · cắt lát')
  prepSign.position.set(SPOTS.prep.x, 2.6, 0)
  scene.add(prepSign)

  // ---------------------------------------------------------------- xe đẩy khay chờ sấy (2 cột × 8 tầng)
  const trolley = new THREE.Group()
  trolley.position.copy(SPOTS.trayRack)
  box(0.72, 0.06, 2.0, M.darkSteel, 0, 0.14, 0, trolley)
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      box(0.04, 1.55, 0.04, M.steel, sx * 0.34, 0.92, sz * 0.98, trolley)
      const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.04, 12), M.rubber)
      wheel.rotation.x = Math.PI / 2
      wheel.position.set(sx * 0.28, 0.05, sz * 0.9)
      trolley.add(wheel)
    }
  }
  for (let r = 0; r < 8; r++) {
    for (const z of [-0.94, -0.03, 0.03, 0.94]) box(0.66, 0.012, 0.02, M.steel, 0, 0.22 + r * 0.17, z, trolley).castShadow = false
  }
  scene.add(trolley)

  // ---------------------------------------------------------------- lò sấy (đường hầm)
  const dryer = new THREE.Group()
  dryer.position.copy(SPOTS.dryer)
  box(8, 3, 4, M.machine, 0, 1.5, 0, dryer)
  box(8.05, 0.3, 4.05, M.frame, 0, 3.05, 0, dryer)
  for (const x of [-3.3, 0.3, 3.3]) box(0.04, 2.6, 0.02, M.machineDark, x, 1.4, 2.01, dryer)
  // cửa tây: xe khay đẩy vào đây
  hole(2.1, 1.75, -4.002, 0.95, 0, -Math.PI / 2, dryer)
  box(0.06, 1.9, 0.08, M.machineDark, -4.03, 0.95, 1.1, dryer)
  box(0.06, 1.9, 0.08, M.machineDark, -4.03, 0.95, -1.1, dryer)
  box(0.06, 0.08, 2.28, M.machineDark, -4.03, 1.88, 0, dryer)
  // cửa kính nhìn vào giá khay bên trong
  const dryerWindow = new THREE.MeshStandardMaterial({ map: dryerInteriorTexture(), emissive: 0xffa860, roughness: 0.9 })
  dryerWindow.emissiveMap = dryerWindow.map
  dryerWindow.emissiveIntensity = 0
  const winIn = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 1.1), dryerWindow)
  winIn.position.set(-1.5, 1.6, 2.004)
  dryer.add(winIn)
  const win = box(2.2, 1.1, 0.03, M.glass, -1.5, 1.6, 2.025, dryer)
  win.castShadow = win.receiveShadow = false
  box(2.3, 0.06, 0.05, M.machineDark, -1.5, 2.18, 2.02, dryer)
  box(2.3, 0.06, 0.05, M.machineDark, -1.5, 1.02, 2.02, dryer)
  const panel = canvasTexture(512, 256, () => {})
  const panelMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.6, 0.8), new THREE.MeshBasicMaterial({ map: panel.tex }))
  panelMesh.position.set(1.8, 1.7, 2.04)
  dryer.add(panelMesh)
  const dryerLamp = new THREE.MeshBasicMaterial({ color: 0x4caf50 })
  const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.35, 16), dryerLamp)
  lamp.position.set(3.6, 3.4, 1.6)
  lamp.userData.dynamic = true // màu đổi theo trạng thái lò → không nướng màu
  dryer.add(lamp)
  // quạt / ống gió trên nóc
  for (const x of [-2.5, 0, 2.5]) {
    const duct = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.8, 18), M.steel)
    duct.position.set(x, 3.6, -0.8)
    duct.castShadow = true
    dryer.add(duct)
  }
  const dryerGlow = new THREE.PointLight(0xff8a3d, 0, 6)
  dryerGlow.position.set(-1.5, 1.6, 2.6)
  dryer.add(dryerGlow)
  scene.add(dryer)
  const dryerSign = makeSign('Lò sấy', 'Sấy bơm nhiệt / khí nóng')
  dryerSign.position.set(SPOTS.dryer.x, 4.6, 2)
  scene.add(dryerSign)

  // ---------------------------------------------------------------- ống chuyển kín lò → chụp làm nguội ở đầu băng 2
  box(0.8, 0.5, 3.75, M.machine, 23.4, 2.35, -3.85)
  for (const z of [-2.6, -4.6]) for (const dx of [-0.36, 0.36]) box(0.06, 2.1, 0.06, M.darkSteel, 23.4 + dx, 1.05, z)
  box(0.8, 0.97, 0.8, M.machine, 23.4, 2.115, -5.75)
  const hood = new THREE.Group()
  hood.position.set(23.55, 0, -6)
  box(2.1, 0.06, 1.34, M.machineDark, 0, 1.6, 0, hood)
  for (const side of [-1, 1]) box(2.1, 0.78, 0.04, M.machine, 0, 1.2, side * 0.65, hood)
  box(0.04, 0.78, 1.3, M.machine, 1.05, 1.2, 0, hood)
  box(0.04, 0.3, 1.3, M.machine, -1.05, 1.43, 0, hood)
  hole(1.26, 0.74, 1.02, 1.2, 0, -Math.PI / 2, hood)
  curtain(1.2, 1.28, 0.93, -1.08, 0, hood, 7)
  scene.add(hood)
  const coolSign = makeSign('Làm nguội', 'Ra lò')
  coolSign.position.set(23.6, 2.6, -7.2)
  scene.add(coolSign)

  // ---------------------------------------------------------------- AI mức sấy + cửa đẩy (quá khô → thùng, chưa đạt → xe khay sấy lại)
  gantry(scene, SPOTS.dryCam.x, -6, 2.3, 'AI mức sấy')
  const ejectOver = makePusher(scene, SPOTS.trash2.x, -6.55, 1, 0.82, 0.32)
  makeChute(SPOTS.trash2.x, -5.5, 0.8, -5.3, 0.62, 0.36)
  const trash2 = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.33, 0.6, 18, 1, true), M.bin)
  trash2.position.set(SPOTS.trash2.x, 0.3, SPOTS.trash2.z)
  trash2.castShadow = true
  scene.add(trash2)
  const trash2Bottom = new THREE.Mesh(new THREE.CircleGeometry(0.33, 18), M.bin)
  trash2Bottom.rotation.x = -Math.PI / 2
  trash2Bottom.position.set(SPOTS.trash2.x, 0.01, SPOTS.trash2.z)
  scene.add(trash2Bottom)
  const ejectRedry = makePusher(scene, SPOTS.redryCart.x, -5.45, -1, 0.82, 0.32)
  makeChute(SPOTS.redryCart.x, -6.5, 0.8, -6.82, 0.64, 0.36)
  // xe khay sấy lại: thùng inox hở nắp trên xe đẩy
  const cart = new THREE.Group()
  cart.position.copy(SPOTS.redryCart)
  box(0.8, 0.06, 0.62, M.darkSteel, 0, 0.2, 0, cart)
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) box(0.04, 0.2, 0.04, M.darkSteel, sx * 0.36, 0.1, sz * 0.27, cart)
  box(0.7, 0.02, 0.5, M.steelDS, 0, 0.24, 0, cart)
  for (const sz of [-1, 1]) box(0.7, 0.36, 0.02, M.steelDS, 0, 0.42, sz * 0.25, cart)
  for (const sx of [-1, 1]) box(0.02, 0.36, 0.5, M.steelDS, sx * 0.35, 0.42, 0, cart)
  scene.add(cart)
  const redrySign = makeSign('Sấy lại', 'Chưa đạt độ ẩm')
  redrySign.position.set(SPOTS.redryCart.x - 0.6, 1.9, -7.6)
  scene.add(redrySign)

  // ---------------------------------------------------------------- trạm QC: tay gắp → bàn mẫu → băng mẫu → cửa chuyển vào phòng kiểm nghiệm
  box(1.0, 0.82, 0.7, M.steel, SPOTS.qcTable.x, 0.41, SPOTS.qcTable.z)
  box(0.42, 0.04, 1.75, M.darkSteel, SPOTS.qc.x, 0.8, -8.2)
  for (const z of [-7.6, -8.8]) for (const dx of [-0.17, 0.17]) box(0.04, 0.78, 0.04, M.darkSteel, SPOTS.qc.x + dx, 0.39, z)
  for (const side of [-1, 1]) box(0.02, 0.06, 1.75, M.yellow, SPOTS.qc.x + side * 0.22, 0.85, -8.2)
  box(0.6, 0.5, 0.4, M.machineDark, SPOTS.qc.x, 1.0, -9.2)
  hole(0.46, 0.3, SPOTS.qc.x, 0.97, -8.998, 0, scene)
  // khung tay gắp (dầm dọc trục z phía trên băng + bàn)
  const armX = SPOTS.qc.x
  for (const z of [-5.3, -7.6]) {
    box(0.07, 1.9, 0.07, M.yellow, armX + 0.6, 0.95, z)
    box(0.62, 0.07, 0.07, M.yellow, armX + 0.3, 1.86, z)
  }
  box(0.09, 0.09, 2.38, M.yellow, armX, 1.86, -6.45)
  const carriage = new THREE.Group()
  carriage.userData.dynamic = true
  carriage.position.set(armX, 1.76, -6.0)
  box(0.18, 0.12, 0.2, M.machineDark, 0, 0, 0, carriage)
  const armRod = box(0.03, 1, 0.03, M.steel, 0, -0.3, 0, carriage)
  const head = new THREE.Group()
  const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.065, 0.04, 14), M.rubber)
  head.add(cup)
  head.position.y = -0.4
  carriage.add(head)
  scene.add(carriage)
  const lab = new THREE.Group()
  lab.position.copy(SPOTS.lab)
  box(3.2, 0.1, 2.2, M.machineDark, 0, 0.05, 0, lab)
  box(3.2, 2.4, 2.2, M.glass, 0, 1.25, 0, lab).castShadow = false
  box(2.4, 0.9, 0.7, M.machine, 0, 0.45, -0.55, lab)
  box(0.5, 0.35, 0.4, M.black, -0.6, 1.07, -0.55, lab) // máy đo độ ẩm
  box(0.4, 0.45, 0.4, M.machineDark, 0.5, 1.12, -0.55, lab) // máy đo aw
  scene.add(lab)
  const labSign = makeSign('Kiểm nghiệm', 'Độ ẩm · aw · vi sinh')
  labSign.position.set(SPOTS.lab.x, 3.1, SPOTS.lab.z)
  scene.add(labSign)

  // ---------------------------------------------------------------- máy đóng gói (VFFS): cửa vào phía băng 2, cửa ra + hàm hàn phía băng 3
  const pack = new THREE.Group()
  pack.position.set(SPOTS.pack.x - 0.4, 0, -6)
  box(1.6, 2.8, 1.6, M.machine, 0, 1.4, 0, pack)
  box(1.0, 0.8, 1.0, M.steel, 0, 3.2, 0, pack) // phễu cân
  box(0.6, 0.5, 0.04, M.black, 0.3, 1.9, 0.81, pack)
  hole(0.72, 0.36, 0.802, 1.02, 0, Math.PI / 2, pack)
  curtain(0.7, 1.2, 0.92, 0.83, 0, pack)
  hole(0.62, 0.3, -0.802, 0.99, 0, -Math.PI / 2, pack)
  scene.add(pack)
  const packHead = box(0.08, 0.07, 0.66, M.steel, SPOTS.pack.x - 0.4 - 0.86, 1.18, -6)
  packHead.userData.dynamic = true
  const packSign = makeSign('Đóng gói', 'Cân · chiết · hàn túi')
  packSign.position.set(SPOTS.pack.x - 0.4, 4.3, -6)
  scene.add(packSign)

  // ---------------------------------------------------------------- máy dán nhãn QR
  const labeler = new THREE.Group()
  labeler.position.set(SPOTS.label.x, 0, -6)
  for (const side of [-1, 1]) box(0.08, 1.55, 0.08, M.darkSteel, side * 0.5, 0.775, -0.55, labeler)
  box(1.1, 0.4, 0.5, M.machine, 0, 1.75, -0.45, labeler)
  box(0.14, 0.08, 0.36, M.machineDark, 0, 1.6, -0.08, labeler)
  scene.add(labeler)
  const labelHead = new THREE.Group()
  labelHead.userData.dynamic = true
  labelHead.position.set(SPOTS.label.x, 1.15, -5.94)
  box(0.18, 0.06, 0.16, M.machineDark, 0, 0, 0, labelHead)
  box(0.03, 0.6, 0.03, M.steel, 0, 0.32, 0, labelHead)
  scene.add(labelHead)
  const labelSign = makeSign('Dán nhãn QR', 'Mã truy xuất lô')
  labelSign.position.set(SPOTS.label.x, 2.8, -6)
  scene.add(labelSign)

  // ---------------------------------------------------------------- trạm vô thùng cuối băng 3 + pallet
  const cartonMat = new THREE.MeshStandardMaterial({ map: cartonTexture(), roughness: 0.95, side: THREE.DoubleSide })
  const st = SPOTS.cartonStation
  box(0.62, 0.26, 0.56, M.darkSteel, st.x, 0.13, st.z)
  box(0.5, 0.01, 0.42, cartonMat, st.x, 0.265, st.z)
  for (const sz of [-1, 1]) box(0.5, 0.36, 0.008, cartonMat, st.x, 0.44, st.z + sz * 0.21)
  for (const sx of [-1, 1]) box(0.008, 0.36, 0.42, cartonMat, st.x + sx * 0.25, 0.44, st.z)
  for (const sz of [-1, 1]) {
    const flap = box(0.5, 0.008, 0.2, cartonMat, st.x, 0.66, st.z + sz * 0.29)
    flap.rotation.x = sz * 0.55
  }
  const pallet = new THREE.Group()
  pallet.position.copy(SPOTS.pallet)
  for (const z of [-0.42, 0, 0.42]) box(1.2, 0.09, 0.12, M.wood, 0, 0.045, z, pallet)
  for (let i = 0; i < 7; i++) box(0.14, 0.025, 1.0, M.wood, -0.54 + i * 0.18, 0.1, 0, pallet)
  scene.add(pallet)
  const palletSign = makeSign('Thành phẩm', 'Thùng 20 túi')
  palletSign.position.set(SPOTS.pallet.x, 2.4, -6)
  scene.add(palletSign)

  // ---------------------------------------------------------------- camera CCTV
  for (const c of CAMERAS) cctv(scene, c.pos, c.look)

  // gộp mesh tĩnh theo vật liệu → giảm mạnh draw call cho 9 khung nhìn
  batchStatic(scene)
  root = null

  return {
    belts, feed, pusher, ejectOver, ejectRedry,
    dryerPanel: panel, dryerLamp, dryerGlow, dryerWindow, hopperPile,
    labelHead, packHead, binFill,
    qcArm: { carriage, rod: armRod, head },
    cartonMat,
  }
}

export { M as MATERIALS, box as makeBox }
