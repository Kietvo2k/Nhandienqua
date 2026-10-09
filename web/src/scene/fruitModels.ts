/**
 * Mô hình quả, phần sản phẩm sau sấy và túi (nhẹ, dùng cho InstancedMesh).
 * Thân quả là hình "quét" dọc trục x (bán kính theo t ∈ [0,1] và góc a) + texture canvas;
 * quả hư dùng texture riêng có đốm thối nâu/đen để nhìn ra được trên camera.
 */
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { FruitType } from '../sim/params'

/** Thân quả chiếm u ∈ [0, BODY_U]; dải u ≥ 0.94 dành cho cuống (v < 0.5) và lá/tai (v ≥ 0.55). */
const BODY_U = 0.92
const STEM_U = 0.97

const pow = (x: number, p: number) => Math.pow(Math.max(0, x), p)
const sin = Math.sin
const PI = Math.PI

function rng(seed: number) {
  let s = seed % 2147483647 || 1
  return () => (s = (s * 16807) % 2147483647) / 2147483647
}

interface SweepOpts {
  len: number
  r: (t: number, a: number, i: number, j: number) => number
  /** độ cong (m) theo trục z ở giữa thân */
  bend?: number
  /** nén theo trục y (quả nằm nghiêng) */
  flatY?: number
  rings?: number
  segs?: number
}

/** Quét một vòng tiết diện dọc đường cong phẳng trong mặt xz; hai đầu bán kính 0 khép kín. */
function sweep(o: SweepOpts) {
  const R = o.rings ?? 28
  const S = o.segs ?? 18
  const bend = o.bend ?? 0
  const flat = o.flatY ?? 1
  const n = (R + 1) * (S + 1)
  const pos = new Float32Array(n * 3)
  const uv = new Float32Array(n * 2)
  const idx: number[] = []
  let k = 0
  for (let i = 0; i <= R; i++) {
    const t = i / R
    const u = 2 * t - 1
    const x = (t - 0.5) * o.len
    const cz = bend * (1 - u * u)
    const slope = (-4 * bend * u) / o.len
    const h = Math.hypot(1, slope)
    const nx = -slope / h
    const nz = 1 / h
    for (let j = 0; j <= S; j++) {
      const a = (j / S) * PI * 2
      const r = Math.max(0, o.r(t, a, i, j % S))
      const ca = Math.cos(a)
      pos[k * 3] = x + nx * r * ca
      pos[k * 3 + 1] = r * sin(a) * flat
      pos[k * 3 + 2] = cz + nz * r * ca
      uv[k * 2] = t * BODY_U
      uv[k * 2 + 1] = j / S
      k++
    }
  }
  for (let i = 0; i < R; i++) {
    for (let j = 0; j < S; j++) {
      const a = i * (S + 1) + j
      const b = a + S + 1
      idx.push(a, b, a + 1, b, b + 1, a + 1)
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  g.setIndex(idx)
  g.computeVertexNormals()
  // vá pháp tuyến ở đường nối (j = 0 ≡ j = S) và hai đầu mút
  const nr = g.attributes.normal as THREE.BufferAttribute
  const v = new THREE.Vector3()
  for (let i = 0; i <= R; i++) {
    const a = i * (S + 1)
    const b = a + S
    v.set(nr.getX(a) + nr.getX(b), nr.getY(a) + nr.getY(b), nr.getZ(a) + nr.getZ(b)).normalize()
    nr.setXYZ(a, v.x, v.y, v.z)
    nr.setXYZ(b, v.x, v.y, v.z)
  }
  for (let j = 0; j <= S; j++) {
    nr.setXYZ(j, -1, 0, 0)
    nr.setXYZ(R * (S + 1) + j, 1, 0, 0)
  }
  return g
}

/** Gán UV cố định cho chi tiết phụ (cuống, tai): u cố định, v chạy v0 → v1 theo trục y của chi tiết. */
function withUV(g: THREE.BufferGeometry, u: number, v0: number, v1 = v0) {
  const p = g.attributes.position
  g.computeBoundingBox()
  const bb = g.boundingBox!
  const hy = bb.max.y - bb.min.y || 1
  const uv = new Float32Array(p.count * 2)
  for (let i = 0; i < p.count; i++) {
    uv[i * 2] = u
    uv[i * 2 + 1] = v0 + ((v1 - v0) * (p.getY(i) - bb.min.y)) / hy
  }
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  return g
}

const tmpM = new THREE.Matrix4()
const X = new THREE.Vector3()
const Y = new THREE.Vector3()
const Z = new THREE.Vector3()

/** Đặt chi tiết: trục y cục bộ → dir, trục z cục bộ ≈ face (mặt phẳng dẹt), gốc → at. */
function place(g: THREE.BufferGeometry, at: THREE.Vector3, dir: THREE.Vector3, face = new THREE.Vector3(0, 0, 1)) {
  Y.copy(dir).normalize()
  Z.copy(face).addScaledVector(Y, -face.dot(Y))
  if (Z.lengthSq() < 1e-6) Z.set(Y.y, -Y.x, 0)
  Z.normalize()
  X.crossVectors(Y, Z)
  tmpM.makeBasis(X, Y, Z).setPosition(at)
  return g.applyMatrix4(tmpM)
}

function stem(r0: number, r1: number, len: number, at: THREE.Vector3, dir: THREE.Vector3) {
  const g = withUV(new THREE.CylinderGeometry(r1, r0, len, 6).translate(0, len / 2, 0), STEM_U, 0.12, 0.38)
  return place(g, at, dir)
}

/** Gộp các phần, đặt đáy quả ở y = 0 và tâm x/z ở gốc. */
function finish(parts: THREE.BufferGeometry[]) {
  const g = parts.length === 1 ? parts[0] : mergeGeometries(parts)!
  g.computeBoundingBox()
  const bb = g.boundingBox!
  g.translate(-(bb.min.x + bb.max.x) / 2, -bb.min.y, -(bb.min.z + bb.max.z) / 2)
  g.computeBoundingBox()
  g.computeBoundingSphere()
  return g
}

// ------------------------------------------------------------------ hình học từng loại quả

function mangoGeometry() {
  const len = 0.2
  // quả thận: đầu cuống (t = 1) đầy vai, đầu mỏ (t = 0) thon; cong nhẹ theo z
  const body = sweep({
    len, bend: 0.02, flatY: 0.8, rings: 26, segs: 20,
    r: (t, a) => 0.078 * pow(sin(PI * t), 0.5) * (0.88 + 0.18 * t) * (1 + 0.04 * Math.cos(a)),
  })
  return finish([body, stem(0.007, 0.005, 0.024, new THREE.Vector3(len / 2 - 0.004, 0.004, 0.0), new THREE.Vector3(1, 0.35, 0))])
}

function bananaGeometry() {
  const len = 0.26
  const STEM_T = 0.84
  const body = sweep({
    len, bend: 0.05, rings: 36, segs: 15,
    r: (t, a) => {
      if (t > 0.985) return 0
      if (t >= STEM_T) return 0.0075
      const tau = t / STEM_T
      let r = 0.031 * pow(sin(PI * tau), 0.45)
      if (tau > 0.5) r = Math.max(r, 0.0075)
      return r * (1 + 0.07 * Math.cos(5 * a))
    },
  })
  return finish([body])
}

function dragonFruitGeometry() {
  const len = 0.19
  const R0 = 0.072
  const flat = 0.9
  const rBody = (t: number) => R0 * pow(sin(PI * t), 0.6)
  const parts: THREE.BufferGeometry[] = [sweep({ len, flatY: flat, rings: 22, segs: 18, r: (t) => rBody(t) })]
  // tai (vảy) xanh nhọn rải theo xoắn ốc, hướng về phía đỉnh (+x) và chĩa ra ngoài
  const golden = PI * (3 - Math.sqrt(5))
  const N = 22
  for (let i = 0; i < N; i++) {
    const t = 0.14 + (0.74 * i) / (N - 1)
    const a = i * golden * 2.1
    if (sin(a) < -0.45) continue // không đặt tai ở mặt đáy (quả nằm trên băng)
    const r = rBody(t)
    const nrm = new THREE.Vector3(0, sin(a), Math.cos(a))
    const at = new THREE.Vector3((t - 0.5) * len, r * sin(a) * flat, r * Math.cos(a)).addScaledVector(nrm, -0.004)
    const dir = new THREE.Vector3(1, 0, 0).multiplyScalar(0.8).addScaledVector(nrm, 0.6)
    const leaf = withUV(new THREE.ConeGeometry(0.019, 0.05, 4, 1).translate(0, 0.025, 0).scale(1, 1, 0.28), STEM_U, 0.58, 0.95)
    parts.push(place(leaf, at, dir, nrm))
  }
  // chùm tai ở đỉnh quả
  for (let k = 0; k < 5; k++) {
    const a = (k / 5) * PI * 2 + 0.3
    const nrm = new THREE.Vector3(0, sin(a), Math.cos(a))
    const at = new THREE.Vector3(len / 2 - 0.012, 0.012 * sin(a), 0.012 * Math.cos(a))
    const leaf = withUV(new THREE.ConeGeometry(0.014, 0.04, 4, 1).translate(0, 0.02, 0).scale(1, 1, 0.3), STEM_U, 0.6, 0.95)
    parts.push(place(leaf, at, new THREE.Vector3(1, 0, 0).addScaledVector(nrm, 0.45), nrm))
  }
  return finish(parts)
}

function jackfruitGeometry() {
  const len = 0.6
  const S = 40
  // gai sần: xen kẽ lồi/lõm theo ô (i + j) chẵn/lẻ + bump map chi tiết
  const body = sweep({
    len, bend: 0.012, flatY: 0.86, rings: 40, segs: S,
    r: (t, a, i, j) => 0.19 * pow(sin(PI * t), 0.45) * (1 + 0.04 * sin(2 * a + 1.3) * sin(3 * PI * t)) * ((i + j) % 2 ? 1.022 : 0.985),
  })
  return finish([body, stem(0.02, 0.015, 0.06, new THREE.Vector3(len / 2 - 0.01, 0, 0), new THREE.Vector3(1, 0.1, 0))])
}

function sweetPotatoGeometry() {
  const len = 0.24
  const body = sweep({
    len, bend: 0.022, flatY: 0.86, rings: 30, segs: 18,
    r: (t, a) => 0.052 * pow(sin(PI * t), 0.75) * (1 + 0.09 * sin(3 * a + 5 * t) + 0.06 * sin(9 * t + a)),
  })
  return finish([
    body,
    stem(0.004, 0.0012, 0.05, new THREE.Vector3(len / 2 - 0.006, 0, 0.004), new THREE.Vector3(1, -0.05, 0.25)),
    stem(0.0035, 0.001, 0.04, new THREE.Vector3(-len / 2 + 0.006, 0, 0.004), new THREE.Vector3(-1, -0.05, 0.2)),
  ])
}

export function fruitGeometry(f: FruitType): THREE.BufferGeometry {
  switch (f) {
    case 'mango': return mangoGeometry()
    case 'banana': return bananaGeometry()
    case 'dragon_fruit': return dragonFruitGeometry()
    case 'jackfruit': return jackfruitGeometry()
    case 'sweet_potato': return sweetPotatoGeometry()
  }
}

// ------------------------------------------------------------------ texture quả (lành / hư)

type Paint = (g: CanvasRenderingContext2D, W: number, H: number, r: () => number) => void

/** canvas y ứng với v (CanvasTexture lật Y): v = 0 ở đáy canvas. Mặt trên quả (sin a > 0) là v ∈ (0, 0.5). */
const vy = (v: number, H: number) => (1 - v) * H

function stemArea(g: CanvasRenderingContext2D, W: number, H: number, leaf0: string, leaf1: string) {
  const x0 = Math.floor(0.94 * W)
  g.fillStyle = '#5a3d1e'
  g.fillRect(x0, H / 2, W - x0, H / 2)
  const grd = g.createLinearGradient(0, vy(0.55, H), 0, vy(0.95, H))
  grd.addColorStop(0, leaf0)
  grd.addColorStop(1, leaf1)
  g.fillStyle = grd
  g.fillRect(x0, 0, W - x0, H / 2)
}

function hGrad(g: CanvasRenderingContext2D, W: number, H: number, stops: [number, string][]) {
  const grd = g.createLinearGradient(0, 0, BODY_U * W, 0)
  for (const [t, c] of stops) grd.addColorStop(t, c)
  g.fillStyle = grd
  g.fillRect(0, 0, Math.ceil(BODY_U * W) + 1, H)
}

function blob(g: CanvasRenderingContext2D, x: number, y: number, rx: number, ry: number, inner: string, outer: string) {
  g.save()
  g.translate(x, y)
  g.scale(1, ry / rx)
  const grd = g.createRadialGradient(0, 0, rx * 0.15, 0, 0, rx)
  grd.addColorStop(0, inner)
  grd.addColorStop(0.55, inner)
  grd.addColorStop(1, outer)
  g.fillStyle = grd
  g.beginPath()
  g.arc(0, 0, rx, 0, PI * 2)
  g.fill()
  g.restore()
}

const PAINT: Record<FruitType, Paint> = {
  mango: (g, W, H, r) => {
    hGrad(g, W, H, [[0, '#f4ad35'], [0.55, '#f2b134'], [0.85, '#c9b23f'], [1, '#93a83a']])
    blob(g, 0.62 * BODY_U * W, vy(0.25, H), 0.32 * H, 0.22 * H, 'rgba(214,82,40,0.55)', 'rgba(214,82,40,0)')
    for (let i = 0; i < 160; i++) {
      g.fillStyle = `rgba(255,240,200,${0.25 + r() * 0.35})`
      g.fillRect(r() * BODY_U * W, r() * H, 1.2, 1.2)
    }
    stemArea(g, W, H, '#5f8a2a', '#3f6a1f')
  },
  banana: (g, W, H, r) => {
    hGrad(g, W, H, [[0, '#3b2a17'], [0.035, '#c9b23a'], [0.08, '#f3d23e'], [0.7, '#f1cf3c'], [0.84, '#b9c24a'], [0.9, '#8c9a3c'], [0.98, '#6a5a2a'], [1, '#3b2a17']])
    g.strokeStyle = 'rgba(150,120,30,0.38)'
    g.lineWidth = 2
    for (let k = 0; k <= 5; k++) {
      const y = vy(k / 5, H)
      g.beginPath()
      g.moveTo(0.04 * W, y)
      g.lineTo(0.77 * W, y)
      g.stroke()
    }
    for (let i = 0; i < 26; i++) {
      g.fillStyle = `rgba(110,70,20,${0.25 + r() * 0.3})`
      g.fillRect(0.08 * W + r() * 0.65 * W, r() * H, 1.5 + r() * 2, 1.5)
    }
    stemArea(g, W, H, '#7d8a3a', '#5d6a2a')
  },
  dragon_fruit: (g, W, H, r) => {
    hGrad(g, W, H, [[0, '#b82a5c'], [0.15, '#d6336c'], [0.8, '#de3f78'], [1, '#c23366']])
    for (let i = 0; i < 40; i++) blob(g, r() * BODY_U * W, r() * H, 6 + r() * 10, 4 + r() * 6, 'rgba(240,110,150,0.35)', 'rgba(240,110,150,0)')
    stemArea(g, W, H, '#d94b7b', '#7fae3c')
  },
  jackfruit: (g, W, H) => {
    hGrad(g, W, H, [[0, '#93a43c'], [0.5, '#a7b545'], [1, '#8e9e38']])
    // gai: lưới điểm so le, tối ở đỉnh gai
    for (let y = 0, row = 0; y < H; y += 4, row++) {
      for (let x = (row % 2) * 3; x < BODY_U * W; x += 6) {
        g.fillStyle = 'rgba(70,82,24,0.75)'
        g.fillRect(x, y, 2, 2)
      }
    }
    stemArea(g, W, H, '#6d7d2a', '#4f5e1e')
  },
  sweet_potato: (g, W, H, r) => {
    hGrad(g, W, H, [[0, '#4e2430'], [0.15, '#6c3242'], [0.5, '#76384a'], [0.85, '#6a3141'], [1, '#4e2430']])
    g.strokeStyle = 'rgba(60,25,35,0.3)'
    for (let i = 0; i < 40; i++) {
      g.lineWidth = 0.5 + r()
      const x = r() * BODY_U * W
      const y = r() * H
      g.beginPath()
      g.moveTo(x, y)
      g.lineTo(x + 10 + r() * 30, y + (r() - 0.5) * 3)
      g.stroke()
    }
    for (let i = 0; i < 14; i++) {
      const x = r() * BODY_U * W
      const y = r() * H
      g.fillStyle = 'rgba(200,150,150,0.35)'
      g.fillRect(x - 1, y - 1, 3, 3)
      g.fillStyle = 'rgba(40,15,20,0.7)'
      g.fillRect(x, y, 1, 1)
    }
    stemArea(g, W, H, '#5a3d1e', '#5a3d1e')
  },
}

/** Đốm thối/bệnh cho quả hư — ưu tiên mặt trên (v < 0.5) để camera nhìn từ trên thấy rõ. */
function paintDefect(f: FruitType, g: CanvasRenderingContext2D, W: number, H: number, r: () => number) {
  const top = () => vy(r() < 0.75 ? 0.08 + r() * 0.38 : r(), H)
  const x = () => (0.08 + r() * 0.78) * BODY_U * W
  const dark = 'rgba(28,16,8,0.95)'
  const brown = 'rgba(92,52,20,0)'
  if (f === 'jackfruit') {
    blob(g, x(), top(), 34 + r() * 16, 26, dark, brown)
    blob(g, x(), top(), 18 + r() * 10, 14, 'rgba(60,35,12,0.9)', brown)
    return
  }
  if (f === 'banana') {
    for (let i = 0; i < 6; i++) blob(g, (0.1 + r() * 0.62) * W, top(), 10 + r() * 16, 5 + r() * 5, dark, brown)
    return
  }
  const n = f === 'sweet_potato' ? 4 : 6
  for (let i = 0; i < n; i++) {
    const rr = 7 + r() * 9
    blob(g, x(), top(), rr * 1.5, rr, dark, brown)
  }
  if (f === 'mango') blob(g, 0.9 * BODY_U * W, top(), 20, 16, 'rgba(60,35,12,0.85)', brown) // thối cuống
  if (f === 'sweet_potato') {
    g.strokeStyle = 'rgba(20,10,8,0.85)'
    g.lineWidth = 1.5
    const x0 = x()
    const y0 = top()
    g.beginPath()
    g.moveTo(x0, y0)
    g.lineTo(x0 + 24, y0 + 3)
    g.lineTo(x0 + 40, y0 - 2)
    g.stroke()
  }
}

function makeTex(c: HTMLCanvasElement, srgb = true) {
  const t = new THREE.CanvasTexture(c)
  if (srgb) t.colorSpace = THREE.SRGBColorSpace
  t.anisotropy = 4
  return t
}

export function fruitMaterials(f: FruitType): { good: THREE.MeshStandardMaterial; bad: THREE.MeshStandardMaterial } {
  const W = 256
  const H = f === 'jackfruit' ? 256 : 128
  const seed = 11 + f.length * 97
  const draw = (defect: boolean) => {
    const c = document.createElement('canvas')
    c.width = W
    c.height = H
    const g = c.getContext('2d')!
    PAINT[f](g, W, H, rng(seed))
    if (defect) paintDefect(f, g, W, H, rng(seed * 3 + 1))
    return c
  }
  const rough = { mango: 0.5, banana: 0.55, dragon_fruit: 0.45, jackfruit: 0.85, sweet_potato: 0.9 }[f]
  const good = new THREE.MeshStandardMaterial({ map: makeTex(draw(false)), roughness: rough })
  const bad = new THREE.MeshStandardMaterial({ map: makeTex(draw(true)), roughness: Math.min(1, rough + 0.15) })
  if (f === 'jackfruit') {
    const c = document.createElement('canvas')
    c.width = c.height = 256
    const g = c.getContext('2d')!
    g.fillStyle = '#000'
    g.fillRect(0, 0, 256, 256)
    g.fillStyle = '#fff'
    for (let y = 0, row = 0; y < 256; y += 4, row++) {
      for (let x = (row % 2) * 3; x < 256; x += 6) {
        g.beginPath()
        g.arc(x + 1, y + 1, 1.6, 0, PI * 2)
        g.fill()
      }
    }
    const bump = makeTex(c, false)
    for (const m of [good, bad]) {
      m.bumpMap = bump
      m.bumpScale = 2.5
    }
  }
  return { good, bad }
}

// ------------------------------------------------------------------ phần sản phẩm sau sấy (khay nhỏ + miếng sấy)

/** Khay inox nhỏ đỡ phần 200 g (đáy ở y = 0). */
export function productTrayGeometry() {
  return new THREE.BoxGeometry(0.25, 0.012, 0.18).translate(0, 0.006, 0)
}

/** Miếng sấy theo loại quả, đặt sẵn trên mặt khay (y = 0.012). */
export function productGeometry(f: FruitType): THREE.BufferGeometry {
  const r = rng(5 + f.length * 31)
  const parts: THREE.BufferGeometry[] = []
  const put = (g: THREE.BufferGeometry, x: number, z: number, yaw: number) =>
    parts.push(g.rotateY(yaw).translate(x, 0.012, z))
  switch (f) {
    case 'mango':
      for (let i = 0; i < 4; i++) {
        const g = new THREE.CapsuleGeometry(0.015, 0.15, 2, 8).rotateZ(PI / 2).scale(1, 0.32, 1.05)
        g.computeBoundingBox()
        put(g.translate(0, -g.boundingBox!.min.y, 0), (r() - 0.5) * 0.02, -0.054 + i * 0.036, (r() - 0.5) * 0.2)
      }
      break
    case 'banana':
      for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) put(new THREE.CylinderGeometry(0.021, 0.021, 0.009, 12).translate(0, 0.0045, 0), -0.075 + i * 0.05 + (r() - 0.5) * 0.008, -0.05 + j * 0.05, 0)
      break
    case 'dragon_fruit':
      for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) put(new THREE.CylinderGeometry(0.036, 0.036, 0.009, 18).translate(0, 0.0045, 0), -0.074 + i * 0.074, -0.039 + j * 0.078, r() * PI)
      break
    case 'jackfruit':
      for (let i = 0; i < 7; i++) {
        const g = new THREE.SphereGeometry(0.026, 10, 6).scale(1.5, 0.45, 0.85)
        put(g.translate(0, 0.012, 0), -0.08 + (i % 4) * 0.053 + (i > 3 ? 0.026 : 0), i > 3 ? 0.035 : -0.035, (r() - 0.5) * 0.8)
      }
      break
    case 'sweet_potato':
      for (let i = 0; i < 7; i++) put(new THREE.BoxGeometry(0.11, 0.012, 0.015).translate(0, 0.006, 0), (r() - 0.5) * 0.06, -0.06 + i * 0.02, (r() - 0.5) * 0.25)
      break
  }
  for (const p of parts) {
    if (p.index === null) throw new Error('geometry phải có index')
  }
  return mergeGeometries(parts)!
}

/** Texture miếng sấy: hạt đường li ti; thanh long có hạt đen. Màu chính lấy từ màu instance. */
export function productMaterial(f: FruitType) {
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')!
  g.fillStyle = '#ffffff'
  g.fillRect(0, 0, 64, 64)
  const r = rng(77 + f.length)
  for (let i = 0; i < 220; i++) {
    const k = 200 + Math.floor(r() * 55)
    g.fillStyle = `rgb(${k},${k},${k})`
    g.fillRect(r() * 64, r() * 64, 1.5, 1.5)
  }
  if (f === 'dragon_fruit') {
    g.fillStyle = '#141414'
    for (let i = 0; i < 70; i++) g.fillRect(r() * 64, r() * 64, 1.4, 1.4)
  }
  return new THREE.MeshStandardMaterial({ map: makeTex(c), roughness: 0.75 })
}

// ------------------------------------------------------------------ túi gối hàn mép

/** Túi gối: mặt trên phồng, hai đầu ép mép phẳng (đáy ở y = 0). */
export function bagGeometry() {
  const W = 0.22
  const D = 0.3
  const g = new THREE.BoxGeometry(W, 0.056, D, 8, 1, 12).translate(0, 0.028, 0)
  const p = g.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) {
    if (p.getY(i) < 0.001) continue
    const fx = 1 - pow(Math.abs((2 * p.getX(i)) / W), 6)
    const zz = Math.abs((2 * p.getZ(i)) / D)
    const fz = zz > 0.9 ? 0 : 1 - pow(zz / 0.9, 5)
    p.setY(i, 0.005 + 0.051 * Math.sqrt(Math.max(0, fx * fz)))
  }
  g.computeVertexNormals()
  return g
}

export function bagMaterial() {
  const c = document.createElement('canvas')
  c.width = 128
  c.height = 160
  const g = c.getContext('2d')!
  g.fillStyle = '#f6f3ec'
  g.fillRect(0, 0, 128, 160)
  // mép hàn răng cưa hai đầu
  g.fillStyle = '#d8d3c8'
  g.fillRect(0, 0, 128, 12)
  g.fillRect(0, 148, 128, 12)
  g.fillStyle = '#bdb6a8'
  for (let x = 0; x < 128; x += 6) {
    g.fillRect(x, 4, 3, 4)
    g.fillRect(x, 152, 3, 4)
  }
  // cửa sổ trong suốt thấy sản phẩm
  g.fillStyle = '#9b7a52'
  g.beginPath()
  g.ellipse(64, 92, 34, 26, 0, 0, PI * 2)
  g.fill()
  g.fillStyle = '#2b2b2b'
  g.font = '600 22px "Segoe UI", sans-serif'
  g.fillText('NDQ', 42, 46)
  g.fillRect(30, 54, 68, 3)
  return new THREE.MeshStandardMaterial({ map: makeTex(c), roughness: 0.35, metalness: 0.05 })
}
