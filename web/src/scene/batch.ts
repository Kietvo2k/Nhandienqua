/**
 * Gộp các mesh tĩnh thành ít mesh nhất có thể (giảm draw call — cảnh được vẽ 9 lần/khung hình + bóng).
 * - Vật liệu trơn (không texture, không trong suốt) được "nướng" màu vào thuộc tính color và dùng chung
 *   một vật liệu theo lớp (kim loại / mờ / không chiếu sáng) × mặt (một mặt / hai mặt).
 * - Vật liệu có texture / trong suốt: gộp theo đúng vật liệu đó.
 * Bỏ qua cây con có `userData.dynamic = true` (cửa gạt, đầu dán nhãn…), InstancedMesh và Sprite.
 */
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

const KEEP = new Set(['position', 'normal', 'uv'])

function isDynamic(o: THREE.Object3D) {
  for (let p: THREE.Object3D | null = o; p; p = p.parent) if (p.userData.dynamic) return true
  return false
}

/** Vật liệu dùng chung (màu theo đỉnh) cho từng lớp. */
const shared = new Map<string, THREE.Material>()
function sharedMaterial(cls: 'metal' | 'matte' | 'basic', side: THREE.Side) {
  const key = `${cls}|${side}`
  let m = shared.get(key)
  if (!m) {
    m = cls === 'basic'
      ? new THREE.MeshBasicMaterial({ vertexColors: true, side })
      : new THREE.MeshStandardMaterial({ vertexColors: true, side, roughness: cls === 'metal' ? 0.48 : 0.72, metalness: cls === 'metal' ? 0.6 : 0.12 })
    shared.set(key, m)
  }
  return m
}

/** Vật liệu có thể nướng màu: trơn, không texture, không trong suốt. Trả về lớp hoặc null. */
function bakeClass(m: THREE.Material): 'metal' | 'matte' | 'basic' | null {
  if (m.transparent) return null
  if ((m as THREE.MeshBasicMaterial).isMeshBasicMaterial) return (m as THREE.MeshBasicMaterial).map ? null : 'basic'
  if ((m as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
    const s = m as THREE.MeshStandardMaterial
    if (s.map || s.emissiveMap || s.bumpMap || s.normalMap) return null
    return s.metalness >= 0.5 ? 'metal' : 'matte'
  }
  return null
}

/** `root` phải là Scene (ma trận gốc = đơn vị). Trả về số mesh đã gộp. */
export function batchStatic(root: THREE.Object3D): number {
  root.updateMatrixWorld(true)
  const groups = new Map<string, { mat: THREE.Material; bake: boolean; list: THREE.Mesh[] }>()
  root.traverse((o) => {
    const m = o as THREE.Mesh
    if (!m.isMesh || (m as THREE.InstancedMesh).isInstancedMesh || Array.isArray(m.material) || isDynamic(m)) return
    const g = m.geometry
    if (!g.index || !g.attributes.normal || !g.attributes.uv) return
    const cls = bakeClass(m.material)
    const mat = cls ? sharedMaterial(cls, m.material.side) : m.material
    const key = `${mat.uuid}|${+m.castShadow}${+m.receiveShadow}|${m.layers.mask}|${m.renderOrder}`
    let grp = groups.get(key)
    if (!grp) groups.set(key, (grp = { mat, bake: !!cls, list: [] }))
    grp.list.push(m)
  })
  let merged = 0
  for (const { mat, bake, list } of groups.values()) {
    if (list.length < 2 && !bake) continue
    const geos = list.map((m) => {
      const g = m.geometry.clone()
      for (const k of Object.keys(g.attributes)) if (!KEEP.has(k)) g.deleteAttribute(k)
      g.clearGroups()
      if (bake) {
        const c = (m.material as THREE.MeshStandardMaterial).color
        const n = g.attributes.position.count
        const a = new Float32Array(n * 3)
        for (let i = 0; i < n; i++) a.set([c.r, c.g, c.b], i * 3)
        g.setAttribute('color', new THREE.BufferAttribute(a, 3))
      }
      return g.applyMatrix4(m.matrixWorld)
    })
    const geo = mergeGeometries(geos)
    for (const g of geos) g.dispose()
    if (!geo) continue
    const first = list[0]
    const mesh = new THREE.Mesh(geo, mat)
    mesh.castShadow = first.castShadow
    mesh.receiveShadow = first.receiveShadow
    mesh.layers.mask = first.layers.mask
    mesh.renderOrder = first.renderOrder
    mesh.matrixAutoUpdate = false
    root.add(mesh)
    for (const m of list) m.removeFromParent()
    merged += list.length
  }
  return merged
}
