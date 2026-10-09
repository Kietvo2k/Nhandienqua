/**
 * Bố trí nhà xưởng (mét, trục Y hướng lên). Ánh xạ toạ độ băng chuyền s (m) của engine → vị trí 3D.
 *
 *   xe tải → sàn dock → băng cấp liệu nghiêng → phễu → BĂNG 1 (x: -10 → 12, z = 0) → sơ chế → xe khay → LÒ SẤY (x 16..24)
 *   lò → ống chuyển kín → chụp làm nguội → BĂNG 2 (x: 24 → 10, z = -6): camera mức sấy, cửa đẩy, trạm QC, máy đóng gói
 *   BĂNG 3 (x: 9 → -1, z = -6): máy dán nhãn QR → trạm vô thùng → pallet thành phẩm
 */
import { Vector3 } from 'three'
import { DEFAULT_LINE } from '../sim/params'

export const BELT_Y = 0.85
export const L = DEFAULT_LINE

export const belt1 = (s: number, out = new Vector3()) => out.set(-10 + s, BELT_Y, 0)
export const belt2 = (s: number, out = new Vector3()) => out.set(24 - s, BELT_Y, -6)
export const belt3 = (s: number, out = new Vector3()) => out.set(9 - s, BELT_Y, -6)

/** Điểm cửa đẩy trên băng 2 (sau camera mức sấy): quá khô → thùng loại, chưa đạt → xe khay sấy lại. */
export const EJECT_OVER_S = L.dryCamAt + 1.0
export const EJECT_REDRY_S = L.dryCamAt + 1.8

/** Băng cấp liệu nghiêng: từ sàn dock (chân) lên miệng phễu (đầu). */
export const FEED = {
  foot: new Vector3(-12.75, 1.0, 0),
  head: new Vector3(-10.3, 2.32, 0),
}

export const SPOTS = {
  truckDock: new Vector3(-15.6, 0, 0),
  truckFar: new Vector3(-46, 0, 0),
  /** phễu đặt ngay trên đầu băng 1 để quả rơi xuống băng */
  hopper: new Vector3(-9.85, 0, 0),
  sortCam: belt1(L.cameraAt),
  gate: belt1(L.gateAt),
  trash: new Vector3(L.gateAt - 10, 0, -2.05),
  /** mặt trước máy sơ chế ở x = 11.6 để quả vào hẳn trong máy trước khi engine xoá */
  prep: new Vector3(12.9, 0, 0),
  /** xe đẩy khay chờ sấy, ngay trước cửa tây lò */
  trayRack: new Vector3(15.1, 0, 0),
  dryer: new Vector3(20, 0, 0),
  dryCam: belt2(L.dryCamAt),
  trash2: new Vector3(belt2(EJECT_OVER_S).x, 0, -4.95),
  redryCart: new Vector3(belt2(EJECT_REDRY_S).x, 0, -7.15),
  qc: belt2(L.qcAt),
  qcTable: new Vector3(24 - L.qcAt, 0, -6.95),
  lab: new Vector3(24 - L.qcAt, 0, -10.2),
  pack: belt2(L.packAt),
  label: belt3(L.labelAt),
  /** trạm vô thùng cuối băng 3 */
  cartonStation: new Vector3(-1.6, 0, -6),
  pallet: new Vector3(-3.35, 0, -6),
}

export interface CamDef {
  id: string
  name: string
  pos: [number, number, number]
  look: [number, number, number]
  fov: number
  /** loại khung phủ: 'sort' vẽ bbox phân loại, 'dry' vẽ nhãn mức sấy */
  overlay?: 'sort' | 'dry' | 'dryer'
}

const v = (p: Vector3, dy = 0): [number, number, number] => [p.x, p.y + dy, p.z]

export const CAMERAS: CamDef[] = [
  { id: 'CAM-01', name: 'Dock nhận hàng', pos: [-8.6, 4.4, 6.2], look: [-12.6, 1.2, 0], fov: 55 },
  { id: 'CAM-02', name: 'AI phân loại quả', pos: [SPOTS.sortCam.x, 2.35, 0.001], look: v(SPOTS.sortCam), fov: 52, overlay: 'sort' },
  { id: 'CAM-03', name: 'Cửa gạt & thùng rác', pos: [SPOTS.gate.x + 2.4, 2.9, 2.4], look: [SPOTS.gate.x, 0.5, -1.2], fov: 55 },
  { id: 'CAM-04', name: 'Lò sấy', pos: [20, 5.5, 9], look: [20, 1.4, 0], fov: 50, overlay: 'dryer' },
  { id: 'CAM-05', name: 'AI mức sấy', pos: [SPOTS.dryCam.x, 2.1, -5.999], look: v(SPOTS.dryCam), fov: 50, overlay: 'dry' },
  { id: 'CAM-06', name: 'Lấy mẫu kiểm nghiệm', pos: [SPOTS.qc.x + 2.1, 2.5, -4.3], look: [SPOTS.qc.x - 0.3, 0.8, -7.6], fov: 55 },
  { id: 'CAM-07', name: 'Máy đóng gói', pos: [SPOTS.pack.x + 2.8, 3.2, -2.6], look: [SPOTS.pack.x - 0.6, 1.1, -6], fov: 55 },
  { id: 'CAM-08', name: 'Dán nhãn QR', pos: [SPOTS.label.x + 0.8, 2.5, -3.6], look: v(SPOTS.label), fov: 55 },
]
