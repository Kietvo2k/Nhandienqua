/**
 * Tham số mô phỏng. Số liệu sấy là MINH HOẠ, lấy theo các quy trình sấy dẻo phổ biến ở VN
 * (vd. xoài: 50–65°C, 6–12 giờ, độ ẩm thành phẩm 15–20%, thu hồi ~30% so với thịt quả).
 * Không dùng làm thông số vận hành thật.
 */
export type FruitType = 'mango' | 'banana' | 'dragon_fruit' | 'jackfruit' | 'sweet_potato'

export interface FruitParams {
  vi: string
  product: string
  /** tỉ lệ quả hư/bệnh trong nguyên liệu đầu vào */
  defectRate: number
  defectTypes: string[]
  massKg: number
  dryTempC: number
  dryHours: number
  /** độ ẩm (% khối lượng ướt) */
  moistureIn: number
  moistureTarget: number
  /** khối lượng thành phẩm / khối lượng quả nhận */
  yield: number
  shelfLifeMonths: number
  color: number
}

export const FRUITS: Record<FruitType, FruitParams> = {
  mango: {
    vi: 'Xoài', product: 'Xoài sấy dẻo', defectRate: 0.12,
    defectTypes: ['thán thư', 'thối cuống', 'đốm đen Alternaria', 'mốc đen'],
    massKg: 0.35, dryTempC: 60, dryHours: 8, moistureIn: 82, moistureTarget: 18, yield: 0.2,
    shelfLifeMonths: 6, color: 0xf2b134,
  },
  banana: {
    vi: 'Chuối', product: 'Chuối sấy dẻo', defectRate: 0.1,
    defectTypes: ['thán thư', 'thối núm', 'dập nát'],
    massKg: 0.15, dryTempC: 60, dryHours: 10, moistureIn: 75, moistureTarget: 18, yield: 0.22,
    shelfLifeMonths: 6, color: 0xf5d547,
  },
  dragon_fruit: {
    vi: 'Thanh long', product: 'Thanh long sấy dẻo', defectRate: 0.08,
    defectTypes: ['thán thư', 'đốm nâu', 'thối mềm'],
    massKg: 0.4, dryTempC: 55, dryHours: 12, moistureIn: 85, moistureTarget: 16, yield: 0.12,
    shelfLifeMonths: 6, color: 0xd6336c,
  },
  jackfruit: {
    vi: 'Mít', product: 'Mít sấy dẻo', defectRate: 0.15,
    defectTypes: ['méo / dị dạng', 'thối', 'sâu'],
    massKg: 8, dryTempC: 60, dryHours: 10, moistureIn: 74, moistureTarget: 16, yield: 0.13,
    shelfLifeMonths: 6, color: 0x9bb03a,
  },
  sweet_potato: {
    vi: 'Khoai lang', product: 'Khoai lang sấy dẻo', defectRate: 0.1,
    defectTypes: ['thối', 'chuột cắn', 'sâu đục', 'nứt'],
    massKg: 0.3, dryTempC: 65, dryHours: 8, moistureIn: 70, moistureTarget: 18, yield: 0.3,
    shelfLifeMonths: 6, color: 0x9c4f6b,
  },
}

export const FRUIT_TYPES = Object.keys(FRUITS) as FruitType[]

export interface LineParams {
  schedule: FruitType[]
  /** số xe tối đa (0 = chạy mãi theo vòng lịch) */
  maxTrucks: number
  /** số quả đại diện mỗi xe (thu nhỏ so với thực tế để xem được) */
  truckLoad: number
  truckArriveS: number
  truckLeaveS: number
  truckIntervalS: number
  /** khoảng cách tối thiểu giữa hai quả trên băng (m) */
  itemGap: number
  beltSpeed: number
  belt1Length: number
  cameraAt: number
  gateAt: number
  /** AI giả lập: xác suất báo hư khi quả hư thật */
  aiRecall: number
  /** AI giả lập: xác suất báo hư khi quả tốt */
  aiFalseReject: number
  /** thời gian sấy thật / thời gian sấy mô phỏng */
  dryerCompression: number
  /** thời gian sấy bổ sung so với chu kỳ chính */
  redryFraction: number
  bagKg: number
  /** độ lệch chuẩn độ ẩm giữa các khay (điểm %) */
  dryingNoiseSd: number
  belt2Speed: number
  belt2Length: number
  unitGap: number
  dryCamAt: number
  qcAt: number
  packAt: number
  belt3Speed: number
  belt3Length: number
  labelAt: number
  /** AI giả lập mức sấy: độ lệch chuẩn sai số ước lượng độ ẩm (điểm %) */
  dryAiSd: number
}

export const DEFAULT_LINE: LineParams = {
  schedule: [...FRUIT_TYPES],
  maxTrucks: 0,
  truckLoad: 120,
  truckArriveS: 6,
  truckLeaveS: 5,
  truckIntervalS: 4,
  itemGap: 0.45,
  beltSpeed: 0.8,
  belt1Length: 22,
  cameraAt: 8,
  gateAt: 11,
  aiRecall: 0.93,
  aiFalseReject: 0.03,
  dryerCompression: 960,
  redryFraction: 0.25,
  bagKg: 0.2,
  dryingNoiseSd: 1.6,
  belt2Speed: 0.7,
  belt2Length: 20,
  unitGap: 0.5,
  dryCamAt: 3,
  qcAt: 8,
  packAt: 14,
  belt3Speed: 0.7,
  belt3Length: 10,
  labelAt: 5,
  dryAiSd: 0.8,
}

/** Ngưỡng phân loại mức sấy theo độ ẩm so với mục tiêu. */
export function dryingLabel(moisture: number, target: number): 'under' | 'ok' | 'over' {
  if (moisture > target + 2.5) return 'under'
  if (moisture < target - 4) return 'over'
  return 'ok'
}
