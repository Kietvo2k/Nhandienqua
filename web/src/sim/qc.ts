/**
 * Cỡ mẫu theo ISO 2859-1, kế hoạch lấy mẫu đơn, kiểm tra thường, mức kiểm tra chung II.
 * Bảng: cỡ lô tối đa → cỡ mẫu (mã chữ A…N).
 */
const ISO2859_II: ReadonlyArray<readonly [number, number]> = [
  [8, 2],
  [15, 3],
  [25, 5],
  [50, 8],
  [90, 13],
  [150, 20],
  [280, 32],
  [500, 50],
  [1200, 80],
  [3200, 125],
  [10000, 200],
  [35000, 315],
  [150000, 500],
  [500000, 800],
]

export function sampleSize(lotSize: number): number {
  if (lotSize <= 0) return 0
  const row = ISO2859_II.find(([max]) => lotSize <= max)
  const n = row ? row[1] : 1250
  return Math.min(n, lotSize)
}
