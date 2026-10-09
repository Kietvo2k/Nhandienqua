/** Chọn địa chỉ mạng mà điện thoại cùng Wi-Fi mở được (bỏ link-local 169.254.x.x). */
export function pickLanUrl(urls: string[]): string | null {
  const score = (u: string) => {
    const host = /^https?:\/\/([\d.]+)/.exec(u)?.[1]
    if (!host) return -1
    const [a, b] = host.split('.').map(Number)
    if (a === 169 && b === 254) return -1 // link-local (APIPA)
    if (a === 127) return -1
    if (a === 192 && b === 168) return 3 // Wi-Fi gia đình / văn phòng phổ biến nhất
    if (a === 10) return 2
    if (a === 172 && b >= 16 && b <= 31) return 1 // hay là mạng ảo (Docker/WSL) → ưu tiên thấp
    return 0
  }
  const best = urls.map((u) => [score(u), u] as const).filter(([s]) => s >= 0).sort((x, y) => y[0] - x[0])[0]
  return best ? best[1] : null
}
