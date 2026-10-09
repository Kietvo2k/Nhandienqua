import { describe, expect, it } from 'vitest'
import { pickLanUrl } from '../src/ui/lan'

describe('pickLanUrl', () => {
  it('bỏ địa chỉ link-local 169.254.x.x, ưu tiên mạng riêng 192.168/10/172.16-31', () => {
    expect(pickLanUrl(['http://169.254.83.107:5173/', 'http://192.168.1.23:5173/'])).toBe('http://192.168.1.23:5173/')
    expect(pickLanUrl(['http://172.20.0.5:5173/', 'http://10.0.0.8:5173/'])).toBe('http://10.0.0.8:5173/')
  })

  it('không có địa chỉ dùng được → null', () => {
    expect(pickLanUrl(['http://169.254.1.2:5173/'])).toBeNull()
    expect(pickLanUrl([])).toBeNull()
  })

  it('chấp nhận IP công cộng khi không có IP mạng riêng', () => {
    expect(pickLanUrl(['http://169.254.1.2:5173/', 'http://203.0.113.5:5173/'])).toBe('http://203.0.113.5:5173/')
  })
})
