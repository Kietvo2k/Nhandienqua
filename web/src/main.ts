import './style.css'

const app = document.querySelector<HTMLElement>('#app')!
const m = /^#\/p\/(.+)$/.exec(location.hash)
if (m) {
  // Trang sản phẩm (mở từ QR) — không tải Three.js
  import('./product/page').then(({ renderProductPage }) => renderProductPage(app, m[1]))
} else if (location.hash === '#/lab') {
  // Trang thử model ONNX thật — không tải Three.js
  import('./lab/page').then(({ renderLab }) => renderLab(app))
} else {
  import('./app').then(({ startFactoryApp }) => startFactoryApp())
}
window.addEventListener('hashchange', () => location.reload())
