import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin, type ViteDevServer } from 'vite'

const ROOT = path.dirname(fileURLToPath(import.meta.url))
const VIDEO_DIR = path.join(ROOT, 'public', 'videos')
const TRACE_RE = /^\/(NDQ-[A-Z]+-\d{8}-\d{3})$/
const MAX_VIDEO_BYTES = 20 * 1024 * 1024

/**
 * Endpoint nhỏ chỉ có ở dev server:
 *  GET  /api/lan                → địa chỉ mạng LAN (để QR mở được trên điện thoại)
 *  POST /api/video/<mã lô>      → lưu clip WebM vào web/public/videos/<mã lô>.webm (trong dự án)
 */
function simServer(): Plugin {
  return {
    name: 'ndq-sim-server',
    configureServer(server: ViteDevServer) {
      server.middlewares.use('/api/lan', (_req, res) => {
        res.setHeader('content-type', 'application/json')
        res.end(JSON.stringify({ network: server.resolvedUrls?.network ?? [] }))
      })
      server.middlewares.use('/api/video', (req, res) => {
        const m = TRACE_RE.exec(req.url ?? '')
        if (req.method !== 'POST' || !m) {
          res.statusCode = req.method !== 'POST' ? 405 : 400
          return res.end()
        }
        const chunks: Buffer[] = []
        let size = 0
        req.on('data', (c: Buffer) => {
          size += c.length
          if (size > MAX_VIDEO_BYTES) {
            res.statusCode = 413
            res.end()
            req.destroy()
          } else chunks.push(c)
        })
        req.on('end', () => {
          if (res.writableEnded) return
          fs.mkdirSync(VIDEO_DIR, { recursive: true })
          fs.writeFileSync(path.join(VIDEO_DIR, `${m[1]}.webm`), Buffer.concat(chunks))
          res.end('ok')
        })
      })
    },
  }
}

export default defineConfig({
  plugins: [simServer()],
  server: { host: true },
  // onnxruntime-web tự nạp file .wasm/.mjs cạnh module của nó → không pre-bundle để Vite phục vụ đúng đường dẫn
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  test: { include: ['tests/**/*.test.ts'] },
} as never)
