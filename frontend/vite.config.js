import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api/analyze-stream': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 1800000,
        proxyTimeout: 1800000,
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache';
            proxyRes.headers['x-accel-buffering'] = 'no';
          });
        },
      },
      '/api/analyze-assembled': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 1800000,
        proxyTimeout: 1800000,
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            proxyRes.headers['cache-control'] = 'no-cache';
            proxyRes.headers['x-accel-buffering'] = 'no';
          });
        },
      },
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 1800000,
        proxyTimeout: 1800000,
      },
      '/uploads': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
