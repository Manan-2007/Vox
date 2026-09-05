import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Honour PORT when it is set. Vite otherwise picks 5173 and walks upward on a
  // collision, which silently strands any tooling that was told which port to
  // expect — including two dev servers running side by side.
  server: { port: Number(process.env.PORT) || 5173 },
})
