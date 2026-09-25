import { existsSync, readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `vite --mode lan` serves over HTTPS so a phone on the same network can open
// the in-page camera (browsers only allow it on secure origins). The
// self-signed certificate lives in tmp/lan-cert/ and is never committed:
//   openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
//     -keyout tmp/lan-cert/key.pem -out tmp/lan-cert/cert.pem \
//     -subj "/CN=winescanner-dev" -addext "subjectAltName=IP:<LAN IP>"
const cert = "tmp/lan-cert";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  root: "frontend",
  build: { outDir: "../dist", emptyOutDir: true },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
    https:
      mode === "lan" && existsSync(`${cert}/key.pem`)
        ? {
            key: readFileSync(`${cert}/key.pem`),
            cert: readFileSync(`${cert}/cert.pem`),
          }
        : undefined,
  },
}));
