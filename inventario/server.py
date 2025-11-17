#!/usr/bin/env python3
"""
Servidor HTTP simple con CORS para desarrollo local
Sirve archivos estáticos para la aplicación web de inventario
"""

import http.server
import socketserver
import os

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Habilitar CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Headers requeridos para ONNX Runtime Web con WebAssembly
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        
        super().end_headers()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("="*70)
        print(f"🌐 Servidor HTTP iniciado en http://localhost:{PORT}")
        print("="*70)
        print(f"📂 Directorio: {os.getcwd()}")
        print(f"📝 Accede a: http://localhost:{PORT}/index.html")
        print("🛑 Presiona Ctrl+C para detener el servidor")
        print("="*70)
        httpd.serve_forever()
