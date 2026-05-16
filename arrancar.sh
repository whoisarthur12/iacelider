#!/bin/bash
echo "🚀 Iniciando CELI en modo local..."
cd "/home/miguelsuarez160209/Proyecto CELIDER13"

# Verificar si existe .env
if [ ! -f .env ]; then
    echo "⚠️  No existe .env - creándolo..."
    echo "GROQ_API_KEY=pon_tu_key_aqui" > .env
    echo "PORT=3000" >> .env
    echo "✅ Edita el archivo .env con tu key real"
    exit 1
fi

# Instalar dependencias si hace falta
if [ ! -d node_modules ]; then
    echo "📦 Instalando dependencias..."
    npm install
fi

echo "✅ Servidor corriendo en http://localhost:3000"
echo "   Presiona Ctrl+C para detener"
node server.js
