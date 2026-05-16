const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { OpenAI } = require('openai');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({ origin: '*', methods: ['GET', 'POST'] }));
app.use(express.json());
app.use(express.static(path.join(__dirname, '.')));

// ─── Cliente Groq ─────────────────────────────────────────────────────────────
const groq = new OpenAI({
    apiKey: process.env.GROQ_API_KEY,
    baseURL: 'https://api.groq.com/openai/v1',
});

// ─── Carga de documentos ──────────────────────────────────────────────────────
const docsPath = path.join(__dirname, 'docs');
const EMBEDDING_CACHE_FILE = path.join(__dirname, '.embeddings_cache.json');
let manualChunks = [];

function cargarDocumentos() {
    try {
        const archivos = fs.readdirSync(docsPath).filter(f => f.endsWith('.md'));
        let tempChunks = [];

        for (const archivo of archivos) {
            const contenido = fs.readFileSync(path.join(docsPath, archivo), 'utf8');
            const lines = contenido.split(/\n/);
            let currentChunk = '';

            for (const line of lines) {
                if ((currentChunk.length + line.length) > 1500 && currentChunk.length > 0) {
                    tempChunks.push(currentChunk.trim());
                    currentChunk = line;
                } else {
                    currentChunk += (currentChunk ? '\n' : '') + line;
                }
            }
            if (currentChunk.trim().length > 0) tempChunks.push(currentChunk.trim());
        }

        manualChunks = tempChunks
            .filter(s => s.length > 50)
            .map((s, i) => ({ id: i, content: s, lower: s.toLowerCase(), embedding: null }));

        console.log(`✅ Documentos cargados: ${archivos.length} | Chunks: ${manualChunks.length}`);
    } catch (error) {
        console.error('❌ Error leyendo documentos:', error);
    }
}

// ─── Embeddings locales (@xenova/transformers) ────────────────────────────────
let embedder = null;

async function obtenerEmbedding(texto) {
    if (!embedder) {
        const { pipeline } = await import('@xenova/transformers');
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    const output = await embedder(texto, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

// ─── Cache de embeddings ──────────────────────────────────────────────────────
function cargarCacheEmbeddings() {
    try {
        if (fs.existsSync(EMBEDDING_CACHE_FILE)) {
            const cache = JSON.parse(fs.readFileSync(EMBEDDING_CACHE_FILE, 'utf8'));
            let hits = 0;
            for (const chunk of manualChunks) {
                if (cache[chunk.id] && cache[chunk.id].content === chunk.content) {
                    chunk.embedding = cache[chunk.id].embedding;
                    hits++;
                }
            }
            console.log(`💾 Cache de embeddings: ${hits}/${manualChunks.length} chunks restaurados`);
            return hits;
        }
    } catch (e) {
        console.warn('⚠️  No se pudo cargar cache:', e.message);
    }
    return 0;
}

function guardarCacheEmbeddings() {
    try {
        const cache = {};
        for (const chunk of manualChunks) {
            if (chunk.embedding) {
                cache[chunk.id] = { content: chunk.content, embedding: chunk.embedding };
            }
        }
        fs.writeFileSync(EMBEDDING_CACHE_FILE, JSON.stringify(cache));
        console.log(`💾 Cache guardado (${Object.keys(cache).length} chunks)`);
    } catch (e) {
        console.warn('⚠️  No se pudo guardar cache:', e.message);
    }
}

async function precalcularEmbeddings() {
    const faltantes = manualChunks.filter(c => !c.embedding);
    if (faltantes.length === 0) {
        console.log('✅ Todos los embeddings ya están en cache');
        return;
    }

    console.log(`⏳ Generando embeddings para ${faltantes.length} chunks...`);
    const BATCH_SIZE = 10;

    for (let i = 0; i < faltantes.length; i += BATCH_SIZE) {
        const batch = faltantes.slice(i, i + BATCH_SIZE);
        try {
            for (const chunk of batch) {
                chunk.embedding = await obtenerEmbedding(chunk.content);
            }
            console.log(`  Embeddings: ${Math.min(i + BATCH_SIZE, faltantes.length)}/${faltantes.length}`);
        } catch (err) {
            console.error(`  ❌ Error en batch ${i}-${i + BATCH_SIZE}:`, err.message);
            await new Promise(r => setTimeout(r, 2000));
        }
    }

    guardarCacheEmbeddings();
    console.log('✅ Embeddings generados y guardados');
}

// ─── Funciones de búsqueda ────────────────────────────────────────────────────
function cosineSim(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

function buscarKeywords(pregunta, topN = 10) {
    const stopwords = new Set(['que', 'los', 'las', 'una', 'con', 'por', 'para', 'del', 'como', 'son', 'sus', 'mas', 'este', 'esta', 'esto', 'ese', 'esos', 'esas', 'hay', 'puede', 'tienen', 'tambien']);
    const palabras = pregunta
        .toLowerCase()
        .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
        .split(/\s+/)
        .filter(p => p.length > 2 && !stopwords.has(p));

    const resultados = manualChunks.map(chunk => {
        let score = 0;
        const chunkNorm = chunk.lower.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
        for (const palabra of palabras) {
            const exactMatches = (chunkNorm.match(new RegExp(`\\b${palabra}\\b`, 'g')) || []).length;
            const partialMatches = (chunkNorm.match(new RegExp(palabra, 'g')) || []).length;
            score += exactMatches * 2 + partialMatches * 0.5;
        }
        if (palabras.some(p => chunk.lower.split('\n')[0].includes(p))) score += 3;
        return { ...chunk, score };
    });

    return resultados
        .filter(r => r.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, topN);
}

async function buscarSemantico(pregunta, topN = 10) {
    const chunksConEmbedding = manualChunks.filter(c => c.embedding);
    if (chunksConEmbedding.length === 0) return [];

    let queryEmbedding;
    try {
        queryEmbedding = await obtenerEmbedding(pregunta);
    } catch (err) {
        console.warn('⚠️  No se pudo generar embedding de la query:', err.message);
        return [];
    }

    return chunksConEmbedding
        .map(chunk => ({ ...chunk, score: cosineSim(queryEmbedding, chunk.embedding) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topN);
}

function rrf(listaKeywords, listaSemantica, k = 60, topN = 4) {
    const scores = new Map();

    const agregar = (lista, peso) => {
        lista.forEach((item, rank) => {
            const prev = scores.get(item.id) || 0;
            scores.set(item.id, prev + peso * (1 / (k + rank + 1)));
        });
    };

    agregar(listaKeywords, 1.0);
    agregar(listaSemantica, 1.2);

    const chunkMap = new Map(manualChunks.map(c => [c.id, c]));
    return [...scores.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, topN)
        .map(([id]) => chunkMap.get(id))
        .filter(Boolean);
}

async function buscarHibrid(pregunta) {
    const [keywordResults, semanticResults] = await Promise.all([
        Promise.resolve(buscarKeywords(pregunta, 10)),
        buscarSemantico(pregunta, 10),
    ]);

    const mejores = rrf(keywordResults, semanticResults, 60, 4);

    if (mejores.length === 0) {
        return 'No se encontró contexto relevante en el manual.';
    }

    return mejores.map(r => r.content).join('\n\n---\n\n');
}

// ─── Endpoints ────────────────────────────────────────────────────────────────
app.get('/api/sugerencias', (req, res) => {
    res.json({
        sugerencias: [
            '¿Qué es el programa Plerd?',
            '¿Cuáles son los requisitos de ingreso?',
            '¿Cuáles son los valores de CELIDER?',
            '¿Qué habilidades esenciales promueve el PLE-RD?',
        ],
    });
});

app.post('/api/chat', async (req, res) => {
    try {
        const { pregunta } = req.body;

        if (!pregunta) return res.status(400).json({ respuesta: 'Por favor, envía una pregunta.' });
        if (pregunta.length > 500) return res.status(400).json({ respuesta: 'Pregunta demasiado larga.' });

        const contexto = await buscarHibrid(pregunta);

        const completion = await groq.chat.completions.create({
            messages: [
                {
                    role: 'system',
                    content: `Eres CELI, el asistente inteligente oficial del PLE-RD (Programa de Liderazgo Estudiantil de la República Dominicana), una iniciativa de CELIDER Grandeza que forma a jóvenes líderes dominicanos en habilidades esenciales, valores y pensamiento crítico.

Tu misión es asistir a delegados, facilitadores y participantes del PLE-RD respondiendo sus dudas de forma clara, directa y útil.

REGLAS:
- Responde directo, sin frases como "basado en el contexto" o "según el documento"
- Si no tienes información suficiente, di: "Esa información no está en mi base de datos del PLE-RD. Consulta a tu facilitador."
- Sé amigable, profesional y conciso
- Máximo 3 párrafos por respuesta
- Habla en español dominicano natural

CONTEXTO DEL MANUAL:
${contexto}`,
                },
                { role: 'user', content: pregunta },
            ],
            model: 'llama-3.3-70b-versatile',
            temperature: 0.3,
            max_tokens: 500,
        });

        const respuesta = completion.choices[0]?.message?.content || 'No pude generar una respuesta.';
        res.json({ respuesta });

    } catch (error) {
        console.error('ERROR:', error.response?.data || error.message);
        res.status(500).json({ respuesta: 'Error interno del servidor.' });
    }
});

app.get('/api/health', (req, res) => {
    const chunksConEmbedding = manualChunks.filter(c => c.embedding).length;
    res.json({
        status: 'ok',
        hasApiKey: !!process.env.GROQ_API_KEY,
        chunksLoaded: manualChunks.length,
        chunksWithEmbedding: chunksConEmbedding,
        embeddingReady: chunksConEmbedding === manualChunks.length,
    });
});

// ─── Inicio ───────────────────────────────────────────────────────────────────
cargarDocumentos();
const embeddingHits = cargarCacheEmbeddings();

app.listen(PORT, async () => {
    console.log(`\n🚀 Servidor corriendo en http://localhost:${PORT}`);
    console.log(`   GET  /api/sugerencias`);
    console.log(`   POST /api/chat`);
    console.log(`   GET  /api/health\n`);

    if (embeddingHits < manualChunks.length) {
        precalcularEmbeddings().catch(console.error);
    }
});