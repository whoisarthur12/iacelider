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

// ─── Proveedores de IA (carrera paralela) ─────────────────────────────────────
const PROVIDERS = [
    {
        name: 'Groq',
        enabled: !!process.env.GROQ_API_KEY,
        client: process.env.GROQ_API_KEY ? new OpenAI({
            apiKey: process.env.GROQ_API_KEY,
            baseURL: 'https://api.groq.com/openai/v1',
        }) : null,
        model: 'llama-3.3-70b-versatile',
        max_tokens: 500,
    },
    {
        name: 'Groq-Gemma',
        enabled: !!process.env.GROQ_API_KEY,
        client: process.env.GROQ_API_KEY ? new OpenAI({
            apiKey: process.env.GROQ_API_KEY,
            baseURL: 'https://api.groq.com/openai/v1',
        }) : null,
        model: 'gemma2-9b-it',
        max_tokens: 500,
    },
    {
        name: 'Groq-Mixtral',
        enabled: !!process.env.GROQ_API_KEY,
        client: process.env.GROQ_API_KEY ? new OpenAI({
            apiKey: process.env.GROQ_API_KEY,
            baseURL: 'https://api.groq.com/openai/v1',
        }) : null,
        model: 'mixtral-8x7b-32768',
        max_tokens: 500,
    },
    {
        name: 'Cerebras',
        enabled: !!process.env.CEREBRAS_API_KEY,
        client: process.env.CEREBRAS_API_KEY ? new OpenAI({
            apiKey: process.env.CEREBRAS_API_KEY,
            baseURL: 'https://api.cerebras.ai/v1',
        }) : null,
        model: 'llama3.1-8b',
        max_tokens: 500,
    },
    {
        name: 'Gemini',
        enabled: !!process.env.GEMINI_API_KEY,
        client: process.env.GEMINI_API_KEY ? new OpenAI({
            apiKey: process.env.GEMINI_API_KEY,
            baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai/',
        }) : null,
        model: 'gemini-2.0-flash',
        max_tokens: 500,
    },
    {
        name: 'OpenRouter-Llama',
        enabled: !!process.env.OPENROUTER_API_KEY,
        client: process.env.OPENROUTER_API_KEY ? new OpenAI({
            apiKey: process.env.OPENROUTER_API_KEY,
            baseURL: 'https://openrouter.ai/api/v1',
        }) : null,
        model: 'meta-llama/llama-3.3-70b-instruct:free',
        max_tokens: 500,
    },
    {
        name: 'OpenRouter-1',
        enabled: !!process.env.OPENROUTER_API_KEY,
        client: process.env.OPENROUTER_API_KEY ? new OpenAI({
            apiKey: process.env.OPENROUTER_API_KEY,
            baseURL: 'https://openrouter.ai/api/v1',
        }) : null,
        model: 'openrouter/auto',
        max_tokens: 500,
    },
    {
        name: 'OpenRouter-2',
        enabled: !!process.env.OPENROUTER_API_KEY,
        client: process.env.OPENROUTER_API_KEY ? new OpenAI({
            apiKey: process.env.OPENROUTER_API_KEY,
            baseURL: 'https://openrouter.ai/api/v1',
        }) : null,
        model: 'openrouter/auto',
        max_tokens: 500,
    },
];

// ─── Estado de cooldowns ──────────────────────────────────────────────────────
const providerState = {};
PROVIDERS.forEach(p => {
    providerState[p.name] = { cooldownUntil: 0, errors: 0 };
});

function proveedorDisponible(provider) {
    if (!provider.enabled) return false;
    return Date.now() > providerState[provider.name].cooldownUntil;
}

function marcarCooldown(providerName, seconds = 600) {
    providerState[providerName].cooldownUntil = Date.now() + seconds * 1000;
    providerState[providerName].errors++;
    console.warn(`⏸️  ${providerName} en cooldown por ${seconds}s`);
}

// ─── LLM con carrera paralela ─────────────────────────────────────────────────
async function llamarLLM(systemPrompt, userMessage, historial = []) {
    const disponibles = PROVIDERS.filter(proveedorDisponible);

    if (disponibles.length === 0) {
        throw new Error('Todos los proveedores están en cooldown o sin configurar.');
    }

    const historialReciente = historial.slice(-6).map(m => ({
        role: m.role,
        content: m.content,
    }));

    const mensajes = [
        { role: 'system', content: systemPrompt },
        ...historialReciente,
        { role: 'user', content: userMessage },
    ];

    const carreras = disponibles.map(provider =>
        new Promise(async (resolve, reject) => {
            try {
                console.log(`🏁 Intentando: ${provider.name}`);

                const completion = await provider.client.chat.completions.create({
                    messages: mensajes,
                    model: provider.model,
                    temperature: 0.3,
                    max_tokens: provider.max_tokens,
                });

                const respuesta = completion.choices[0]?.message?.content;
                if (!respuesta) throw new Error('Respuesta vacía');

                providerState[provider.name].errors = 0;
                providerState[provider.name].cooldownUntil = 0;

                resolve({ respuesta, provider: provider.name });

            } catch (err) {
                const msg = err.message || '';
                const status = err.status || err.response?.status;
                console.error(`❌ ${provider.name} | status: ${status} | ${msg.slice(0, 80)}`);

                if (status === 429 || msg.includes('rate limit') || msg.includes('Rate limit')) {
                    const waitMatch = msg.match(/(\d+)m(\d+(\.\d+)?)s/);
                    const waitSeconds = waitMatch
                        ? parseInt(waitMatch[1]) * 60 + parseFloat(waitMatch[2])
                        : 600;
                    marcarCooldown(provider.name, Math.ceil(waitSeconds) + 30);
                }

                if (status === 401 || msg.includes('API key')) {
                    marcarCooldown(provider.name, 3600);
                }

                reject(new Error(`${provider.name} falló`));
            }
        })
    );

    try {
        const resultado = await Promise.any(carreras);
        console.log(`✅ Ganó la carrera: ${resultado.provider}`);
        return resultado;
    } catch {
        throw new Error('Ningún proveedor pudo responder.');
    }
}

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

// ─── Embeddings locales ───────────────────────────────────────────────────────
let embedder = null;

async function obtenerEmbedding(texto) {
    if (!embedder) {
        const { pipeline } = await import('@xenova/transformers');
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    const output = await embedder(texto, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

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
            console.log(`💾 Cache embeddings: ${hits}/${manualChunks.length} chunks restaurados`);
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

// ─── Búsqueda híbrida ─────────────────────────────────────────────────────────
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
    const stopwords = new Set([
        'que', 'los', 'las', 'una', 'con', 'por', 'para', 'del', 'como',
        'son', 'sus', 'mas', 'este', 'esta', 'esto', 'ese', 'esos', 'esas',
        'hay', 'puede', 'tienen', 'tambien',
    ]);
    const palabras = pregunta
        .toLowerCase()
        .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
        .split(/\s+/)
        .filter(p => p.length > 2 && !stopwords.has(p));

    return manualChunks
        .map(chunk => {
            let score = 0;
            const chunkNorm = chunk.lower.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
            for (const palabra of palabras) {
                score += (chunkNorm.match(new RegExp(`\\b${palabra}\\b`, 'g')) || []).length * 2;
                score += (chunkNorm.match(new RegExp(palabra, 'g')) || []).length * 0.5;
            }
            if (palabras.some(p => chunk.lower.split('\n')[0].includes(p))) score += 3;
            return { ...chunk, score };
        })
        .filter(r => r.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, topN);
}

async function buscarSemantico(pregunta, topN = 10) {
    const chunksConEmbedding = manualChunks.filter(c => c.embedding);
    if (chunksConEmbedding.length === 0) return [];

    try {
        const queryEmbedding = await obtenerEmbedding(pregunta);
        return chunksConEmbedding
            .map(chunk => ({ ...chunk, score: cosineSim(queryEmbedding, chunk.embedding) }))
            .sort((a, b) => b.score - a.score)
            .slice(0, topN);
    } catch (err) {
        console.warn('⚠️  Embedding query fallido:', err.message);
        return [];
    }
}

function rrf(listaKeywords, listaSemantica, k = 60, topN = 4) {
    const scores = new Map();
    const agregar = (lista, peso) => {
        lista.forEach((item, rank) => {
            scores.set(item.id, (scores.get(item.id) || 0) + peso * (1 / (k + rank + 1)));
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
    return mejores.length === 0
        ? 'No se encontró contexto relevante en el manual.'
        : mejores.map(r => r.content).join('\n\n---\n\n');
}

// ─── Saludo simple (sin llamar al LLM) ───────────────────────────────────────
const SALUDOS = /^(hola|hello|hi|buenas|buenos días|buenas tardes|buenas noches|hey|qué tal|que tal|ey|saludos|ok|okay|gracias|vale|listo|perfecto|entendido|pero|claro|sí|si|no)[\s!?.]*$/i;

// ─── System prompt ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `Eres CELI, el asistente oficial del PLE-RD (Programa de Liderazgo Estudiantil de la República Dominicana) de CELIDER Grandeza.

QUIÉN ERES:
- Eres como un delegado veterano que conoce el programa por dentro
- Hablas en español dominicano natural, directo y sin rodeos
- Eres amigable pero vas al punto — no eres un robot ni un manual parlante

CÓMO RESPONDES:
- Da la información completa que la pregunta necesita, ni más ni menos
- Si es algo simple, responde en 1-2 oraciones
- Si es algo que necesita explicación, usa hasta 3 párrafos cortos y claros
- Explica siempre con tus propias palabras — nunca copies frases del manual
- Solo haz una pregunta de seguimiento si genuinamente abre algo útil, no por costumbre

TEMAS QUE MANEJAS:
- Estructura y funcionamiento del PLE-RD
- Reglas y procedimientos de los debates y MUN
- Roles: delegados, Mesa Directiva, facilitadores
- Mociones, puntos de orden, puntos de información, privilegio personal
- Valores, habilidades y filosofía de CELIDER
- Preparación para competencias y etapas del programa

LÍMITES:
- Si no está en tu base de datos: "Eso no lo tengo en mi base de datos, consulta a tu facilitador 😊"
- Si la pregunta no es del PLE-RD: "Solo manejo temas del PLE-RD, ¿en qué te puedo ayudar con el programa?"
- Nunca inventes información que no esté en el contexto dado

PROHIBIDO:
- Frases de relleno: "recuerda que...", "es importante destacar...", "cabe mencionar..."
- Copiar párrafos o reglas textuales del manual
- Terminar cada respuesta con una pregunta forzada
- Sonar formal o corporativo`;

// ─── Endpoints ────────────────────────────────────────────────────────────────
app.get('/api/sugerencias', (req, res) => {
    res.json({
        sugerencias: [
            '¿Qué es el PLE-RD?',
            '¿Cuáles son los requisitos de ingreso?',
            '¿Cuáles son los valores de CELIDER?',
            '¿Qué habilidades esenciales promueve el PLE-RD?',
        ],
    });
});

app.post('/api/chat', async (req, res) => {
    try {
        const { pregunta, historial = [] } = req.body;
        if (!pregunta) return res.status(400).json({ respuesta: 'Por favor, envía una pregunta.' });
        if (pregunta.length > 500) return res.status(400).json({ respuesta: 'Pregunta demasiado larga.' });

        // Saludo simple — respuesta instantánea sin LLM
        if (SALUDOS.test(pregunta.trim())) {
            return res.json({ respuesta: '¡Hola! Soy CELI, tu asistente del PLE-RD. ¿En qué puedo ayudarte hoy?' });
        }

        const contexto = await buscarHibrid(pregunta);
        const systemConContexto = contexto.includes('No se encontró')
            ? SYSTEM_PROMPT
            : `${SYSTEM_PROMPT}\n\nINFORMACIÓN DEL PLE-RD (usa solo esto para responder):\n${contexto}`;

        const { respuesta, provider } = await llamarLLM(systemConContexto, pregunta, historial);
        console.log(`✅ Respuesta generada por ${provider}`);
        res.json({ respuesta });

    } catch (error) {
        console.error('ERROR:', error.message);
        res.status(503).json({
            respuesta: 'Servicio temporalmente no disponible. Intenta en unos minutos.',
        });
    }
});

app.get('/api/health', (req, res) => {
    const chunksConEmbedding = manualChunks.filter(c => c.embedding).length;
    const estadoProveedores = PROVIDERS.map(p => ({
        name: p.name,
        enabled: p.enabled,
        disponible: proveedorDisponible(p),
        cooldownHasta: providerState[p.name].cooldownUntil > Date.now()
            ? new Date(providerState[p.name].cooldownUntil).toISOString()
            : null,
    }));

    res.json({
        status: 'ok',
        chunksLoaded: manualChunks.length,
        chunksWithEmbedding: chunksConEmbedding,
        embeddingReady: chunksConEmbedding === manualChunks.length,
        proveedores: estadoProveedores,
    });
});

// ─── Inicio ───────────────────────────────────────────────────────────────────
cargarDocumentos();
const embeddingHits = cargarCacheEmbeddings();

const proveedoresActivos = PROVIDERS.filter(p => p.enabled).map(p => p.name);
console.log(`🤖 Proveedores activos: ${proveedoresActivos.length > 0 ? proveedoresActivos.join(', ') : 'ninguno — revisa tus API keys'}`);

app.listen(PORT, async () => {
    console.log(`\n🚀 Servidor corriendo en http://localhost:${PORT}`);
    console.log(`   GET  /api/sugerencias`);
    console.log(`   POST /api/chat`);
    console.log(`   GET  /api/health\n`);

    if (embeddingHits < manualChunks.length) {
        precalcularEmbeddings().catch(console.error);
    }
});