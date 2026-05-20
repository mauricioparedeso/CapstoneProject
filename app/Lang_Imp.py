with open("API_KEY.txt", "r") as f:
    API_KEYS = [line.strip() for line in f.readlines()]
    API_KEY = API_KEYS[1]

# git reset --soft HEAD~1

"""
Correr con: python -m app.Lang_Imp
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from transformers import AutoTokenizer

import json, os, unicodedata, enchant
from datetime import datetime
from collections import defaultdict

from app.Chroma_Imp import vector_store

import re
from datetime import datetime

os.environ["TAVILY_API_KEY"] = "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"

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Eres un asistente de respuesta consolidada. Sigue estas reglas estrictas:\n"
        "1. EJECUCIÓN PARALELA: Llama a todas las herramientas necesarias (clima, archivos) en un solo turno. No vayas una por una.\n"
        "2. CONOCIMIENTO PROPIO: Resuelve matemáticas (sumas) y curiosidades (zorros) usando tu propio conocimiento inmediatamente.\n"
        "3. REGLA DE NO REINTENTO: Si una herramienta devuelve 'No se encontró información', acéptalo y no vuelvas a llamarla.\n"
        "4. RESPUESTA ÚNICA: Espera a tener todos los resultados de las herramientas para dar una única respuesta final que incluya: suma, climas, archivos y zorros.\n"
        "5. BREVEDAD: Responde con precisión quirúrgica, sin introducciones ni texto innecesario."
    )
)

PROMPT_NODE_PROMPT = SystemMessage(content=(
    "Analiza el mensaje del usuario y decide si necesita herramientas externas para ser respondido.\n"
    "Analiza el archivo memory_log.json para ver si preguntas similares fueron respondidas antes. Si el mensaje es similar a uno pasado, responde con needs_tools: false y resuelve con tu conocimiento. Si es una pregunta nueva, responde con needs_tools: true para que el planner genere tareas.\n"
    "Responde ÚNICAMENTE con un JSON con este formato, sin texto adicional:\n"
    '{"needs_tools": true} o {"needs_tools": false}\n'
))

PLANNER_PROMPT = SystemMessage(content=(
    "Eres un planificador. Analiza el mensaje del usuario y descomponlo en tareas.\n"
    "Cada tarea es una petición atómica del usuario.\n"
    "Para cada tarea, decide qué herramienta usar, y que mensaje enviar como prompt.\n\n"
    "La intention de cada tarea es entender qué información específica el usuario quiere obtener. Las opciones posibles son: ['weather', 'file listing', 'general analysis', 'detect redundancy', 'detect incorrect info', 'detect conflicts', 'detect obsolescence', 'detect outdated content', 'web_search'].\n"
    "Usa 'detect outdated content' ÚNICAMENTE para detectar fechas numéricas pasadas en documentos (por ejemplo, fechas de 2024 estando en 2026). Usa 'detect obsolescence' ÚNICAMENTE para detectar frameworks o librerías tecnológicamente desactualizadas. Usa 'detect incorrect info' para detectar datos erróneos como fechas imposibles o errores ortográficos evidentes. Usa 'general analysis' para tareas de análisis que no encajan en las otras categorías. Usa 'web_search' ÚNICAMENTE para preguntas sobre eventos recientes, noticias actuales, o información externa que NO pueda estar en los documentos de la knowledge base. NO uses 'web_search' para preguntas sobre documentos subidos, análisis de contenido, errores ortográficos o fechas en archivos.\n"
    "CASO ESPECIAL — Resumen ejecutivo: Si el usuario pregunta algo que requiere comparar el contenido de los documentos con información externa actual (por ejemplo: '¿mis documentos están desactualizados?', '¿qué tan relevante es mi contenido?'), genera DOS tasks: la primera con 'consultar_knowledge_base' y la segunda con 'buscar_en_web'. El writer consolidará ambos resultados en un resumen ejecutivo.\n"
    "Si hay 2 tareas con la misma intención, combínalas en una sola tarea con un mensaje que incluya ambas peticiones, para optimizar el uso de herramientas.\n"
    "Responde ÚNICAMENTE con un JSON array, sin texto adicional, con esta estructura:\n"
    ' {"task_name": "str", "status": "pending", "task_message": "str", "intention": "str", "used_tool": "str"}\n'
))

EXECUTOR_PROMPT = SystemMessage(content=(
    "Eres un ejecutor. Se te dará una tarea con su herramienta y argumentos.\n"
    "Ejecuta la herramienta indicada y reporta el resultado exacto.\n"
    "No agregues texto innecesario."
))

WRITER_PROMPT = SystemMessage(content=(
    "Eres un redactor final. Se te dará la lista completa de tareas con sus resultados.\n"
    "Genera una respuesta consolidada y clara para el usuario.\n"
    "Si alguna tarea tiene status 'failed', menciónalo explicando qué falló y por qué no fue respondida.\n"
    "Para tareas sin herramienta (used_tool: 'none'), resuélvelas tú mismo con tu conocimiento.\n"
    "Si el resultado de alguna tarea contiene 'URLs_FUENTES:', extrae esas URLs y colócalas al final en una sección 'Fuentes:' como lista.\n"
    "CASO ESPECIAL — Resumen ejecutivo: Si hay resultados de AMBAS tools 'consultar_knowledge_base' Y 'buscar_en_web', genera un resumen ejecutivo con esta estructura:\n"
    "1. **Estado actual de los documentos** — qué encontró en la knowledge base\n"
    "2. **Contexto externo** — qué dice la web sobre el mismo tema\n"
    "3. **Brecha identificada** — qué hay en los documentos que ya no es válido o está desactualizado\n"
    "4. **Recomendación** — qué debería actualizar el instructor\n"
    "Sé conciso y directo."
))

MEMORY_PROMPT = SystemMessage(content=(
    "Descompón el mensaje del usuario en tareas atómicas.\n\n"
    "PARA CADA TAREA:\n"
    "Busca en memory_log.json y determina si puede resolverse con memoria.\n\n"
    "CRITERIO ESTRICTO DE MEMORIA:\n"
    "SOLO marca una tarea como 'done' si:\n"
    "- Existe una tarea en memory_log con el MISMO objetivo específico\n"
    "- Y puedes COPIAR una respuesta REAL, concreta y completa desde el log\n"
    "- Y esa respuesta contiene DATOS ESPECÍFICOS (NO placeholders como 'archivo1', 'example', 'test', etc.)\n"
    "- Y corresponde al mismo contexto (misma consulta o mismos datos relevantes)\n\n"
    "SI ocurre cualquiera de estos casos:\n"
    "- La coincidencia es solo por keywords o intención general\n"
    "- La respuesta es genérica, incompleta o ambigua\n"
    "- No puedes copiar exactamente una respuesta válida del log\n\n"
    "→ ENTONCES: status = 'pending'\n\n"
    "REGLA CRÍTICA:\n"
    "NUNCA inventes datos para tareas 'done'.\n"
    "SI no puedes reutilizar memoria real → la tarea es 'pending'.\n\n"
    "FORMATO POR TAREA:\n"
    "Si está en memoria:\n"
    '{"task_name": "str", "status": "done", "message": "COPIA EXACTA DEL LOG", "intention": "str", "used_tool": "memory"}\n\n'
    "Si NO está en memoria:\n"
    '{"task_name": "str", "status": "pending", "message": "", "intention": "str", "used_tool": "none | get_weather | consultar_knowledge_base | buscar_en_web"}\n\n'
    "INTENCIONES PERMITIDAS:\n"
    "weather, file listing, general analysis, detect redundancy, detect incorrect info, detect conflicts, detect obsolescence, detect outdated content, web_search\n\n"
    "MENSAJE PARA PLANNER:\n"
    "Construye un string que contenga SOLO los nombres de las tareas con status='pending'.\n"
    "Formato:\n"
    "message_planner: tarea1, tarea2, tarea3\n\n"
    "SALIDA ESTRICTA (OBLIGATORIA):\n"
    "- Primero: un JSON array válido con TODAS las tareas\n"
    "- Luego: un carácter ';'\n"
    "- Luego: el string del mensaje para planner\n\n"
    "PROHIBIDO:\n"
    "- Markdown\n"
    "- Bloques de código\n"
    "- Texto adicional\n"
    "- Explicaciones\n\n"
    "EJEMPLO DE SALIDA CORRECTA:\n"
    '[{"task_name":"A","status":"done","message":"respuesta real","intention":"file listing","used_tool":"memory"},'
    '{"task_name":"B","status":"pending","message":"","intention":"detect incorrect info","used_tool":"consultar_knowledge_base"}];'
    'mensaje_planner: B\n'
))

QUERY_PROMPT = SystemMessage(content=(
    "Eres un generador de queries para recuperación semántica en una base vectorial.\n"
    "Tu objetivo es recuperar el fragmento ORIGINAL del documento donde aparece un texto problemático.\n\n"
    "REGLAS:\n"
    "1. La query debe parecerse lo máximo posible al texto original del documento.\n"
    "2. Incluye:\n"
    "   - La palabra problemática exacta (ej: 'caido')\n"
    "   - 5 a 15 palabras de contexto cercano (si están disponibles)\n"
    "3. NO incluyas:\n"
    "   - Explicaciones\n"
    "   - 'error ortográfico', 'problema', etc\n"
    "   - lenguaje meta\n\n"
    "4. La query debe ser una frase natural que podría existir dentro del documento.\n\n"
    "EJEMPLO:\n"
    "Entrada:\n"
    "error: 'caido'\n"
    "Salida:\n"
    "\"...se habia caido detras de la casa mientras la luz palida filtrava...\"\n\n"
    "SALIDA (JSON estricto):\n"
    "[{\"task_name\": \"str\", \"needs_suggestion\": true/false, \"suggestion_message\": \"query\"}]"
))

SPELLING_PROMPT = SystemMessage(content=(
    "Filtra errores ortográficos reales en textos.\n\n"
    "REGLAS:\n"
    "- Solo incluye palabras incorrectas o parcialmente incorrectas en español o inglés\n"
    "- Elimina:\n"
    "  * nombres propios\n"
    "  * siglas\n"
    "  * URLs\n"
    "  * código o identificadores técnicos\n"
    "  * tecnicismos\n"
    "- NO expliques nada\n"
    "- NO agregues texto adicional\n"
    "- NO repitas sources\n\n"
    "FORMATO OBLIGATORIO:\n"
    "Devuelve JSON válido:\n"
    "[{\"source\": \"str\", \"errors\": [\"str\"]}]\n\n"
    "Añadir al final, un Json con el número total de errores detectados antes de limpieza, para referencia:\n"
    "{\"total_errors\": \"int\"}\n\n"
    "REGLAS DE SALIDA:\n"
    "- Enviarás una lista de las 15 palabras más relevantes, junto con su fuente"
))

TOOL_SET = SystemMessage(
    content=(
        "Tienes acceso a las siguientes herramientas:\n"
        "1. get_weather(location): Devuelve el clima actual para una ubicación dada.\n"
        "2. consultar_knowledge_base(query): Consulta la base de datos de documentos. Si la query pide nombres, lista todos.\n"
        "3. buscar_en_web(query): Busca información actualizada en internet. Úsala para preguntas sobre eventos recientes o información externa.\n"
    )
)

MEMORY_FILE = "app/memory_log.json"

class Task(TypedDict):
    task_name: str
    task_message: str
    intention: str
    status: Literal["pending", "completed", "failed"]
    message: str
    used_tool: str
    need_suggestion: bool
    suggestion_message: str
    suggestion: str

class State(TypedDict):
    actual_node: str
    messages: Annotated[list, add_messages]
    conditional_message: str
    iterations: int
    tasks: list[Task]
    memory_tasks: list[Task]

graph = StateGraph(State)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()

def normalize_safe(data):
    if isinstance(data, dict):
        return {k: normalize_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_safe(x) for x in data]
    elif isinstance(data, str):
        return normalize_text(data)
    else:
        return data

def save_memory(state: State):
    if not os.path.exists(MEMORY_FILE):
        data = []
    else:
        try:
            with open(MEMORY_FILE, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        except json.JSONDecodeError:
            print("[WARNING] JSON corrupto, reiniciando memoria")
            data = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "node": state.get("actual_node", "unknown"),
        "iterations": state.get("iterations", 0),
        "message": normalize_text(state.get("messages", [])[-1].content if state.get("messages") else ""),
        "tasks": (state.get("tasks", []) if state.get("actual_node", "unknown") != "tool_executor_node" else []),
        "conditional_message": state.get("conditional_message"),
        "memory_tasks": state.get("memory_tasks", []),
    }

    data.append(normalize_safe(entry))

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_memory_data() -> list[SystemMessage]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    except:
        return []

    for entry in reversed(data):
        tasks = entry.get("tasks", [])
        if tasks:
            completed = [t for t in tasks if t.get("status") in ["done", "completed"]]
            if completed:
                return [SystemMessage(content=json.dumps(completed))]
    return []

def del_memory_data():
    if os.path.exists(MEMORY_FILE) and os.path.getsize(MEMORY_FILE) > 0:
        with open(MEMORY_FILE, 'w') as f:
            pass

# ====================================================================================
# TOOLS
# ====================================================================================
@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["yorkshire"]:
        return "It's cold and wet."
    else:
        return "It's warm and sunny."

@tool
def consultar_knowledge_base(query: str):
    """Consulta la base de datos de documentos. Si la query pide nombres, lista todos."""
    message = query.split("..INTENTION :")[0].strip()
    intention = query.split("..INTENTION :")[1].strip() if "..INTENTION :" in query else ""
    hallazgos = [" "]

    try:
        if intention == "file listing":
            data = vector_store.get()
            if not data or not data["metadatas"]:
                return "La base de datos está vacía."
            nombres = list(set([m.get("source") for m in data["metadatas"]]))
            return f"Archivos indexados en la base de datos: {nombres}"

        if intention == "general analysis":
            try:
                docs = vector_store.similarity_search(message, k=5)
                if not docs:
                    hallazgos.append("No se encontró contenido relevante en los documentos.")
                else:
                    fragmentos = "\n\n".join([
                        f"[Fuente: {d.metadata.get('source', 'desconocido')}]\n{d.page_content}"
                        for d in docs
                    ])
                    response = llm.invoke([
                        SystemMessage(content=(
                            "Eres un analizador de documentos académicos.\n"
                            "Se te darán fragmentos de documentos. Genera un resumen conciso que incluya:\n"
                            "- Temas principales cubiertos\n"
                            "- Nivel de profundidad del contenido\n"
                            "- Tecnologías o conceptos clave mencionados\n"
                            "- Observaciones sobre relevancia o actualidad\n"
                            "Sé conciso y directo."
                        )),
                        SystemMessage(content=f"Fragmentos a analizar:\n{fragmentos}")
                    ])
                    hallazgos.append(response.content)
            except Exception as e:
                hallazgos.append(f"Error en análisis general: {str(e)}")

        if intention == "detect redundancy":
            hallazgos.append(check_redundancy(message))

        if intention == "detect incorrect info":
            hallazgos.append(detectar_fechas_invalidas(message))
            hallazgos.append(detectar_errores_ortograficos(message))

        if intention == "detect conflicts":
            hallazgos.append("check_conflicts() aún no implementada.")

        if intention == "detect obsolescence":
            hallazgos.append("check_obsolescence() aún no implementada.")

        if intention == "detect outdated content":
            hallazgos.append(check_outdated_content(message))

        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No encontré información específica sobre eso en los documentos."

        return f"\n <br> Se encontraron los siguientes hallazgos: {' '.join(hallazgos)}"
    except Exception as e:
        return f"Error técnico: {str(e)}"

@tool
def buscar_en_web(query: str):
    """Busca información actualizada en internet usando Tavily."""
    from datetime import datetime

    DOMINIOS_EXCLUIDOS = [
        "youtube.com", "youtu.be",
        "instagram.com", "tiktok.com",
        "twitter.com", "x.com",
        "facebook.com"
    ]

    try:
        # Agregar año actual si la query no tiene fecha
        año_actual = datetime.now().year
        if str(año_actual) not in query and str(año_actual - 1) not in query:
            query = f"{query} {año_actual}"

        search = TavilySearch(max_results=5)
        response = search.invoke(query)

        # TavilySearch devuelve un dict con clave 'results'
        if isinstance(response, dict):
            results = response.get('results', [])
        elif isinstance(response, list):
            results = response
        else:
            return str(response)

        # Filtrar fuentes no textuales
        results = [
            r for r in results
            if isinstance(r, dict) and not any(
                dominio in r.get('url', '')
                for dominio in DOMINIOS_EXCLUIDOS
            )
        ]

        if not results:
            return "No se encontraron resultados en fuentes textuales."

        output = []
        urls = []
        for i, r in enumerate(results, 1):
            url = r.get('url', '')
            content = r.get('content', '')
            if url:
                urls.append(url)
            output.append(f"[Resultado {i}]\nContenido: {content}")

        resultado = "\n\n---\n\n".join(output)
        if urls:
            resultado += f"\n\nURLs_FUENTES: {' | '.join(urls)}"
        return resultado

    except Exception as e:
        return f"Error en búsqueda web: {str(e)}"

@tool
def generar_sugerencias(query: str):
    """A partir de RAG, genera una sugerencia para abordar los problemas encontrados."""
    return "Herramienta de generación de sugerencias aún no implementada."

@tool
def memory_tool(query: str):
    """Herramienta para acceder a la memoria de interacciones pasadas."""
    return "Función de memoria aún no implementada."

def is_valid_word(word):
    if len(word) <= 2:
        return False
    if re.search(r"\d", word):
        return False
    if word.isupper():
        return False
    if re.search(r"[a-zA-Z]+\d+[a-zA-Z]*", word):
        return False
    return True

def detectar_errores_ortograficos(query: str):
    """Detecta errores ortográficos reales en la knowledge base de forma robusta."""
    message = query.split("..INTENTION :")[0].strip()
    dictionary = enchant.Dict("es")

    try:
        docs = vector_store.similarity_search("texto general", k=20)
        grouped_errors = defaultdict(set)

        for d in docs:
            text = d.page_content
            source = d.metadata.get("source", "desconocido")
            words = re.findall(r"\b\w+\b", text.lower())
            for w in words:
                w_norm = normalize_text(w)
                if not is_valid_word(w_norm):
                    continue
                if not dictionary.check(w_norm):
                    grouped_errors[source].add(w_norm)

        cleaned_input = [
            {"source": src, "errors": list(errors)[:15]}
            for src, errors in grouped_errors.items()
            if errors
        ]

        if not cleaned_input:
            return "No se encontraron errores ortográficos en los fragmentos analizados."

        response = llm.invoke([
            SPELLING_PROMPT,
            SystemMessage(content=f"Fragmentos a revisar:\n{cleaned_input}"),
            SystemMessage(content=f"Número total de errores detectados antes de limpieza: {sum(len(e['errors']) for e in cleaned_input)}")
        ])

        try:
            parsed = json.loads(response.content)
            final = []
            seen_sources = set()
            for item in parsed:
                source = item.get("source")
                errors = item.get("errors", [])
                if not source or source in seen_sources:
                    continue
                final.append({"source": source, "errors": list(set(errors))[:15]})
                seen_sources.add(source)

            if not final:
                return "No se encontraron errores ortográficos reales después de filtrado."

            return "\n".join([
                f"- fuente: {item['source']}\n  errores: {', '.join(item['errors'])}"
                for item in final
            ])
        except Exception:
            if not cleaned_input:
                return "Error al procesar errores ortográficos."
            return "\n".join([
                f"- fuente: {item['source']}\n  errores: {', '.join(item['errors'])}"
                for item in cleaned_input
            ])
    except Exception as e:
        return f"Error técnico en detector ortográfico: {str(e)}"

def detectar_fechas_invalidas(query: str):
    """Detecta fechas numéricas inválidas en fragmentos relevantes de la knowledge base."""
    message = query.split("..INTENTION :")[0].strip()

    try:
        docs = vector_store.similarity_search("all", k=20)
        if not docs:
            return "No encontré fragmentos relevantes para revisar fechas."

        pattern_dmy = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
        pattern_ymd = re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b")
        hallazgos = []

        for d in docs:
            source = d.metadata.get("source", "desconocido")
            text = d.page_content
            for match in pattern_dmy.finditer(text):
                day, month, year = map(int, match.groups())
                try:
                    datetime(year, month, day)
                except ValueError:
                    hallazgos.append(f"- fuente: {source}\n  fecha_detectada: {match.group(0)}\n  problema: fecha calendario inválida")
            for match in pattern_ymd.finditer(text):
                year, month, day = map(int, match.groups())
                try:
                    datetime(year, month, day)
                except ValueError:
                    hallazgos.append(f"- fuente: {source}\n  fecha_detectada: {match.group(0)}\n  problema: fecha calendario inválida")

        if not hallazgos:
            return "No encontré fechas numéricas inválidas en los fragmentos analizados."
        return "\n\n".join(hallazgos[:15])
    except Exception as e:
        return f"Error técnico en detector de fechas: {str(e)}"

def check_redundancy(query: str):
    """Detecta información redundante en la knowledge base."""
    try:
        docs = vector_store.similarity_search(query, k=20)
        if not docs:
            return "No encontré fragmentos relevantes para revisar."

        contexto = "\n\n".join([
            f"[Fuente: {d.metadata.get('source', 'desconocido')}]\n{d.page_content}"
            for d in docs
        ])

        response = llm.invoke([
            SystemMessage(content=(
                "Eres un detector de redundancias en documentos académicos.\n"
                "Se te darán fragmentos de distintos documentos.\n"
                "Identifica fragmentos que contengan información igual o muy similar.\n"
                "REGLAS:\n"
                "- Solo reporta redundancias reales, no coincidencias temáticas generales\n"
                "- Indica la fuente de cada fragmento redundante\n"
                "- Sé conciso\n"
                "FORMATO:\n"
                "[{\"fuente_1\": \"str\", \"fuente_2\": \"str\", \"descripcion\": \"str\"}]"
            )),
            SystemMessage(content=f"Fragmentos a analizar:\n{contexto}")
        ])
        return response.content
    except Exception as e:
        return f"Error técnico en detector de redundancias: {str(e)}"

def check_conflicts(query: str):
    return "Función de detección de conflictos aún no implementada."

def check_obsolescence(query: str):
    return "Función de detección de obsolescencia aún no implementada."

def check_outdated_content(query: str):
    """Detecta contenido con fechas pasadas (más de 1 año) en la knowledge base."""
    try:
        docs = vector_store.similarity_search(query, k=20)
        if not docs:
            return "No encontré fragmentos relevantes para revisar."

        pattern_dmy = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
        pattern_ymd = re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b")
        pattern_year = re.compile(r"\b(20\d{2})\b")
        umbral = datetime.now().replace(year=datetime.now().year - 1)
        hallazgos = []

        for d in docs:
            source = d.metadata.get("source", "desconocido")
            text = d.page_content
            for match in pattern_dmy.finditer(text):
                day, month, year = map(int, match.groups())
                try:
                    fecha = datetime(year, month, day)
                    if fecha < umbral:
                        hallazgos.append(f"- fuente: {source}\n  fecha_detectada: {match.group(0)}\n  problema: fecha desactualizada (más de 1 año)")
                except ValueError:
                    pass
            for match in pattern_ymd.finditer(text):
                year, month, day = map(int, match.groups())
                try:
                    fecha = datetime(year, month, day)
                    if fecha < umbral:
                        hallazgos.append(f"- fuente: {source}\n  fecha_detectada: {match.group(0)}\n  problema: fecha desactualizada (más de 1 año)")
                except ValueError:
                    pass
            for match in pattern_year.finditer(text):
                year = int(match.group(1))
                try:
                    fecha = datetime(year, 1, 1)
                    if fecha < umbral:
                        hallazgos.append(f"- fuente: {source}\n  año_detectado: {match.group(0)}\n  problema: año desactualizado (más de 1 año)")
                except ValueError:
                    pass

        if not hallazgos:
            return "No se encontró contenido desactualizado en los fragmentos analizados."
        return "\n\n".join(hallazgos[:15])
    except Exception as e:
        return f"Error técnico en detector de contenido desactualizado: {str(e)}"

def check_general_analysis(query: str):
    return "Función de análisis general aún no implementada."

TOOLS_MAP = {
    "get_weather": get_weather,
    "consultar_knowledge_base": consultar_knowledge_base,
    "buscar_en_web": buscar_en_web,
}

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=API_KEY
)

tools = [get_weather, consultar_knowledge_base, buscar_en_web]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

graph.add_node("tool_node", tool_node)

# ====================================================================================
# NODOS
# ====================================================================================
def prompt_node(state: State) -> State:
    """Decide si el mensaje necesita tools o puede responderse con memoria."""
    memory_data = get_memory_data()
    response = llm.invoke([PROMPT_NODE_PROMPT] + memory_data + [TOOL_SET] + state["messages"])
    del_memory_data()

    try:
        parsed = json.loads(response.content)
        need_tools = "yes" if parsed.get("needs_tools") else "no"
    except Exception:
        need_tools = "no"

    state["actual_node"] = "prompt_node"
    save_memory(state)
    return {
        "messages": [response],
        "conditional_message": need_tools,
        "iterations": state.get("iterations", 0) + 1,
    }

def planner_node(state: State) -> State:
    """Descompone el mensaje en tasks y las enlista en el estado."""
    response = llm.invoke([PLANNER_PROMPT] + [TOOL_SET] + state["messages"])

    try:
        raw_tasks = json.loads(response.content)
        tasks: list[Task] = [
            {
                "task_name": t["task_name"],
                "task_message": t["task_message"],
                "intention": t["intention"],
                "status": "pending",
                "message": "",
                "used_tool": t["used_tool"],
            }
            for t in raw_tasks
        ]
    except Exception as e:
        tasks = [{
            "task_name": "tarea_general",
            "task_message": "No se pudo parsear el plan",
            "intention": "sin intencion",
            "status": "failed",
            "message": f"Error al planificar: {e}",
            "used_tool": "none",
        }]

    state["actual_node"] = "planner_node"
    save_memory(state)
    return {
        "messages": [response],
        "tasks": tasks,
    }

def tool_executor_node(state: State) -> State:
    """Toma la primera task pendiente, ejecuta su tool y actualiza su estado."""
    tasks = list(state["tasks"])

    pending_index = next(
        (i for i, t in enumerate(tasks) if t["status"] == "pending"),
        None
    )

    if pending_index is None:
        return {"tasks": tasks, "conditional_message": "done"}

    task = tasks[pending_index]

    if task["used_tool"] == "none":
        tasks[pending_index] = {**task, "status": "completed", "message": "Resuelta por conocimiento propio"}
    else:
        try:
            tool_fn = TOOLS_MAP[task["used_tool"]]
            resultado = tool_fn.invoke(f"{task['task_message']} ..INTENTION :{task['intention']}")
            tasks[pending_index] = {**task, "status": "completed", "message": resultado}
        except Exception as e:
            tasks[pending_index] = {**task, "status": "failed", "message": f"Error: {e}"}

    hay_pending = any(t["status"] == "pending" for t in tasks)

    return {
        "tasks": tasks,
        "conditional_message": "pending" if hay_pending else "done",
        "iterations": state.get("iterations", 0) + 1,
    }

def suggest_executor_node(state: State) -> State:
    pass

def writer_node(state: State) -> State:
    """Genera la respuesta final iterando todas las tasks."""
    tasks = state.get("tasks", [])
    user_message = state["messages"][0] if state["messages"] else "Sin mensaje"
    user_msg_content = user_message.content if hasattr(user_message, 'content') else str(user_message)

    if not tasks:
        response = llm.invoke([
            WRITER_PROMPT,
            SystemMessage(content=f"Mensaje del usuario: {user_msg_content}")
        ])
        print(f"Uso de tokens en writer: {response.response_metadata.get('token_usage', {})}")
        return {"messages": [response]}

    resumen = "\n".join([
        f"- [{t['status'].upper()}] {t['task_name']}: {t['message'] or t['task_message']}"
        for t in tasks
    ])

    writer_context = SystemMessage(content=(
        f"{WRITER_PROMPT.content}\n\n"
        f"Mensaje original del usuario: {user_msg_content}\n\n"
        f"Lista de tareas ejecutadas:\n{resumen}"
    ))

    response = llm.invoke([writer_context])

    if not response.content or response.content.strip() == "":
        print("[WARNING] Writer retornó respuesta vacía, usando fallback")
        return {"messages": [SystemMessage(content=f"Respuesta: {resumen}")]}

    print(f"Uso de tokens en writer: {response.response_metadata.get('token_usage', {})}")
    state["actual_node"] = "writer_node"
    save_memory(state)
    return {"messages": [response]}

def memory_node(state: State) -> State:
    """Decide si puede resolver con memoria o redirige al planner."""
    memory_data = get_memory_data()
    response = llm.invoke([MEMORY_PROMPT] + memory_data + state["messages"])

    try:
        raw_tasks = json.loads(response.content.split(";")[0])
        tasks: list[Task] = [
            {
                "task_name": t["task_name"],
                "task_message": t["task_message"],
                "intention": t["intention"],
                "status": "pending",
                "message": "",
                "used_tool": t.get("used_tool", "none"),
            }
            for t in raw_tasks if "status" == "completed"
        ]
    except Exception as e:
        print(f"[WARNING] Error al parsear memoria: {e}")
        tasks = []

    state["actual_node"] = "memory_node"
    save_memory(state)

    mensaje = response.content.split(";")[-1].strip()[16:]
    return {
        "messages": [mensaje],
        "memory_tasks": tasks,
        "conditional_message": "yes"
    }

def query_node(state: State) -> State:
    tasks = state.get("tasks", [])

    TASKS_CONTEXT = "TAREAS:\n" + "\n".join([
        f"- task_name: {t['task_name']}\n"
        f"  status: {t['status']}\n"
        f"  resultado: {t['message']}\n"
        for t in tasks
    ])

    response = llm.invoke([QUERY_PROMPT] + [TASKS_CONTEXT])

    try:
        suggestions = json.loads(response.content)
        suggestion_map = {s["task_name"]: s for s in suggestions}
        new_tasks = []
        for t in tasks:
            s = suggestion_map.get(t["task_name"], {})
            new_tasks.append({
                **t,
                "need_suggestion": s.get("needs_suggestion", False),
                "suggestion_message": s.get("suggestion_message", "")
            })
    except Exception as e:
        new_tasks = [
            {**t, "need_suggestion": False, "suggestion_message": ""}
            for t in tasks
        ]

    return {
        "tasks": new_tasks,
        "messages": [response]
    }

# ====================================================================================
# EDGES
# ====================================================================================
def after_prompt(state: State) -> Literal["planner_node", "memory_node"]:
    return "planner_node" if state["conditional_message"] == "yes" else "memory_node"

def after_memory(state: State) -> Literal["planner_node", "writer_node"]:
    return "planner_node" if state["conditional_message"] == "yes" else "writer_node"

def after_executor(state: State) -> Literal["tool_executor_node", "query_node"]:
    return "tool_executor_node" if state["conditional_message"] == "pending" else "query_node"

graph = StateGraph(State)

graph.add_node("prompt_node",        prompt_node)
graph.add_node("planner_node",       planner_node)
graph.add_node("tool_executor_node", tool_executor_node)
graph.add_node("writer_node",        writer_node)
graph.add_node("memory_node",        memory_node)
graph.add_node("query_node",         query_node)
graph.set_entry_point("prompt_node")

graph.add_conditional_edges("prompt_node",        after_prompt)
graph.add_edge(              "planner_node",       "tool_executor_node")
graph.add_conditional_edges("memory_node",         after_memory)
graph.add_conditional_edges("tool_executor_node",  after_executor)
graph.add_edge(              "query_node",         "writer_node")
graph.add_edge(              "writer_node",        "__end__")

APP = graph.compile()

if __name__ == "__main__":
    print("=================================================\nINICIO DE LA COMPILACION\n=================================================")
    new_state = APP.invoke({"messages": ["dame la suma del 1 al 10, el clima en yorkshire, y el clima en bogota, y los nombres de los archivos de mi knowledge base, revisa que errores de fecha y ortografia tienen, y dime porque los zorros articos comen zapatos"]})
    print(new_state["messages"][-1].content)
    print("=================================================\nFIN DE LA COMPILACION\n=================================================")

    print("=================================================\nINICIO DEL GRAFO\n=================================================")
    from langchain_core.runnables.graph import MermaidDrawMethod
    print(APP.get_graph().draw_mermaid())
    print("=================================================\nFIN DEL GRAFO\n=================================================")