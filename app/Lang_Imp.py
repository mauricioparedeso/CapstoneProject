with open("API_KEY.txt", "r") as f:
    API_KEYS = [line.strip() for line in f.readlines()]
    API_KEY = API_KEYS[0]
    LANGFUSE_SECRET_KEY=API_KEYS[10]
    LANGFUSE_PUBLIC_KEY=API_KEYS[11]
    LANGFUSE_BASE_URL=API_KEYS[12]

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
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
#from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from transformers import AutoTokenizer

import json, os, unicodedata, enchant
from datetime import datetime
from collections import defaultdict

from uuid_utils import uuid4

from app.Chroma_Imp import vector_store
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langfuse import observe

import re
from datetime import datetime

from difflib import SequenceMatcher
import traceback

os.environ["TAVILY_API_KEY"] = "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"

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Eres un asistente de respuesta consolidada. Sigue estas reglas estrictas:<br>\n"
        "1. EJECUCIÓN PARALELA: Llama a todas las herramientas necesarias (clima, archivos) en un solo turno. No vayas una por una.<br>\n"
        "2. CONOCIMIENTO PROPIO: Resuelve matemáticas (sumas) y curiosidades (zorros) usando tu propio conocimiento inmediatamente.<br>\n"
        "3. REGLA DE NO REINTENTO: Si una herramienta devuelve 'No se encontró información', acéptalo y no vuelvas a llamarla.<br>\n"
        "4. RESPUESTA ÚNICA: Espera a tener todos los resultados de las herramientas para dar una única respuesta final que incluya: suma, climas, archivos y zorros.<br>\n"
        "5. BREVEDAD: Responde con precisión quirúrgica, sin introducciones ni texto innecesario."
    )
)

PROMPT_NODE_PROMPT = SystemMessage(content=(
    "Analiza el mensaje del usuario y decide si necesita herramientas externas para ser respondido.<br>\n"
    "Analiza el archivo memory_log.json para ver si preguntas similares fueron respondidas antes. Si el mensaje es similar a uno pasado, responde con needs_tools: false y resuelve con tu conocimiento. Si es una pregunta nueva, responde con needs_tools: true para que el planner genere tareas.<br>\n"
    "Responde ÚNICAMENTE con un JSON con este formato, sin texto adicional:<br>\n"
    '{"needs_tools": true} o {"needs_tools": false}<br>\n'
))

PLANNER_PROMPT = SystemMessage(content=(
    "Eres un planificador. Analiza el mensaje del usuario y descomponlo en tareas.<br>\n"
    "Cada tarea es una petición atómica del usuario.<br>\n"
    "Para cada tarea, decide qué herramienta usar, y que mensaje enviar como prompt.<br>\n<br>\n"
    "La intention de cada tarea es entender qué información específica el usuario quiere obtener. Las opciones posibles son: ['weather', 'file listing', 'general analysis', 'detect redundancy', 'detect incorrect info', 'detect conflicts', 'detect obsolescence', 'detect outdated content', 'web_search'].<br>\n"
    "Usa 'detect outdated content' ÚNICAMENTE para detectar fechas numéricas pasadas en documentos (por ejemplo, fechas de 2024 estando en 2026). Usa 'detect obsolescence' ÚNICAMENTE para detectar frameworks o librerías tecnológicamente desactualizadas. Usa 'detect incorrect info' para detectar datos erróneos como fechas imposibles o errores ortográficos evidentes. Usa 'general analysis' para tareas de análisis que no encajan en las otras categorías. Usa 'web_search' ÚNICAMENTE para preguntas sobre eventos recientes, noticias actuales, o información externa que NO pueda estar en los documentos de la knowledge base. NO uses 'web_search' para preguntas sobre documentos subidos, análisis de contenido, errores ortográficos o fechas en archivos.<br>\n"
    "CASO ESPECIAL — Resumen ejecutivo: Si el usuario pregunta algo que requiere comparar el contenido de los documentos con información externa actual (por ejemplo: '¿mis documentos están desactualizados?', '¿qué tan relevante es mi contenido?'), genera DOS tasks: la primera con 'consultar_knowledge_base' y la segunda con 'buscar_en_web'. El writer consolidará ambos resultados en un resumen ejecutivo.<br>\n"
    "Si hay 2 tareas con la misma intención, combínalas en una sola tarea con un mensaje que incluya ambas peticiones, para optimizar el uso de herramientas.<br>\n"
    "Responde ÚNICAMENTE con un JSON array, sin texto adicional, con esta estructura:<br>\n"
    ' {"task_name": "str", "status": "pending", "task_message": "str", "intention": "str", "used_tool": "str"}<br>\n'
))

EXECUTOR_PROMPT = SystemMessage(content=(
    "Eres un ejecutor. Se te dará una tarea con su herramienta y argumentos.<br>\n"
    "Ejecuta la herramienta indicada y reporta el resultado exacto.<br>\n"
    "No agregues texto innecesario."
))

WRITER_PROMPT = SystemMessage(content=(
    "Eres un analizador y redactor final.<br>\n"
    "Recibirás una lista completa de tareas ejecutadas, incluyendo sus resultados, estado y herramienta utilizada.<br>\n<br>\n"
    "Tu objetivo es generar un resumen general claro y útil para el usuario.<br>\n<br>\n"
    "CASO ESPECIAL — Resumen ejecutivo: Si hay resultados de AMBAS tools 'consultar_knowledge_base' Y 'buscar_en_web', genera un resumen ejecutivo con esta estructura:<br>\n"
    "1. **Estado actual de los documentos** — qué encontró en la knowledge base<br>\n"
    "2. **Contexto externo** — qué dice la web sobre el mismo tema<br>\n"
    "3. **Brecha identificada** — qué hay en los documentos que ya no es válido o está desactualizado<br>\n"
    "4. **Recomendación** — qué debería actualizar el instructor<br>\n"
    "Si el resultado de alguna tarea contiene 'URLs_FUENTES:', extrae esas URLs y colócalas al final en una sección 'Fuentes:' como lista.<br>\n<br>\n"
    #=======================CHECK_CONCLFICTS=======================================================
    "CASO SKIPPED — check_conflicts skipped: Si el resultado de una tarea contiene 'status: skipped' y 'reason: files_not_found', NO reportes ningún conflicto.<br>\n"
    "Informa al usuario que los archivos solicitados no están indexados en la base de datos y lista los archivos disponibles. Usa el campo 'writer_hint' como base para redactar.<br>\n"
    "CASO CONFLICTOS — check_conflicts con resultados: Si el resultado contiene 'status: success' y 'conflicts_found > 0':<br>\n"
    "1. Extrae cada conflicto del array 'conflicts'<br>\n"
    "2. Para CADA conflicto, redacta:<br>\n"
    "   - El concepto en conflicto<br>\n"
    "   - Qué dice el archivo A: [fragmento_a]<br>\n"
    "   - Qué dice el archivo B: [fragmento_b]<br>\n"
    "   - La diferencia específica (de descripcion_conflicto)<br>\n"
    "   - El nivel de confianza<br>\n"
    #=======================CHECK_CONCLFICTS=======================================================
    "CASO NORMAL — para cualquier otra combinación de tools:<br>\n"
    "- Resumir brevemente qué se analizó.<br>\n"
    "- Indicar qué herramientas se utilizaron.<br>\n"
    "- Contar cuántos errores, inconsistencias o problemas fueron detectados por cada herramienta o tarea.<br>\n"
    "- Mencionar si alguna tarea falló, explicando de forma breve qué ocurrió.<br>\n"
    "- Para tareas con used_tool='none', resolverlas usando tu propio conocimiento.<br>\n"
    "- Destacar hallazgos importantes o repetitivos.<br>\n"
    "- Dar recomendaciones simples y prácticas basadas en los resultados.<br>\n<br>\n"
    "Las recomendaciones deben ser cortas, accionables y fáciles de entender.<br>\n"
    "Mantén un tono profesional, claro y conciso.<br>\n"
    "No inventes errores que no aparezcan en los resultados.<br>\n"
    "Si no se detectaron problemas, indícalo explícitamente."
))

MEMORY_PROMPT = SystemMessage(content=( 
    "Descompón el mensaje del usuario en tareas atómicas.<br>\n<br>\n" 
    "PARA CADA TAREA:<br>\n" "Busca en memory_log.json y determina si puede resolverse con memoria.<br>\n<br>\n" 
    "CRITERIO ESTRICTO DE MEMORIA:<br>\n" "SOLO marca una tarea como 'done' si:<br>\n" 
    "- Existe una tarea en memory_log con el MISMO objetivo específico<br>\n" 
    "- Y puedes COPIAR una respuesta REAL, concreta y completa desde el log<br>\n" 
    "- Y esa respuesta contiene DATOS ESPECÍFICOS (NO placeholders como 'archivo1', 'example', 'test', etc.)<br>\n" 
    "- Y corresponde al mismo contexto (misma consulta o mismos datos relevantes)<br>\n<br>\n" 
    "SI ocurre cualquiera de estos casos:<br>\n" 
    "- La coincidencia es solo por keywords o intención general<br>\n" 
    "- La respuesta es genérica, incompleta o ambigua<br>\n" 
    "- No puedes copiar exactamente una respuesta válida del log<br>\n<br>\n" 
    "→ ENTONCES: status = 'pending'<br>\n<br>\n" 
    "REGLA CRÍTICA:<br>\n" 
    "NUNCA inventes datos para tareas 'done'.<br>\n" 
    "SI no puedes reutilizar memoria real → la tarea es 'pending'.<br>\n<br>\n" 
    "FORMATO POR TAREA:<br>\n"
    "Si está en memoria:<br>\n" 
    '{"task_name": "str", "status": "done", "message": "COPIA EXACTA DEL LOG", "intention": "str", "used_tool": "memory"}<br>\n<br>\n' 
    "Si NO está en memoria:<br>\n" 
    '{"task_name": "str", "status": "pending", "message": "", "intention": "str", "used_tool": "none | get_weather | consultar_knowledge_base | buscar_en_web"}<br>\n<br>\n' 
    "INTENCIONES PERMITIDAS:<br>\n" "weather, file listing, general analysis, detect redundancy, detect incorrect info, detect conflicts, detect obsolescence, detect outdated content<br>\n<br>\n" 
    "MENSAJE PARA PLANNER:<br>\n" 
    "Construye un string que contenga SOLO los nombres de las tareas con status='pending'.<br>\n" 
    "Formato:<br>\n" 
    "message_planner: tarea1, tarea2, tarea3<br>\n<br>\n" 
    "SALIDA ESTRICTA (OBLIGATORIA):<br>\n" 
    "- Primero: un JSON array válido con TODAS las tareas<br>\n" 
    "- Luego: un carácter ';'<br>\n" 
    "- Luego: el string del mensaje para planner<br>\n<br>\n" 
    "PROHIBIDO:<br>\n" 
    "- Markdown<br>\n" 
    "- Bloques de código<br>\n" 
    "- Texto adicional<br>\n" 
    "- Explicaciones<br>\n<br>\n" 
    "EJEMPLO DE SALIDA CORRECTA:<br>\n" 
    '[{"task_name":"A","status":"done","message":"respuesta real","intention":"file listing","used_tool":"memory"},' 
    '{"task_name":"B","status":"pending","message":"","intention":"detect incorrect info","used_tool":"consultar_knowledge_base"}];' 'mensaje_planner: B<br>\n' ))



QUERY_PROMPT = SystemMessage(content=(
    "Eres un generador de queries para recuperación semántica en una base vectorial.<br>\n"
    "Tu objetivo es recuperar el fragmento ORIGINAL del documento donde aparece un texto problemático.<br>\n<br>\n"

    "REGLAS:<br>\n"
    "1. La query debe parecerse lo máximo posible al texto original del documento.<br>\n"
    "2. Incluye:<br>\n"
    "   - La palabra problemática exacta (ej: 'caido')<br>\n"
    "   - 5 a 15 palabras de contexto cercano (si están disponibles)<br>\n"
    "3. NO incluyas:<br>\n"
    "   - Explicaciones<br>\n"
    "   - 'error ortográfico', 'problema', etc<br>\n"
    "   - lenguaje meta<br>\n<br>\n"

    "4. La query debe ser una frase natural que podría existir dentro del documento.<br>\n<br>\n"

    "EJEMPLO:<br>\n"
    "Entrada:<br>\n"
    "error: 'caido'<br>\n"
    "Salida:<br>\n"
    "\"...se habia caido detras de la casa mientras la luz palida filtrava...<br>\n<br>\n"

    "SALIDA (JSON estricto):<br>\n"
    "[{\"task_name\": \"str\", \"needs_suggestion\": true/false, \"suggestion_message\": \"query\"}]"
))

SPELLING_PROMPT = SystemMessage(content=(
    "Filtra errores ortográficos reales en textos.<br>\n<br>\n"

    "REGLAS:<br>\n"
    "- Solo incluye palabras incorrectas o parcialmente incorrectas en español o inglés<br>\n"
    "- Elimina:<br>\n"
    "  * nombres propios<br>\n"
    "  * siglas<br>\n"
    "  * URLs<br>\n"
    "  * código o identificadores técnicos<br>\n"
    "  * tecnicismos<br>\n"
    "- NO expliques nada<br>\n"
    "- NO agregues texto adicional<br>\n"
    "- NO repitas sources<br>\n<br>\n"

    "FORMATO OBLIGATORIO:<br>\n"
    "Devuelve JSON válido:<br>\n"
    "[{\"source\": \"str\", \"errors\": [\"str\"]}]<br>\n<br>\n"
    "Añadir al final, un Json con el número total de errores detectados antes de limpieza, para referencia:<br>\n"
    "{\"total_errors\": \"int\"}<br>\n<br>\n"

    "REGLAS DE SALIDA:<br>\n"
    "- Enviarás una lista de las 15 palabras más relevantes, junto con su fuente"
))



TOOL_SET = SystemMessage(
    content=(
        "Tienes acceso a las siguientes herramientas:<br>\n"
        "1. get_weather(location): Devuelve el clima actual para una ubicación dada.<br>\n"
        "2. consultar_knowledge_base(query): Consulta la base de datos de documentos. Si la query pide nombres, lista todos.<br>\n"
        "3. buscar_en_web(query): Busca información actualizada en internet. Úsala para preguntas sobre eventos recientes, noticias actuales o información externa.<br>\n"
    )
)

MEMORY_FILE = "app/memory_log.json"



#====================================================================================
# Definimos el estado de nuestra aplicación como un diccionario de variables
# El estado se define como en que nodo está el grafo y que mensajes se han generado hasta el momento.
# Tambien iterations, que es el número de veces que se ha ejecutado el grafo, para evitar loops infinitos.

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
    conditional_message: str  # Para trackear el mensaje que decide a qué nodo ir
    iterations: int
    tasks: list[Task]  # Para trackear tareas asincronas
    memory_tasks: list[Task]  # Para trackear tareas resueltas con memoria

#====================================================================================
# Definimos funciones importantes
#====================================================================================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()

def normalize_safe(data):
    if isinstance(data, dict):
        return {
            k: normalize_safe(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [normalize_safe(x) for x in data]
    elif isinstance(data, str):
        return normalize_text(data)
    else:
        return data

def save_memory(state: State):
    if not os.path.exists(MEMORY_FILE):
        data = ["MEMORY DATA:"]
    else:
        try:
            with open(MEMORY_FILE, "r") as f:
                content = f.read().strip()

                if content == "":
                    data = []  # archivo vacío
                else:
                    data = json.loads(content)

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
            completed = [
                t for t in tasks
                if t.get("status") in ["done", "completed"]
            ]
            if completed:
                return [SystemMessage(content=json.dumps(completed))]

    return []

def del_memory_data():
    if os.path.exists(MEMORY_FILE) and os.path.getsize(MEMORY_FILE) > 0:
        with open(MEMORY_FILE, 'w') as f:
            pass



#====================================================================================
# Definimos herramientas, osease, funciones que pueden ser llamadas por un toolnode
#====================================================================================
@tool
@observe(name="get_weather")
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["yorkshire"]: 
        return "It's cold and wet."
    else:
        return "It's warm and sunny."

@tool
@observe(name="consultar_knowledge_base")
def consultar_knowledge_base(query: str):
    """Consulta la base de datos de documentos. Si la query pide nombres, lista todos."""
    #['weather', 'file listing', 'general analysis', 'detect redundancy', 'detect incorrect info', 'detect conflicts', 'detect obsolescence', 'detect outdated content']
    message = query.split("..INTENTION :")[0].strip()
    intention = query.split("..INTENTION :")[1].strip() if "..INTENTION :" in query else ""
    
    hallazgos = [" "]

    try:
        if intention == "file listing":
            # Si la pregunta es sobre 'nombres' o 'lista', traemos todo lo que haya
            # .get() trae los registros sin filtrar por similitud de texto
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
            resultado = check_conflicts(message)
            # check_conflicts retorna un dict, convertir a string JSON
            if isinstance(resultado, dict):
                hallazgos.append(json.dumps(resultado, indent=2, ensure_ascii=False))
            else:
                hallazgos.append(str(resultado))

        if intention == "detect obsolescence":
            hallazgos.append("check_obsolescence() aún no implementada.")

        if intention == "detect outdated content":
            hallazgos.append(check_outdated_content(message))

        # Búsqueda normal de contenido
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No encontré información específica sobre eso en los documentos."
        
        separator = '<br>\n'
        return f"<br>\n <br>\n Se encontraron los siguientes hallazgos: {separator.join(hallazgos)}"
    except Exception as e:
        return f"Error técnico: {str(e)}"
    
@tool
def generar_sugerencias(query: str):
    """A partir de RAG, genera una sugerencia para abordar los problemas encontrados."""
    return "Herramienta de generación de sugerencias aún no implementada."
    
@tool
def memory_tool(query: str): #Reemplaza al memory node
    """Herramienta para acceder a la memoria de interacciones pasadas."""
    # Esta función podría ser llamada por el planner si el prompt_node o memory_node indican que es necesario usar memoria.
    # El query podría incluir información sobre qué tipo de información se busca en la memoria (ej: "¿He respondido algo similar antes?").
    # La función podría analizar el memory_log.json y devolver información relevante al LLM para que la tenga en cuenta al generar la respuesta.
    return "Función de memoria aún no implementada."

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
        año_actual = datetime.now().year
        if str(año_actual) not in query and str(año_actual - 1) not in query:
            query = f"{query} {año_actual}"
        search = TavilySearch(max_results=5)
        response = search.invoke(query)
        if isinstance(response, dict):
            results = response.get("results", [])
        elif isinstance(response, list):
            results = response
        else:
            return str(response)
        results = [r for r in results if isinstance(r, dict) and not any(d in r.get("url", "") for d in DOMINIOS_EXCLUIDOS)]
        if not results:
            return "No se encontraron resultados en fuentes textuales."
        output = []
        urls = []
        for i, r in enumerate(results, 1):
            url = r.get("url", "")
            content = r.get("content", "")
            if url:
                urls.append(url)
            output.append(f"[Resultado {i}]\nContenido: {content}")
        resultado = "\n\n---\n\n".join(output)
        if urls:
            resultado += "\n\nURLs_FUENTES: " + " | ".join(urls)
        return resultado
    except Exception as e:
        return f"Error en búsqueda web: {str(e)}"


def is_valid_word(word):
    # longitud mínima
    if len(word) <= 2:
        return False

    # números o alfanuméricos
    if re.search(r"\d", word):
        return False

    # todo mayúsculas o códigos raros
    if word.isupper():
        return False

    # mezcla rara tipo computaciónma03439
    if re.search(r"[a-zA-Z]+\d+[a-zA-Z]*", word):
        return False

    return True

def detectar_errores_ortograficos(query: str):
    """Detecta errores ortográficos reales en la knowledge base de forma robusta."""

    message = query.split("..INTENTION :")[0].strip()
    dictionary = enchant.Dict("es")

    try:
        docs = vector_store.similarity_search("texto general", k=50)

        grouped_errors = defaultdict(set)

        # 1. PRE-FILTRADO LOCAL (rápido y determinista)
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

        # 2. LIMPIEZA Y REDUCCIÓN (clave para el LLM)
        cleaned_input = [
            {
                "source": src,
                "errors": list(errors)[:15]  # límite duro
            }
            for src, errors in grouped_errors.items()
            if errors
        ]

        if not cleaned_input:
            return "No se encontraron errores ortográficos en los fragmentos analizados."

        # 3. LLAMADA AL LLM (solo para refinar)
        response = llm.invoke([
            SPELLING_PROMPT,
            SystemMessage(content=f"Fragmentos a revisar:<br>\n{cleaned_input}"),
            SystemMessage(content=f"Número total de errores detectados antes de limpieza: {sum(len(e['errors']) for e in cleaned_input)}")
        ])

        # 4. VALIDACIÓN DE SALIDA
        try:
            parsed = json.loads(response.content)

            final = []
            seen_sources = set()

            for item in parsed:
                source = item.get("source")
                errors = item.get("errors", [])

                if not source or source in seen_sources:
                    continue

                final.append({
                    "source": source,
                    "errors": list(set(errors))[:15]
                })

                seen_sources.add(source)

            # Convertir a string formateado
            if not final:
                return "No se encontraron errores ortográficos reales después de filtrado."
            
            resultado = "<br>\n".join([
                f"- fuente: {item['source']}<br>\n  errores: {', '.join(item['errors'])}"
                for item in final
            ])
            return resultado

        except Exception:
            # fallback seguro si el LLM falla - formatear cleaned_input como string
            if not cleaned_input:
                return "Error al procesar errores ortográficos."
            resultado = "<br>\n".join([
                f"- fuente: {item['source']}<br>\n  errores: {', '.join(item['errors'])}"
                for item in cleaned_input
            ])
            return resultado

    except Exception as e:
        return f"Error técnico en detector ortográfico: {str(e)}"

def detectar_fechas_invalidas(query: str):
    """Detecta fechas numéricas inválidas en fragmentos relevantes de la knowledge base."""
    message = query.split("..INTENTION :")[0].strip()

    try:
        docs = vector_store.similarity_search("texto general", k=50)

        if not docs:
            return "No encontré fragmentos relevantes para revisar fechas."

        # Formatos soportados:
        # dd/mm/yyyy, dd-mm-yyyy, yyyy/mm/dd, yyyy-mm-dd
        pattern_dmy = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
        pattern_ymd = re.compile(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b")

        hallazgos = []

        for d in docs:
            source = d.metadata.get("source", "desconocido")
            text = d.page_content

            for match in pattern_dmy.finditer(text):
                day, month, year = map(int, match.groups())
                fecha_str = match.group(0)

                try:
                    datetime(year, month, day)
                except ValueError:
                    hallazgos.append(
                        f"- fuente: {source}\n"
                        f"  fecha_detectada: {fecha_str}\n"
                        f"  problema: fecha calendario inválida"
                    )

            for match in pattern_ymd.finditer(text):
                year, month, day = map(int, match.groups())
                fecha_str = match.group(0)

                try:
                    datetime(year, month, day)
                except ValueError:
                    hallazgos.append(
                        f"- fuente: {source}\n"
                        f"  fecha_detectada: {fecha_str}\n"
                        f"  problema: fecha calendario inválida"
                    )

        if not hallazgos:
            return (
                "No encontré fechas numéricas inválidas en los fragmentos analizados. "
                "Esta versión detecta fechas imposibles, no contradicciones semánticas."
            )

        return "\n\n".join(hallazgos[:15])

    except Exception as e:
        return f"Error técnico en detector de fechas: {str(e)}"
    
def check_redundancy(query: str):
    """Detecta información redundante en la knowledge base."""
    try:
        docs = vector_store.similarity_search(query, k=50)

        if not docs:
            return "No encontré fragmentos relevantes para revisar."

        # Preparar contexto para el LLM
        contexto = "<br>\n<br>\n".join([
            f"[Fuente: {d.metadata.get('source', 'desconocido')}]<br>\n{d.page_content}"
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
            SystemMessage(content=f"Fragmentos a analizar:<br>\n{contexto}")
        ])

        return response.content

    except Exception as e:
        return f"Error técnico en detector de redundancias: {str(e)}"


#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#CHECK CONFLICTS
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================

# ====================================================================================
# CHECK CONFLICTS — HYBRID SEMANTIC CONFLICT DETECTOR
# ====================================================================================
#
# Arquitectura:
#
#   1. Recuperación de chunks desde ChromaDB
#   2. Agrupación por archivo
#   3. Extracción semántica de claims (LLM)
#   4. Normalización de conceptos
#   5. Agrupación semántica de claims
#   6. Comparación determinista en Python
#   7. Explicación y confidence score
#   8. Formateo final
#
# Filosofía:
#
#   - El LLM EXTRAE semántica
#   - Python DETECTA contradicciones
#   - El LLM NO decide toda la lógica
#
# ====================================================================================


# ====================================================================================
# PROMPT — EXTRACCIÓN DE CLAIMS
# ====================================================================================

CONFLICT_EXTRACTION_PROMPT = SystemMessage(content=(
    "Eres un extractor de HECHOS VERIFICABLES para detectar contradicciones entre documentos.<br>\n"
    "Tu objetivo es extraer SOLO afirmaciones concretas que PUEDAN CONTRADECIRSE.<br>\n<br>\n"

    "ESTRUCTURA DE CADA HECHO:<br>\n"
    "- concepto_clave: El HECHO ESPECÍFICO y VERIFICABLE. Ej: 'año de publicación de X', 'inventor de Y', 'descubridor de Z', 'versión de framework W'<br>\n"
    "- sujeto: QUIÉN, QUÉ o ENTIDAD ESPECÍFICA. Ej: 'Alan Turing', 'Python 3.8', 'red neuronal convolucional'<br>\n"
    "- complementos: detalles contextuales {tipo, valor}. Ej: {'tipo': 'año', 'valor': '1936'}<br>\n"
    "- fragmento_original: el texto exacto de donde se extrajo<br>\n<br>\n"

    "REGLAS DE EXTRACCIÓN:<br>\n"
    "1. Extrae SOLO hechos que PUEDEN CONTRADECIRSE entre documentos:<br>\n"
    "   ✅ BUENO: 'año de publicación del paper X', 'creador del algoritmo Y', 'versión lanzada de Z'<br>\n"
    "   ✅ BUENO: 'autor de la propuesta X', 'fecha de lanzamiento de framework Y'<br>\n"
    "   ❌ MALO: 'definición de machine learning' (es conceptual, raramente contradictoria)<br>\n"
    "   ❌ MALO: 'explicación de una fórmula' (típicamente igual en todas partes)<br>\n<br>\n"

    "2. Prioriza hechos HISTÓRICOS, TEMPORALES o DE AUTORÍA:<br>\n"
    "   ✅ BUENO: 'año de creación del lenguaje Python'<br>\n"
    "   ✅ BUENO: 'investigador que propuso la arquitectura X'<br>\n"
    "   ✅ BUENO: 'versión actual de TensorFlow'<br>\n"
    "   ❌ MALO: 'qué es programación orientada a objetos' (definición conceptual)<br>\n<br>\n"

    "3. Estructura para HECHOS ESPECÍFICOS:<br>\n"
    "   Concepto_clave DEBE incluir qué se está verificando:<br>\n"
    "   - '[año|fecha|versión] de [concepto específico]'<br>\n"
    "   - '[creador|inventor|autor|descubridor] de [concepto específico]'<br>\n"
    "   - '[característica] de [entidad específica]'<br>\n<br>\n"

    "4. NO extraigas:<br>\n"
    "   - Definiciones o explicaciones teóricas<br>\n"
    "   - Principios universales que no varían<br>\n"
    "   - Descripciones genéricas sin sujeto específico<br>\n"
    "   - Diálogos, narrativa o contenido sin valor factual<br>\n"
    "   - Hechos obvios o universalmente conocidos<br>\n<br>\n"

    "5. Mantén valores EXACTOS (sin normalizar):<br>\n"
    "   - Nombres de personas: exactos del texto<br>\n"
    "   - Años/fechas: exactos del texto<br>\n"
    "   - Nombres de tecnologías: exactos del texto<br>\n<br>\n"

    "6. Si NO HAY hechos verificables/contradictorios, devuelve []<br>\n<br>\n"

    "FORMATO DE SALIDA — JSON array estricto, sin texto adicional, sin bloques markdown:<br>\n"
    '[{"concepto_clave":"str","sujeto":"str|null","complementos":[{"tipo":"str","valor":"str"}],"fragmento_original":"str"}]'
))

# ====================================================================================
# NORMALIZACIÓN SEMÁNTICA
# ====================================================================================

def normalize_concept(text: str) -> str:
    """
    Normaliza conceptos para agrupar variantes semánticamente similares.

    Ejemplo:
        "Descubrimiento de América"
        "descubrimiento america"
        "QUIEN DESCUBRIO AMERICA"

    → todos convergen a una forma similar.
    """

    if not text:
        return ""

    text = normalize_text(text)

    # eliminar caracteres raros
    text = re.sub(r"[^\w\s]", " ", text)

    # eliminar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ====================================================================================
# SIMILITUD DE CONCEPTOS
# ====================================================================================

def concept_similarity(a: str, b: str) -> float:
    """
    Similaridad textual básica.

    FUTURO:
        reemplazar por embeddings similarity
        usando Chroma/HuggingFaceEmbeddings.
    """

    return SequenceMatcher(None, a, b).ratio()


# ====================================================================================
# EXTRACCIÓN DE CLAIMS
# ====================================================================================

def extract_claims_from_text(
    source: str,
    text: str
) -> list[dict]:
    """
    Extrae claims semánticos desde texto usando LLM.

    Retorna:
        [
            {
                "archivo": "...",
                "concepto_clave": "...",
                "sujeto": "...",
                "complementos": [...],
                "fragmento_original": "..."
            }
        ]
    """

    try:

        response = llm.invoke([
            CONFLICT_EXTRACTION_PROMPT,
            SystemMessage(content=(
                f"Archivo: {source}<br>\n"
                f"Texto:<br>\n{text}"
            ))
        ])

        content = response.content.strip()

        # limpieza defensiva
        content = re.sub(r"```json", "", content)
        content = re.sub(r"```", "", content)

        parsed = json.loads(content)

        if not isinstance(parsed, list):
            return []

        claims = []

        for claim in parsed:

            if not isinstance(claim, dict):
                continue

            claims.append({
                "archivo": source,
                "concepto_clave": claim.get("concepto_clave", ""),
                "concepto_normalizado": normalize_concept(
                    claim.get("concepto_clave", "")
                ),
                "sujeto": claim.get("sujeto"),
                "complementos": claim.get("complementos", []),
                "fragmento_original": claim.get("fragmento_original", "")
            })
        return claims

    except Exception as e:

        print(
            f"[check_conflicts] Error extrayendo claims "
            f"de '{source}': {e}"
        )

        return []


# ====================================================================================
# AGRUPACIÓN SEMÁNTICA DE CLAIMS
# ====================================================================================

def group_claims_semantically(
    claims: list[dict],
    threshold: float = 0.7
) -> dict[str, list[dict]]:
    """
    Agrupa claims por similitud semántica de concepto.

    IMPORTANTE:
        No usa igualdad exacta.
        Usa similitud textual aproximada.

    FUTURO:
        reemplazar por clustering con embeddings.
    """

    groups = {}

    for claim in claims:

        concept = claim["concepto_normalizado"]
        matched_group = None

        for existing_group in groups.keys():
            similarity = concept_similarity(
                concept,
                existing_group
            )

            if similarity >= threshold:
                matched_group = existing_group
                break

        if matched_group:
            groups[matched_group].append(claim)
        else:
            groups[concept] = [claim]

    return groups


# ====================================================================================
# CONVERSIÓN DE COMPLEMENTOS A DICT
# ====================================================================================

def complements_to_dict(complements: list[dict]) -> dict:
    """
    Convierte:
        [
            {"tipo":"año","valor":"1492"}
        ]
    →
        {
            "año":"1492"
        }
    """
    result = {}

    for c in complements:
        tipo = normalize_text(
            str(c.get("tipo", ""))
        )
        valor = str(c.get("valor", "")).strip()

        if tipo:
            result[tipo] = valor

    return result


# ====================================================================================
# COMPARADOR DETERMINISTA
# ====================================================================================

def compare_claims_within_group(
    claims: list[dict]
) -> list[dict]:
    """
    Detecta conflictos de forma determinista.
    NO usa LLM para decidir contradicciones.
    """

    conflicts = []

    for i in range(len(claims)):

        for j in range(i + 1, len(claims)):
            c1 = claims[i]
            c2 = claims[j]

            # ============================================================
            # NUNCA comparar dentro del mismo archivo
            # ============================================================

            if c1["archivo"] == c2["archivo"]:
                continue

            sujeto_1 = normalize_text(
                str(c1.get("sujeto", ""))
            )

            sujeto_2 = normalize_text(
                str(c2.get("sujeto", ""))
            )

            sujeto_difiere = (
                sujeto_1 != sujeto_2
                and sujeto_1 != ""
                and sujeto_2 != ""
            )

            comp_1 = complements_to_dict(
                c1.get("complementos", [])
            )

            comp_2 = complements_to_dict(
                c2.get("complementos", [])
            )

            shared_keys = (
                set(comp_1.keys())
                & set(comp_2.keys())
            )

            complementos_en_conflicto = []

            for key in shared_keys:

                v1 = normalize_text(
                    str(comp_1[key])
                )

                v2 = normalize_text(
                    str(comp_2[key])
                )

                # incompatibilidad real
                if v1 != v2:
                    complementos_en_conflicto.append(key)

            # ============================================================
            # NO hay conflicto
            # ============================================================

            if (
                not sujeto_difiere
                and not complementos_en_conflicto
            ):
                continue

            # ============================================================
            # CLASIFICACIÓN
            # ============================================================

            if sujeto_difiere and complementos_en_conflicto:
                tipo = "Tipo 1"
            else:
                tipo = "Tipo 2"

            # ============================================================
            # CONFIDENCE SCORE
            # ============================================================

            confidence = 0.75

            if sujeto_difiere:
                confidence += 0.10

            if complementos_en_conflicto:
                confidence += min(
                    0.15,
                    0.05 * len(complementos_en_conflicto)
                )

            confidence = round(
                min(confidence, 0.99),
                2
            )

            # ============================================================
            # DESCRIPCIÓN
            # ============================================================

            descripcion = []

            if sujeto_difiere:
                descripcion.append(
                    f"sujeto distinto "
                    f"('{c1.get('sujeto')}' vs '{c2.get('sujeto')}')"
                )

            for campo in complementos_en_conflicto:

                descripcion.append(
                    f"{campo} distinto "
                    f"('{comp_1.get(campo)}' vs '{comp_2.get(campo)}')"
                )

            conflicts.append({
                "concepto_clave":
                    c1.get("concepto_clave"),
                "tipo_conflicto":
                    tipo,
                "descripcion_conflicto":
                    "; ".join(descripcion),
                "confidence":
                    confidence,
                "archivo_a":
                    c1["archivo"],
                "fragmento_a":
                    c1["fragmento_original"],
                "archivo_b":
                    c2["archivo"],
                "fragmento_b":
                    c2["fragmento_original"],
                "elemento_en_conflicto": {
                    "sujeto_difiere":
                        sujeto_difiere,
                    "complementos_en_conflicto":
                        complementos_en_conflicto
                }
            })

    return conflicts

def _extract_mentioned_files(message: str) -> list[str]:
    """
    Extrae nombres de archivo mencionados explícitamente en el mensaje.

    Detecta patrones como: archivo_1.txt, documento.pdf, mi_cv.docx
    """
    pattern = re.compile(r"\b[\w\-]+\.[a-zA-Z]{2,5}\b")
    return pattern.findall(message)

# ====================================================================================
# FUNCIÓN PRINCIPAL
# ====================================================================================

def check_conflicts(query: str):
    """
    Detector híbrido de contradicciones semánticas.
    """

    message = query.split(
        "..INTENTION :"
    )[0].strip()

    try:

        # ================================================================
        # 1. RECUPERACIÓN GLOBAL
        # ================================================================

        data = vector_store.get()

        if not data or not data.get("documents"):
            return (
                "La base de datos está vacía."
            )

        # ================================================================
        # 2. AGRUPAR CHUNKS POR ARCHIVO
        # ================================================================

        archivos = defaultdict(list)

        for doc, metadata in zip(
            data["documents"],
            data["metadatas"]
        ):

            source = metadata.get(
                "source",
                "desconocido"
            )

            if doc and doc.strip():
                archivos[source].append(doc.strip())

# =========================================================
# DEBUG — VER ARCHIVOS DETECTADOS EN CHROMADB
# =========================================================

        # print("\n===== ARCHIVOS DETECTADOS EN CHROMADB =====")

        # for nombre in archivos.keys():
        #     print(nombre)

# =========================================================
# DEBUG — VER ARCHIVOS DETECTADOS EN CHROMADB
# =========================================================

        archivos_en_kb = set(archivos.keys())
        archivos_solicitados = _extract_mentioned_files(message)

        if archivos_solicitados:
            # El usuario mencionó archivos específicos
            archivos_encontrados = [
                f for f in archivos_solicitados
                if f in archivos_en_kb
            ]
            archivos_no_encontrados = [
                f for f in archivos_solicitados
                if f not in archivos_en_kb
            ]

            # Avisar si hay archivos faltantes
            if archivos_no_encontrados:
                print(
                    f"\n[WARNING] Archivos solicitados no encontrados: "
                    f"{', '.join(archivos_no_encontrados)}"
                )

            # Caso: NINGUNO de los archivos solicitados existe
            if len(archivos_encontrados) == 0:
                return {
                    "tool": "check_conflicts",
                    "status": "skipped",
                    "reason": "no_files_found",
                    "query": message,
                    "files_requested": archivos_solicitados,
                    "files_found": [],
                    "files_not_found": archivos_no_encontrados,
                    "files_available": list(archivos_en_kb),
                    "writer_hint": (
                        f"Ninguno de los archivos solicitados "
                        f"({', '.join(archivos_solicitados)}) "
                        f"se encuentra indexado en la base de datos vectorial. "
                        f"Archivos disponibles: {', '.join(archivos_en_kb) if archivos_en_kb else 'ninguno'}."
                    )
                }

            # Caso: Solo 1 archivo encontrado (insuficiente para detectar conflictos)
            if len(archivos_encontrados) == 1:
                return {
                    "tool": "check_conflicts",
                    "status": "skipped",
                    "reason": "insufficient_files",
                    "query": message,
                    "files_requested": archivos_solicitados,
                    "files_found": archivos_encontrados,
                    "files_not_found": archivos_no_encontrados,
                    "files_available": list(archivos_en_kb),
                    "writer_hint": (
                        f"De los archivos solicitados, solo se encontró 1 "
                        f"({archivos_encontrados[0]}) en la base de datos. "
                        f"Se requieren al menos 2 archivos para detectar conflictos. "
                        f"Archivos no encontrados: {', '.join(archivos_no_encontrados)}. "
                        f"Archivos disponibles: {', '.join(archivos_en_kb)}."
                    )
                }

            # Caso: 2 o más archivos encontrados → proceder
            # Filtrar para analizar SOLO los solicitados que existen
            archivos = {
                src: chunks
                for src, chunks in archivos.items()
                if src in archivos_encontrados
            }

        if len(archivos) < 2:

            return (
                "Se requieren al menos "
                "2 archivos distintos."
            )


        # ================================================================
        # 3. EXTRAER CLAIMS CHUNK-BY-CHUNK
        # ================================================================
        #
        # IMPORTANTE:
        #
        # NO truncamos archivos completos.
        # Procesamos chunk por chunk.
        #
        # Esto evita perder contexto importante.
        #
        # ================================================================

        all_claims = []

        MAX_CHUNKS_PER_FILE = 4

        for source, chunks in archivos.items():
            selected_chunks = chunks[:MAX_CHUNKS_PER_FILE]
            for chunk in selected_chunks:

                if len(chunk.strip()) < 20:
                    continue

                claims = extract_claims_from_text(
                    source=source,
                    text=chunk
                )
# ============================TEST====================================
#                print(f"[DEBUG] Claims de {source}: {claims}")
# ============================TEST====================================
                
                all_claims.extend(claims)

        if len(all_claims) < 2:
            return (
                "No se pudieron extraer "
                "claims suficientes."
            )

        # ================================================================
        # 4. AGRUPACIÓN SEMÁNTICA
        # ================================================================

        grouped_claims = group_claims_semantically(
            all_claims
        )

        # ================================================================
        # 5. DETECCIÓN DETERMINISTA
        # ================================================================

        all_conflicts = []

        for concept, claims in grouped_claims.items():
            if len(claims) < 2:
                continue
            conflicts = compare_claims_within_group(
                claims
            )
            all_conflicts.extend(conflicts)

        # ================================================================
        # 6. RESULTADO VACÍO
        # ================================================================

        if not all_conflicts:
            return {
                "tool": "check_conflicts",
                "status": "success",
                "query": message,
                "files_analyzed": list(archivos.keys()),
                "claims_extracted": len(all_claims),
                "conflicts_found": 0,
                "conflicts_by_type": {
                    "Tipo 1": 0,
                    "Tipo 2": 0
                },
                "conflicts": []
            }

        # ================================================================
        # 7. ESTADÍSTICAS DE CONFLICTOS
        # ================================================================

        tipo1 = [
            c for c in all_conflicts
            if c["tipo_conflicto"] == "Tipo 1"
        ]
        tipo2 = [
            c for c in all_conflicts
            if c["tipo_conflicto"] == "Tipo 2"
        ]

        # ================================================================
        # 8. RESPUESTA ESTRUCTURADA PARA EL GRAFO
        # ================================================================
        #
        # IMPORTANTE:
        #
        # Esta tool NO genera lenguaje natural final.
        #
        # El writer_node será el encargado de:
        #   - resumir
        #   - explicar
        #   - priorizar
        #   - redactar
        #
        # Aquí solo devolvemos datos estructurados.
        #
        # ================================================================

        return {
            "tool": "check_conflicts",
            "status": "success",
            "query": message,
            "files_analyzed": list(archivos.keys()),
            "claims_extracted": len(all_claims),
            "conflicts_found": len(all_conflicts),
            "conflicts_by_type": {
                "Tipo 1": len(tipo1),
                "Tipo 2": len(tipo2)
            },
            "conflicts": all_conflicts
        }

    except Exception as e:
        # ================================================================
        # ERROR ESTRUCTURADO
        # ================================================================
        return {
            "tool": "check_conflicts",
            "status": "error",
            "query": message,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#CHECK CONFLICTS
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================

def check_obsolescence(query: str):
    """Detecta información obsoleta o que ya no es relevante en la knowledge base."""
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
                        hallazgos.append(
                            f"- fuente: {source}<br>\n"
                            f"  fecha_detectada: {match.group(0)}<br>\n"
                            f"  problema: fecha desactualizada (más de 1 año)"
                        )
                except ValueError:
                    pass

            for match in pattern_ymd.finditer(text):
                year, month, day = map(int, match.groups())
                try:
                    fecha = datetime(year, month, day)
                    if fecha < umbral:
                        hallazgos.append(
                            f"- fuente: {source}<br>\n"
                            f"  fecha_detectada: {match.group(0)}<br>\n"
                            f"  problema: fecha desactualizada (más de 1 año)"
                        )
                except ValueError:
                    pass

            for match in pattern_year.finditer(text):
                year = int(match.group(1))
                try:
                    fecha = datetime(year, 1, 1)
                    if fecha < umbral:
                        hallazgos.append(
                            f"- fuente: {source}<br>\n"
                            f"  año_detectado: {match.group(0)}<br>\n"
                            f"  problema: año desactualizado (más de 1 año)"
                        )
                except ValueError:
                    pass

        if not hallazgos:
            return "No se encontró contenido desactualizado en los fragmentos analizados."

        return "\n\n".join(hallazgos[:15])

    except Exception as e:
        return f"Error técnico en detector de contenido desactualizado: {str(e)}"

def check_general_analysis(query: str):
    """Realiza un análisis general del estado de la knowledge base respecto a una query."""
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

# Actualiza tu lista de herramientas
tools = [get_weather, consultar_knowledge_base, buscar_en_web]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools) # El ToolNode manejará automáticamente la ejecución

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_BASE_URL,
    should_export_span=lambda span: (
        span.instrumentation_scope is not None
        and span.instrumentation_scope.name == "langfuse-sdk"
    )
)

langfuse_handler = CallbackHandler()


CONFIG = {
    "callbacks": [langfuse_handler],
    "run_name": "knowledge_graph",
    "run_id": uuid4(),
    "recursion_limit": 50,
    "metadata": {
        "langfuse_session_id": "session-123",
        "langfuse_user_id": "juan",
        "agent": "knowledge_analyzer",
        "version": "v1"
    }
}












# ====================================================================================
# DEFINICION DE NODOS
# ====================================================================================
@observe(name="prompt_node")
def prompt_node(state: State) -> State:
    """Decide si el mensaje necesita tools o puede responderse con memoria."""
    memory_data = get_memory_data()
    response = llm.invoke([PROMPT_NODE_PROMPT] + memory_data + [TOOL_SET] + state["messages"])
    del_memory_data()

    try:
        parsed = json.loads(response.content)
        need_tools  = "yes" if parsed.get("needs_tools") else "no"
    except Exception:
        need_tools = "no"  # si falla el parse, responde directo

    state["actual_node"] = "prompt_node"
    save_memory(state)
    return {
        "messages":            [response],
        "conditional_message": need_tools,
        "iterations":          state.get("iterations", 0) + 1,
    }



@observe(name="planner_node")
def planner_node(state: State) -> State:
    """Descompone el mensaje en tasks y las enlista en el estado."""
    response = llm.invoke([PLANNER_PROMPT] + [TOOL_SET] + state["messages"])

    try:
        raw_tasks = json.loads(response.content)
        tasks: list[Task] = [
            {
                "task_name":    t["task_name"],
                "task_message": t["task_message"],
                "intention":    t["intention"],
                "status":       "pending",
                "message":      "",
                "used_tool":    t["used_tool"],
            }
            for t in raw_tasks
        ]
    except Exception as e:
        # Si el LLM no devuelve JSON válido, creamos una task genérica
        tasks = [{
            "task_name":    "tarea_general",
            "task_message": "No se pudo parsear el plan",
            "intention":    "sin intencion",
            "status":       "failed",
            "message":      f"Error al planificar: {e}",
            "used_tool":    "none",
        }]

    state["actual_node"] = "planner_node"
    save_memory(state)
    return {
        "messages": [response],
        "tasks":    tasks,
    }



@observe(name="tool_executor_node")
def tool_executor_node(state: State) -> State:
    """Toma la primera task pendiente, ejecuta su tool y actualiza su estado."""
    tasks = list(state["tasks"])  # copia para no mutar el estado

    # Encuentra la primera task pendiente
    pending_index = next(
        (i for i, t in enumerate(tasks) if t["status"] == "pending"),
        None
    )

    if pending_index is None:
        # No debería llegar aquí, pero por seguridad
        return {"tasks": tasks, "conditional_message": "done"}

    task = tasks[pending_index]

    # Si la task no necesita tool, la marcamos completed directamente
    if task["used_tool"] == "none":
        tasks[pending_index] = {**task, "status": "completed", "message": "Resuelta por conocimiento propio"}
    else:
        try:
            tool_fn   = TOOLS_MAP[task["used_tool"]]
            resultado = tool_fn.invoke(f"{task['task_message']} ..INTENTION :{task['intention']}")
            tasks[pending_index] = {**task, "status": "completed", "message": resultado}
        except Exception as e:
            tasks[pending_index] = {**task, "status": "failed", "message": f"Error: {e}"}

    # Chequea si quedan pendientes
    hay_pending = any(t["status"] == "pending" for t in tasks)

    return {
        "tasks":               tasks,
        "conditional_message": "pending" if hay_pending else "done",
        "iterations":          state.get("iterations", 0) + 1,
    }

def suggest_executor_node(state: State) -> State:
    """Toma la primera task done, ejecuta su suggest y actualiza su estado."""
    pass

@observe(name="writer_node")
def writer_node(state: State) -> State:
    """Genera la respuesta final iterando todas las tasks y sus sugerencias."""

    tasks = state.get("tasks", [])

    user_message = state["messages"][0] if state["messages"] else "Sin mensaje"

    user_msg_content = (
        user_message.content
        if hasattr(user_message, "content")
        else str(user_message)
    )

    if not tasks:
        response = llm.invoke([
            WRITER_PROMPT,
            SystemMessage(content=f"Mensaje del usuario: {user_msg_content}")
        ])

        print(f"Uso de tokens en writer: {response.response_metadata.get('token_usage', {})}")

        return {
            "messages": [
                AIMessage(content=response.content)
            ]
        }

    # =========================
    # RESUMEN PARA EL LLM
    # =========================

    resumen = "<br>\n".join([
        f"- [{t['status'].upper()}] TASK NAME: {t['task_name']}: MESSAGE: {t['message']}  TASK MESSAGE: {t['task_message']}   USED TOOL: {t['used_tool']}"
        for t in tasks
    ])

    writer_context = SystemMessage(content=(
        f"{WRITER_PROMPT.content}<br>\n<br>\n"
        f"Mensaje original del usuario:<br>\n{user_msg_content}<br>\n<br>\n"
        f"Lista de tareas ejecutadas:<br>\n{resumen}"
    ))

    response = llm.invoke([writer_context])

    if not response.content or response.content.strip() == "":
        print("[WARNING] Writer retornó respuesta vacía")
        response_content = "No se pudo generar un resumen automático."
    else:
        response_content = response.content

    # =========================
    # ANALISIS COMPLETO
    # =========================
    detailed_analysis = "<br>\n<br>\n=== ANALISIS COMPLETO ===<br>\n"

    for idx, task in enumerate(tasks, start=1):

        task_name = task.get("task_name", "Sin nombre")
        task_status = task.get("status", "unknown")
        task_tool = task.get("used_tool", "none")
        task_message = task.get("message", "Sin resultado")

        detailed_analysis += (
            f"<br>\n[{idx}] {task_name} <br>\n"
            f"Status: {task_status} <br>\n"
            f"Tool: {task_tool} <br>\n"
            f"Resultado: <br>\n{task_message} <br>\n"
        )

    # =========================
    # MENSAJE FINAL
    # =========================
    final_message = (
        f"{response_content}<br><br><br>\n\n"
        f"{detailed_analysis}"
    )

    print(f"Uso de tokens en writer: {response.response_metadata.get('token_usage', {})}")

    state["actual_node"] = "writer_node"

    save_memory(state)

    return {
        "messages": [
            AIMessage(content=final_message)
        ]
    }




@observe(name="memory_node")
def memory_node(state: State) -> State:
    """Decide si puede resolver con memoria o redirige al planner."""

    memory_data = get_memory_data()
    response = llm.invoke([MEMORY_PROMPT] + memory_data + state["messages"])

    try:
        raw_tasks = json.loads(response.content.split(";")[0])  # Solo la parte del JSON, ignorando el mensaje para planner

        tasks: list[Task] = [
            {
                "task_name":    t["task_name"],
                "task_message": t["task_message"],
                "intention":    t["intention"],
                "status":       "pending",
                "message":      "",
                "used_tool":    t.get("used_tool", "none"),
            }
            for t in raw_tasks if "status" == "completed"
        ]

    except Exception as e:
        print(f"[WARNING] Error al parsear memoria: {e}")
        tasks = []
    
    state["actual_node"] = "memory_node"
    save_memory(state)
    
    #Ahora quitamos el "mensaje_planner: " para quedarnos solo con la lista de tareas pendientes
    mensaje = response.content.split(";")[-1].strip()[16:]  # Quitar "mensaje_planner: "
    return {
        "messages":            [mensaje],  # Solo el mensaje para planner
        "memory_tasks":               tasks,
        "conditional_message": "yes"
    }


@observe(name="query_node")
def query_node(state: State) -> State:
    tasks = state.get("tasks", [])

    # Construimos contexto claro para el LLM
    TASKS_CONTEXT = "TAREAS:<br>\n" + "<br>\n".join([
        f"- task_name: {t['task_name']}<br>\n"
        f"  status: {t['status']}<br>\n"
        f"  resultado: {t['message']}<br>\n"
        for t in tasks
    ])

    response = llm.invoke([QUERY_PROMPT] + [TASKS_CONTEXT])

    try:
        suggestions = json.loads(response.content)

        # Mapeo por nombre de task
        suggestion_map = {
            s["task_name"]: s for s in suggestions
        }

        new_tasks = []
        for t in tasks:
            s = suggestion_map.get(t["task_name"], {})

            new_tasks.append({
                **t,
                "need_suggestion": s.get("needs_suggestion", False),
                "suggestion_message": s.get("suggestion_message", "")
            })

    except Exception as e:
        # fallback seguro
        new_tasks = [
            {**t, "need_suggestion": False, "suggestion_message": ""}
            for t in tasks
        ]

    return {
        "tasks": new_tasks,
        "messages": [response]
    }









# Este es un nodo condicional, según el mensaje generado por prompt node, elije lo que nosotros le digamos.
def after_prompt(state: State) -> Literal["planner_node", "memory_node"]:
    return "planner_node" if state["conditional_message"] == "yes" else "memory_node"

def after_memory(state: State) -> Literal["planner_node", "writer_node"]:
    # Si necesita tools, va al planner
    return "planner_node" if state["conditional_message"] == "yes" else "writer_node"

def after_executor(state: State) -> Literal["tool_executor_node", "query_node"]:
    return "tool_executor_node" if state["conditional_message"] == "pending" else "query_node"

# Agregamos nodos y edges al grafo
graph = StateGraph(State)

graph.add_node("prompt_node",       prompt_node)
graph.add_node("planner_node",      planner_node)
graph.add_node("tool_executor_node",tool_executor_node)
graph.add_node("writer_node",       writer_node)
graph.add_node("memory_node",       memory_node)
graph.add_node("query_node",        query_node)
graph.set_entry_point("prompt_node")

graph.add_conditional_edges("prompt_node",        after_prompt)
graph.add_edge(              "planner_node",       "tool_executor_node")
graph.add_conditional_edges("memory_node",        after_memory)
graph.add_conditional_edges("tool_executor_node",  after_executor)
graph.add_edge(              "query_node",         "writer_node")
graph.add_edge(              "writer_node",        "__end__")

APP = graph.compile().with_config({
    "callbacks": [langfuse_handler]
})

# if __name__ == "__main__":

#     import json

#     print(
#         "\n================================================="
#         "\nTEST AISLADO — CHECK_CONFLICTS"
#         "\n=================================================\n"
#     )

#     resultado = check_conflicts(
#         "Analiza conflictos entre fuentes_energia_A.pdf y fuentes_energia_C.txt"
#     )

#     print(
#         json.dumps(
#             resultado,
#             indent=2,
#             ensure_ascii=False
#         )
#     )

#     print(
#         "\n================================================="
#         "\nFIN TEST"
#         "\n=================================================\n"
#     )




# if __name__ == "__main__":
#     print("=================================================\nINICIO DE LA COMPILACION\n=================================================")

#     new_state = APP.invoke(
#     {
#         "messages": [
#             "dame la suma del 1 al 10, el clima en yorkshire, y el clima en bogota, y los nombres de los archivos de mi knowledge base, revisa que errores de fecha y ortografia tienen, y dime porque los zorros articos comen zapatos"
#         ]
#     },
#     config=CONFIG
#     )
    
    
    
    
    
    
    
    
    
#     print(new_state["messages"][-1].content)
#     print("=================================================\nFIN DE LA COMPILACION\n=================================================")

#     print("=================================================\nINICIO DEL GRAFO\n=================================================")
#     from langchain_core.runnables.graph import MermaidDrawMethod
#     print(APP.get_graph().draw_mermaid())
#     # Pegan el resultado en https://mermaid.live/ para visualizar el grafo.
#     print("=================================================\nFIN DEL GRAFO\n=================================================")
