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
    "La intention de cada tarea es entender qué información específica el usuario quiere obtener. Las opciones posibles son: ['weather', 'file listing', 'general analysis', 'detect redundancy', 'detect incorrect info', 'detect conflicts', 'detect obsolescence', 'detect outdated content'].<br>\n"
   "Usa 'detect outdated content' ÚNICAMENTE para detectar fechas numéricas pasadas en documentos (por ejemplo, fechas de 2024 estando en 2026). Usa 'detect obsolescence' ÚNICAMENTE para detectar frameworks o librerías tecnológicamente desactualizadas. Usa 'detect incorrect info' para detectar datos erróneos como fechas imposibles o errores ortográficos evidentes. Usa 'general analysis' para tareas de análisis que no encajan en las otras categorías.<br>\n"
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

    "Debes:<br>\n"
    "- Resumir brevemente qué se analizó.<br>\n"
    "- Indicar qué herramientas se utilizaron.<br>\n"
    "- Contar cuántos errores, inconsistencias o problemas fueron detectados por cada herramienta o tarea.<br>\n"
    "- Mencionar si alguna tarea falló, explicando de forma breve qué ocurrió.<br>\n"
    "- Para tareas con used_tool='none', resolverlas usando tu propio conocimiento.<br>\n"
    "- Destacar hallazgos importantes o repetitivos.<br>\n"
    "- Dar recomendaciones simples y prácticas basadas en los resultados.<br>\n<br>\n"

    "Las recomendaciones deben ser cortas, accionables y fáciles de entender.<br>\n"
    "Ejemplos:<br>\n"
    "- corregir formatos de fecha inconsistentes<br>\n"
    "- revisar ortografía antes de subir documentos<br>\n"
    "- renombrar archivos ambiguos<br>\n"
    "- validar datos faltantes<br>\n<br>\n"

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
    '{"task_name": "str", "status": "pending", "message": "", "intention": "str", "used_tool": "none | get_weather | consultar_knowledge_base"}<br>\n<br>\n' 
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
            pass

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


def check_conflicts(query: str):
    """Detecta información conflictiva o contradictoria en la knowledge base."""
    return "Función de detección de conflictos aún no implementada."

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
}





llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=API_KEY
)

# Actualiza tu lista de herramientas
tools = [get_weather, consultar_knowledge_base]
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



if __name__ == "__main__":
    print("=================================================\nINICIO DE LA COMPILACION\n=================================================")

    new_state = APP.invoke(
    {
        "messages": [
            "dame la suma del 1 al 10, el clima en yorkshire, y el clima en bogota, y los nombres de los archivos de mi knowledge base, revisa que errores de fecha y ortografia tienen, y dime porque los zorros articos comen zapatos"
        ]
    },
    config=CONFIG
    )
    
    
    
    
    
    
    
    
    
    print(new_state["messages"][-1].content)
    print("=================================================\nFIN DE LA COMPILACION\n=================================================")

    print("=================================================\nINICIO DEL GRAFO\n=================================================")
    from langchain_core.runnables.graph import MermaidDrawMethod
    print(APP.get_graph().draw_mermaid())
    # Pegan el resultado en https://mermaid.live/ para visualizar el grafo.
    print("=================================================\nFIN DEL GRAFO\n=================================================")
