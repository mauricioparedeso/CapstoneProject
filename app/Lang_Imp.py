with open(".gitignore/API_KEY.txt", "r") as f:
    API_KEY = f.read().strip()

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
#from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from transformers import AutoTokenizer

import json, os, unicodedata
from datetime import datetime

from app.Chroma_Imp import vector_store 

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
    "Para cada tarea, decide qué herramienta usar, y que mensaje enviar como prompt. Si una tarea no necesita herramienta, márcala con used_tool: 'none'.\n\n"
    "La intention de cada tarea es entender qué información específica el usuario quiere obtener. Las opciones posibles son: ['weather', 'file listing', 'general analysis', 'detect redundancy', 'detect incorrect info', 'detect conflicts', 'detect obsolescence', 'detect outdated content'].\n"
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
    "Sé conciso y directo."
))

MEMORY_PROMPT = SystemMessage(content=(
    "Tu tarea es tener en cuenta el archivo memory_log.json, que contiene un historial de interacciones pasadas. Úsalo para evitar repetir información o cometer los mismos errores. No es necesario que cites el memory_log, pero úsalo como referencia para mejorar tus respuestas."
    "Si ves que el usuario hace una pregunta similar a una interacción pasada, intenta dar una respuesta diferente o más completa, aprendiendo de lo que se hizo antes."
    "En caso de que sea una pregunta nueva, envía como need_tools: true para que el planner genere tareas normalmente. Si es una pregunta repetida, responde con need_tools: false y resuelve con tu conocimiento, pero teniendo en cuenta lo que se hizo antes."
    "Si consideras que es necesaria una tool, envía como message el motivo por el cual crees que la tool es necesaria, para que el planner pueda tomarlo en cuenta al generar las tareas."
    "Responde ÚNICAMENTE con un JSON con este formato, sin texto adicional:\n"
    '{"needs_tools": [true,false], "message": "str"}\n'
))

TOOL_SET = SystemMessage(
    content=(
        "Tienes acceso a las siguientes herramientas:\n"
        "1. get_weather(location): Devuelve el clima actual para una ubicación dada.\n"
        "2. consultar_knowledge_base(query): Consulta la base de datos de documentos. Si la query pide nombres, lista todos."
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

class State(TypedDict):
    actual_node: str
    messages: Annotated[list, add_messages]
    conditional_message: str  # Para trackear el mensaje que decide a qué nodo ir
    iterations: int
    tasks: list[Task]  # Para trackear tareas asincronas

graph = StateGraph(State)
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
    }

    data.append(normalize_safe(entry))

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_memory_data() -> list[SystemMessage]:
    if not os.path.exists(MEMORY_FILE):
        return []
    
    try:
        with open(MEMORY_FILE, "r") as f:
            content = f.read().strip()
            if content == "":
                return []
            data = json.loads(content)
    except json.JSONDecodeError:
        print("[WARNING] JSON corrupto, no se cargará memoria")
        return []

    # Convertimos cada entrada del log en un SystemMessage para el LLM
    messages = []
    for entry in data[-10:]:  # limitamos a las últimas 10 entradas para no saturar
        msg_content = (
            f"En el pasado, en el nodo '{entry['node']}' con iteración {entry['iterations']}, se generó el mensaje: '{entry['message']}'. "
            f"Las tareas asociadas fueron: {entry.get('tasks', [])}. "
            f"La decisión condicional fue: '{entry.get('conditional_message')}'."
        )
        messages.append(SystemMessage(content=msg_content))

    return messages






#====================================================================================
# Definimos herramientas, osease, funciones que pueden ser llamadas por un toolnode
#====================================================================================
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
    #['weather', 'file listing', 'general analysis', 'detect redundancy', 'detect incorrect info', 'detect conflicts', 'detect obsolescence', 'detect outdated content']
    message = query.split("..INTENTION :")[0].strip()
    intention = query.split("..INTENTION :")[1].strip() if "..INTENTION :" in query else ""
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
            pass
        if intention == "detect incorrect info":
            pass
        if intention == "detect conflicts":
            pass
        if intention == "detect obsolescence":
            pass
        if intention == "detect outdated content":
            pass

        # Búsqueda normal de contenido
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No encontré información específica sobre eso en los documentos."
        
        return "\n <br>".join([f"Del archivo {d.metadata.get('source')}: {d.page_content}" for d in docs])
    except Exception as e:
        return f"Error técnico: {str(e)}"

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


graph.add_node("tool_node", tool_node)















# ====================================================================================
# DEFINICION DE NODOS
# ====================================================================================
def prompt_node(state: State) -> State:
    """Decide si el mensaje necesita tools o puede responderse con memoria."""
    memory_data = get_memory_data()
    response = llm.invoke([PROMPT_NODE_PROMPT] + memory_data + [TOOL_SET] + state["messages"])

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
                "used_tool":    t.get("used_tool", "none"),
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




def writer_node(state: State) -> State:
    """Genera la respuesta final iterando todas las tasks."""

    tasks = state.get("tasks", [])
    
    if not tasks:
        # fallback cuando no hay planner
        response = llm.invoke([WRITER_PROMPT] + state["messages"])
        return {"messages": [response]}

    resumen = "\n".join([
        f"- [{t['status'].upper()}] {t['task_name']}: {t['message'] or t['task_message']}"
        for t in state["tasks"]
    ])

    writer_context = SystemMessage(content=(
        f"{WRITER_PROMPT.content}\n\n"
        f"Lista de tareas ejecutadas:\n{resumen}"
    ))

    response = llm.invoke([writer_context] + state["messages"])

    state["actual_node"] = "writer_node"
    save_memory(state)

    return {"messages": [response]}

def memory_node(state: State) -> State:
    """Usa el registro de memoria para responder. Si el mensaje es nuevo, redirige al planner."""

    memory_data = get_memory_data()  # función para cargar y formatear el memory_log.json
    response = llm.invoke([MEMORY_PROMPT] + memory_data + state["messages"])

    try:
        parsed = json.loads(response.content)
        need_tools  = "yes" if parsed.get("needs_tools") else "no"
    except Exception:
        need_tools = "no"  # si falla el parse, responde directo

    state["actual_node"] = "memory_node"
    save_memory(state)
    return {
        "messages":            [response],
        "conditional_message": need_tools,
        "iterations":          state.get("iterations", 0) + 1,
    }

graph.add_node("prompt_node", prompt_node)




# Este es un nodo condicional, según el mensaje generado por prompt node, elije lo que nosotros le digamos.
def after_prompt(state: State) -> Literal["planner_node", "memory_node"]:
    return "planner_node" if state["conditional_message"] == "yes" else "memory_node"

def after_memory(state: State) -> Literal["planner_node", "writer_node"]:
    return "planner_node" if state["conditional_message"] == "yes" else "writer_node"

def after_executor(state: State) -> Literal["tool_executor_node", "writer_node"]:
    return "tool_executor_node" if state["conditional_message"] == "pending" else "writer_node"

# Agregamos nodos y edges al grafo
graph = StateGraph(State)

graph.add_node("prompt_node",       prompt_node)
graph.add_node("planner_node",      planner_node)
graph.add_node("tool_executor_node",tool_executor_node)
graph.add_node("writer_node",       writer_node)
graph.add_node("memory_node",       memory_node)

graph.set_entry_point("prompt_node")

graph.add_conditional_edges("prompt_node",        after_prompt)
graph.add_edge(              "planner_node",       "tool_executor_node")
graph.add_conditional_edges("memory_node",        after_memory)
graph.add_conditional_edges("tool_executor_node",  after_executor)
graph.add_edge(              "writer_node",        "__end__")

APP = graph.compile()

if __name__ == "__main__":
    print("=================================================\nINICIO DE LA COMPILACION\n=================================================")
    new_state = APP.invoke({"messages": ["dame la suma del 1 al 10, el clima en yorkshire, y el clima en bogotá, y los nombres de los archivos de mi knowledge base, y porque los zorros articos comen zapatos"]})
    print(new_state["messages"][-1].content)
    print("=================================================\nFIN DE LA COMPILACION\n=================================================")

    print("=================================================\nINICIO DEL GRAFO\n=================================================")
    from langchain_core.runnables.graph import MermaidDrawMethod
    print(APP.get_graph().draw_mermaid())
    # Pegan el resultado en https://mermaid.live/ para visualizar el grafo.
    print("=================================================\nFIN DEL GRAFO\n=================================================")
