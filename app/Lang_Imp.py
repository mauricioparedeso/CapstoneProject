API_KEY= ""

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

import json

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
    "Responde ÚNICAMENTE con un JSON con este formato, sin texto adicional:\n"
    '{"needs_tools": true} o {"needs_tools": false}\n'
))

PLANNER_PROMPT = SystemMessage(content=(
    "Eres un planificador. Analiza el mensaje del usuario y descomponlo en tareas.\n"
    "Cada tarea es una petición atómica del usuario.\n"
    "Para cada tarea, decide qué herramienta usar, y que mensaje enviar como prompt. Si una tarea no necesita herramienta, márcala con used_tool: 'none'.\n\n"
    "Responde ÚNICAMENTE con un JSON array, sin texto adicional, con esta estructura:\n"
    ' {"task_name": "str", "status": "pending", "task_message": "str", "used_tool": "str"}\n'
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

TOOL_SET = SystemMessage(
    content=(
        "Tienes acceso a las siguientes herramientas:\n"
        "1. get_weather(location): Devuelve el clima actual para una ubicación dada.\n"
        "2. consultar_knowledge_base(query): Consulta la base de datos de documentos. Si la query pide nombres, lista todos."
    )
)

#====================================================================================
# Definimos el estado de nuestra aplicación como un diccionario de variables
# El estado se define como en que nodo está el grafo y que mensajes se han generado hasta el momento.
# Tambien iterations, que es el número de veces que se ha ejecutado el grafo, para evitar loops infinitos.

class Task(TypedDict):
    task_name: str
    task_message: str
    status: Literal["pending", "completed", "failed"]
    message: str
    used_tool: str

class State(TypedDict):
    messages: Annotated[list, add_messages]
    conditional_message: str  # Para trackear el mensaje que decide a qué nodo ir
    iterations: int
    tasks: list[Task]  # Para trackear tareas asincronas

graph = StateGraph(State)
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
    try:
        # Si la pregunta es sobre 'nombres' o 'lista', traemos todo lo que haya
        if "nombre" in query.lower() or "archivo" in query.lower() or "lista" in query.lower():
            # .get() trae los registros sin filtrar por similitud de texto
            data = vector_store.get()
            if not data or not data["metadatas"]:
                return "La base de datos está vacía."
            nombres = list(set([m.get("source") for m in data["metadatas"]]))
            return f"Archivos indexados en la base de datos: {nombres}"

        # Búsqueda normal de contenido
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No encontré información específica sobre eso en los documentos."
        
        return "\n\n".join([f"Del archivo {d.metadata.get('source')}: {d.page_content}" for d in docs])
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


# Primer nodo del grafo, que se encarga de generar un mensaje a partir del estado actual
# Además de generar un mensaje, este nodo también puede llamar a toolnodes, dependiendo del mensaje generado.4
def prompt_node(state: State) -> State:
    """Decide si el mensaje necesita tools o puede responderse directamente."""
    response = llm.invoke([PROMPT_NODE_PROMPT] + [TOOL_SET] + state["messages"])

    try:
        parsed = json.loads(response.content)
        need_tools  = "yes" if parsed.get("needs_tools") else "no"
    except Exception:
        need_tools = "no"  # si falla el parse, responde directo

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
            "status":       "failed",
            "message":      f"Error al planificar: {e}",
            "used_tool":    "none",
        }]

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
            resultado = tool_fn.invoke(task["task_message"])
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
    resumen = "\n".join([
        f"- [{t['status'].upper()}] {t['task_name']}: {t['message'] or t['task_message']}"
        for t in state["tasks"]
    ])

    writer_context = SystemMessage(content=(
        f"{WRITER_PROMPT.content}\n\n"
        f"Lista de tareas ejecutadas:\n{resumen}"
    ))

    response = llm.invoke([writer_context] + state["messages"])
    return {"messages": [response]}

graph.add_node("prompt_node", prompt_node)

# Este es un nodo condicional, según el mensaje generado por prompt node, elije lo que nosotros le digamos.
def after_prompt(state: State) -> Literal["planner_node", "writer_node"]:
    return "planner_node" if state["conditional_message"] == "yes" else "writer_node"

def after_executor(state: State) -> Literal["tool_executor_node", "writer_node"]:
    return "tool_executor_node" if state["conditional_message"] == "pending" else "writer_node"

# Agregamos nodos y edges al grafo
graph = StateGraph(State)

graph.add_node("prompt_node",       prompt_node)
graph.add_node("planner_node",      planner_node)
graph.add_node("tool_executor_node",tool_executor_node)
graph.add_node("writer_node",       writer_node)

graph.set_entry_point("prompt_node")

graph.add_conditional_edges("prompt_node",        after_prompt)
graph.add_edge(              "planner_node",       "tool_executor_node")
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
