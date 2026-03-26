API_KEY= ""
# Para visualizar, Ctrl+Shift+P -> LangGraph: Visualize Graph

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

from app.Chroma_Imp import vector_store 


#====================================================================================
# Definimos el estado de nuestra aplicación como un diccionario de variables
# El estado se define como en que nodo está el grafo y que mensajes se han generado hasta el momento.
class State(TypedDict):
    messages: Annotated[list, add_messages]

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
# Además de generar un mensaje, este nodo también puede llamar a toolnodes, dependiendo del mensaje generado.
def prompt_node(state: State) -> State:
    instrucciones = SystemMessage(
    content=(
        "Eres un asistente de respuesta consolidada. Sigue estas reglas estrictas:\n"
        "1. EJECUCIÓN PARALELA: Llama a todas las herramientas necesarias (clima, archivos) en un solo turno. No vayas una por una.\n"
        "2. CONOCIMIENTO PROPIO: Resuelve matemáticas (sumas) y curiosidades (zorros) usando tu propio conocimiento inmediatamente.\n"
        "3. REGLA DE NO REINTENTO: Si una herramienta devuelve 'No se encontró información', acéptalo y no vuelvas a llamarla.\n"
        "4. RESPUESTA ÚNICA: Espera a tener todos los resultados de las herramientas para dar una única respuesta final que incluya: suma, climas, archivos y zorros.\n"
        "5. BREVEDAD: Responde con precisión quirúrgica, sin introducciones ni texto innecesario."
    )
)

    new_message = llm_with_tools.invoke([instrucciones] + state["messages"])
    return {"messages": [new_message]}

graph.add_node("prompt_node", prompt_node)

# Este es un nodo condicional, según el mensaje generado por prompt node, elije lo que nosotros le digamos.
def conditional_edge(state: State) -> Literal['tool_node', '__end__']:
    last_message = state["messages"][-1]
    if last_message.tool_calls:    # Si el último mensaje tiene llamadas a herramientas, vamos al nodo de herramientas
        return "tool_node"
    else:
        return "__end__"

# Agregamos nodos y edges al grafo
graph.add_conditional_edges(
    'prompt_node',
    conditional_edge
)
graph.add_edge("tool_node", "prompt_node")
graph.set_entry_point("prompt_node")

APP = graph.compile() # Esto es lo que importa la API

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
