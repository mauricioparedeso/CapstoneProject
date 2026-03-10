# API KEY:
# Para visualizar, Ctrl+Shift+P -> LangGraph: Visualize Graph

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
#from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

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

    
    
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=""
)

tools = [get_weather]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

graph.add_node("tool_node", tool_node)


# Primer nodo del grafo, que se encarga de generar un mensaje a partir del estado actual
# Además de generar un mensaje, este nodo también puede llamar a toolnodes, dependiendo del mensaje generado.
def prompt_node(state: State) -> State:
    new_message = llm_with_tools.invoke(state["messages"])
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


print("=================================================\nINICIO DE LA COMPILACION\n=================================================")
APP = graph.compile()

new_state = APP.invoke({"messages": ["Dime cual es la suma del 1 al 10 y el clima en Yorkshire."]})

print(new_state["messages"][-1].content)
print("=================================================\nFIN DE LA COMPILACION\n=================================================")


print("=================================================\nINICIO DEL GRAFO\n=================================================")
from langchain_core.runnables.graph import MermaidDrawMethod
print(APP.get_graph().draw_mermaid())
# Pegan el resultado en https://mermaid.live/ para visualizar el grafo.
print("=================================================\nFIN DEL GRAFO\n=================================================")