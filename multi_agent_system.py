# ===============================
# Multi-Agent System using LangGraph
# ===============================

from dotenv import load_dotenv
load_dotenv()  # Loads OPENAI_API_KEY from .env

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -------------------------------
# Agent Definitions
# -------------------------------

def planner_agent(state: dict) -> dict:
    """Breaks the user query into a clear plan"""
    response = llm.invoke(
        f"Create a clear step-by-step plan for the following task:\n{state['input']}"
    )
    return {"plan": response.content}


def research_agent(state: dict) -> dict:
    """Researches information based on the plan"""
    response = llm.invoke(
        f"Provide technical research and explanation for this plan:\n{state['plan']}"
    )
    return {"research": response.content}


def coding_agent(state: dict) -> dict:
    """Generates code based on research"""
    response = llm.invoke(
        f"Write clean and correct Python code based on this research:\n{state['research']}"
    )
    return {"code": response.content}


# -------------------------------
# LangGraph Workflow
# -------------------------------

graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("researcher", research_agent)
graph.add_node("coder", coding_agent)

graph.set_entry_point("planner")

graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "coder")

app = graph.compile()

# -------------------------------
# Run the Multi-Agent System
# -------------------------------

if __name__ == "__main__":
    user_input = "Build a REST API using FastAPI"

    result = app.invoke({"input": user_input})

    print("\n===== FINAL OUTPUT =====\n")
    print("USER INPUT:\n", result["input"])
    print("\nPLAN:\n", result["plan"])
    print("\nRESEARCH:\n", result["research"])
    print("\nCODE:\n", result["code"])
