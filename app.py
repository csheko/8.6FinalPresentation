# ─────────────────────────  app.py  ───────────────────────────────────────────
"""
Business-Intelligence Assistant
• Upload any CSV → auto-loads into SQLite
• Ask questions → agent answers with SQL + Python
• Charts are saved as PNG files in static/charts/ and displayed inline
"""

import os, uuid                               # uuid used both here and in agent code
from pathlib import Path

# ── Prevent GUI pop-ups from Matplotlib
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
from flask import Flask, request, jsonify, render_template, session
from sqlalchemy import create_engine
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# ── Environment / folder setup ────────────────────────────────────────────────
load_dotenv()

STATIC_CHART_DIR = Path("static/charts")
STATIC_CHART_DIR.mkdir(parents=True, exist_ok=True)  # ensure charts dir exists

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

UPLOAD_DIR = "./db_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_key, model_name="gpt-4o", temperature=0.3)

# ── Per-session memory cache ──────────────────────────────────────────────────
_USER_MEMORY: dict[str, ConversationBufferMemory] = {}  # {sid: memory}


def _get_session_id() -> str:
    """Return a stable UUID stored in the Flask session cookie."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
        session.modified = True
    return session["sid"]


# ── Agent instructions ────────────────────────────────────────────────────────
AGENT_PREFIX = f"""
You are a Business-Intelligence assistant with two tools:

1. query_sql_db – run SQL on the current SQLite database.
2. python_repl  – execute Python for statistics, ML, or charting.

╭─ If a question requires numbers
│   • Run SQL first, then do calculations in python_repl and embed results.
│
├─ If a question asks for a graph
│   • In python_repl:
│       import matplotlib.pyplot as plt, uuid, os
│       fname = f"charts/{{uuid.uuid4().hex}}.png"
│       full_path = os.path.join("static", fname)
│       plt.savefig(full_path, bbox_inches="tight")      # Never call plt.show()
│       print(f'<img src="/static/{{fname}}" style="max-width:100%;height:auto;" />')
│       plt.close()
│   • Do NOT print anything else.
│
└─ Always finish with **Final Answer:** containing
    • A concise narrative of findings,
    • (optional) the <img …> tag on its own line,
    • A **Next-Steps Recommendation** bullet list.

Never reveal internal stack traces unless asked explicitly.
"""

AGENT_FORMAT = """
Question: <user input>
Thought: <reasoning>
Action: <tool name>
Action Input: <input for tool>
Observation: <tool result>
... (repeat as needed)
Final Answer: <clear, user-facing explanation>
"""

# ── Python REPL tool definition ───────────────────────────────────────────────
python_tool = PythonAstREPLTool(
    name="python_repl",
    description=(
        "Execute Python for KPI calculations, statistics, ML, or matplotlib "
        "charts. When graphing, save the figure to static/charts/ and PRINT one "
        "<img> tag pointing to that PNG."
    ),
)

# ── Helper: build / reuse agent bound to a SQLite DB ──────────────────────────
def create_agent_for_db(db_uri: str):
    sid = _get_session_id()
    memory = _USER_MEMORY.get(sid)
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        _USER_MEMORY[sid] = memory

    db = SQLDatabase.from_uri(db_uri)
    sql_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()
    tools = sql_tools + [python_tool]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        max_iterations=25,
        memory=memory,
        agent_kwargs={
            "prefix": AGENT_PREFIX,
            "format_instructions": AGENT_FORMAT,
        },
    )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form.get("question", "")
    file_obj = request.files.get("file")

    # Handle optional CSV upload
    if file_obj:
        try:
            filename = f"{uuid.uuid4().hex}_{file_obj.filename}"
            csv_path = os.path.join(UPLOAD_DIR, filename)
            file_obj.save(csv_path)

            df = pd.read_csv(csv_path)
            table_name = Path(file_obj.filename).stem.replace(" ", "_")
            db_path = os.path.join(UPLOAD_DIR, f"{table_name}.db")

            engine = create_engine(f"sqlite:///{db_path}")
            df.to_sql(table_name, con=engine, if_exists="replace", index=False)

            session["db_path"] = db_path
            dataset_map = session.get("dataset_map", {})
            dataset_map[table_name] = db_path
            session["dataset_map"] = dataset_map
            session.modified = True

        except Exception as e:
            return jsonify({"answer": f"<em>Error processing uploaded file: {e}</em>"})

    if "db_path" not in session:
        return jsonify({"answer": "No dataset uploaded yet."})

    # Invoke the agent
    try:
        agent = create_agent_for_db(f"sqlite:///{session['db_path']}")
        result = agent.invoke({"input": question})
        return jsonify({"answer": result["output"]})
    except Exception as e:
        return jsonify({"answer": f"<em>Error: {e}</em>"})


@app.route("/reset", methods=["POST"])
def reset():
    _USER_MEMORY.pop(session.get("sid", ""), None)
    session.clear()
    return jsonify({"message": "Session reset."})

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
