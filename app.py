
import os, uuid
from pathlib import Path

# Azure & headless: force non-GUI backend for matplotlib
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
from flask import Flask, request, jsonify, render_template, session
from sqlalchemy import create_engine
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType

# ── env / folders ────────────────────────────────────────────────────────────
load_dotenv()

STATIC_CHART_DIR = Path("static/charts")
STATIC_CHART_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = "./db_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_key, model_name="gpt-4o", temperature=0.3)

# one ConversationBufferMemory per browser session
_USER_MEMORY: dict[str, ConversationBufferMemory] = {}


def _get_session_id() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
        session.modified = True
    return session["sid"]


# ── Agent prompt (extended) ──────────────────────────────────────────────────
AGENT_PREFIX = r"""
You are a Business-Intelligence assistant with two tools:

1. **query_sql_db** – run SQL on the active SQLite database.
2. **python_repl**  – execute Python for calculations or charts.

╭─ If the user asks for numbers
│   • Run SQL first, then compute with python_repl.
│
├─ If the user asks for a chart
│   • In python_repl:
│       import matplotlib.pyplot as plt, uuid, os
│       fname = f"charts/{uuid.uuid4().hex}.png"
│       full_path = os.path.join("static", fname)
│       plt.savefig(full_path, bbox_inches="tight")
│       plt.close()
│       print(
│         f'<img src="/static/{fname}" '           # leading slash is critical
│         f'style="max-width:100%;height:auto;" />'
│       )
│
└─ In **every Final Answer**
    1. Paste exactly what python_repl printed on its own line
       (that <img> tag or any numeric output).
    2. Follow with a concise plain-English insight and, if applicable,
       a **Next Steps** bullet list.

Never reveal internal stack traces unless explicitly asked.
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

python_tool = PythonAstREPLTool(
    name="python_repl",
    description=(
        "Execute Python for KPI calculations, statistics, ML, or matplotlib "
        "charts. When graphing, save the figure to static/charts/ and PRINT a "
        "single <img> tag pointing to that PNG."
    ),
)

# ── helper: build / reuse agent bound to SQLite DB ───────────────────────────
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

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form.get("question", "")
    file_obj = request.files.get("file")

    # handle optional CSV upload
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


# ── Entrypoint (local) ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
