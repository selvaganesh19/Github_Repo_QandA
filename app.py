import os, shutil, tempfile, re
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from git import Repo
from langchain_openai import AzureChatOpenAI

load_dotenv()
API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOY=os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION=os.getenv("AZURE_OPENAI_VERSION")

ALLOWED_EXT={".py",".ipynb",".md",".txt",".js",".ts",".tsx",".jsx",".java",".kt",".c",".cpp",".cs",".go",".rs",".rb",".php",".sql",".html",".css",".yml",".yaml",".toml",".ini",".json"}
SKIP_DIRS={"node_modules",".git","dist","build","out","venv",".venv","__pycache__",".next",".cache","target","bin","obj",".idea",".vscode"}
MAX_FILE_BYTES=800_000

def clone_repo(url):
    d=Path(tempfile.mkdtemp(prefix=".tmp_repo_")).resolve()
    Repo.clone_from(url, d, depth=1)
    return d

def read_repo_text(repo_dir:Path):
    buf=[]
    for root,dirs,files in os.walk(repo_dir):
        dirs[:]=[x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            p=Path(root)/f
            if p.suffix.lower() in ALLOWED_EXT and p.stat().st_size<=MAX_FILE_BYTES:
                try:
                    txt=p.read_text(encoding="utf-8",errors="ignore")
                    if txt.strip():
                        rel=str(p.relative_to(repo_dir))
                        buf.append(f"\n=== FILE: {rel} ===\n{txt}")
                except Exception:
                    pass
    return "\n".join(buf)

def analyze_repo(url):
    if not url or not re.match(r"^https?://",url.strip()): return None,"Invalid URL"
    repo_dir=None
    try:
        repo_dir=clone_repo(url.strip())
        text=read_repo_text(repo_dir)
        if not text.strip(): return None,"No readable text files found"
        return text,"Ready. Repo text loaded."
    except Exception as e:
        return None,f"Error: {e}"
    finally:
        if repo_dir and Path(repo_dir).exists(): shutil.rmtree(repo_dir,ignore_errors=True)

def make_llm(temp=0.8):
    return AzureChatOpenAI(
        deployment_name=CHAT_DEPLOY,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
        temperature=temp
    )

def generate_qa_from_context(repo_text, n_questions=10):
    llm=make_llm(0.2)
    sys=("You are a principal engineer conducting a rigorous technical interview about a specific GitHub repository. Ask only realistic, challenging interview questions directly grounded in the repo's actual code, configuration, and design choices. Focus on probing the candidate’s reasoning behind trade-offs. Why they chose one tool, library, or framework over alternatives. Why this approach is better or worse compared to others. How decisions impact performance, scalability, testing, security, and maintainability. Do not ask generic or surface-level question Keep questions precise, technical, and focused on why this, not that reasoning.")
    fmt=("Using only the repo context below, produce {n} Q&A pairs.\n"
         "Context:\n\"\"\"\n{ctx}\n\"\"\"\n"
         "Format strictly:\nQ1: ...\nA1: ...\nQ2: ...\nA2: ...\n... up to Q{n}/A{n}.\n"
         "Do not add extra commentary.")
    prompt=fmt.format(n=n_questions,ctx=repo_text[:100000])
    res=llm.invoke([{"role":"system","content":sys},{"role":"user","content":prompt}])
    return res.content

def ask_one(repo_text, topic):
    llm=make_llm(0.2)
    ctx=repo_text[:6000]
    sys=("You are a senior interviewer. Ask ONE tough, repo-specific question, then give the detailed answer.")
    usr=(f"Repo context:\n\"\"\"\n{ctx}\n\"\"\"\n"
         f"Focus: {topic or 'most critical part of this repository'}\n"
         "Output format:\nQ: <question>\nA: <answer>\nNo preamble.")
    res=llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}])
    return res.content

def on_analyze(url):
    repo_text,status=analyze_repo(url)
    return repo_text,status

def on_generate(repo_text, n):
    if not repo_text: return "Please analyze a repo first."
    return generate_qa_from_context(repo_text,int(n))

def on_ask_one(repo_text, topic):
    if not repo_text: return "Please analyze a repo first."
    return ask_one(repo_text, topic or "")

with gr.Blocks(title="Repo Interview Prep · Azure OpenAI (No Embeddings)") as demo:
    gr.Markdown("# Repo Interview Prep\nPaste a GitHub repo URL. Get **real interview questions** grounded in its code, with **model answers**. (No embeddings used)")
    repo_state=gr.State()
    with gr.Row():
        repo_url=gr.Textbox(label="GitHub repo URL",placeholder="https://github.com/owner/repo")
    with gr.Row():
        analyze_btn=gr.Button("Analyze Repo")
        analyze_status=gr.Markdown()
    with gr.Row():
        nq=gr.Slider(5,20,step=1,value=10,label="Number of Q&A")
        gen_btn=gr.Button("Generate Q&A")
    qa_out=gr.Markdown(label="Q&A")
    gr.Markdown("### Ask One Question")
    with gr.Row():
        topic=gr.Textbox(label="Optional focus (e.g., auth, DB, CI/CD)")
        ask_btn=gr.Button("Ask One")
    single_out=gr.Markdown()

    analyze_btn.click(on_analyze,inputs=[repo_url],outputs=[repo_state,analyze_status])
    gen_btn.click(on_generate,inputs=[repo_state,nq],outputs=[qa_out])
    ask_btn.click(on_ask_one,inputs=[repo_state,topic],outputs=[single_out])

if __name__=="__main__":
    demo.launch(debug=True)