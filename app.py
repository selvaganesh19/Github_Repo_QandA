import os, shutil, tempfile, re, json, hashlib
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()
API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOY=os.getenv("AZURE_OPENAI_DEPLOYMENT")
EMB_DEPLOY=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

ALLOWED_EXT={".py",".ipynb",".md",".txt",".js",".ts",".tsx",".jsx",".java",".kt",".c",".cpp",".cs",".go",".rs",".rb",".php",".sql",".html",".css",".yml",".yaml",".toml",".ini",".json"}
SKIP_DIRS={"node_modules",".git","dist","build","out","venv",".venv","__pycache__",".next",".cache","target","bin","obj",".idea",".vscode"}
MAX_FILE_BYTES=800_000
INDEX_DIR="faiss_index"

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

def chunk_text(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    return splitter.split_text(text)

def url_to_dir(url):
    h=hashlib.sha1(url.encode()).hexdigest()[:12]
    return str(Path(INDEX_DIR)/h)

def build_vs(chunks, idx_dir):
    emb=AzureOpenAIEmbeddings(deployment=EMB_DEPLOY,azure_endpoint=ENDPOINT,api_key=API_KEY)
    vs=FAISS.from_texts(chunks,emb)
    vs.save_local(idx_dir)
    return idx_dir

def load_vs(idx_dir):
    emb=AzureOpenAIEmbeddings(deployment=EMB_DEPLOY,azure_endpoint=ENDPOINT,api_key=API_KEY)
    return FAISS.load_local(idx_dir,emb,allow_dangerous_deserialization=True)

def make_llm(temp=0.3):
    return AzureChatOpenAI(deployment_name=CHAT_DEPLOY,azure_endpoint=ENDPOINT,api_key=API_KEY,temperature=temp)

def sample_corpus_for_brief(vs, n=30):
    qs=[
        "project purpose and architecture","key modules and dependencies",
        "data models and schemas","APIs and endpoints","build and deploy setup",
        "testing and CI/CD","security and auth","performance and scaling",
        "coding conventions","edge cases and failure modes"
    ]
    docs=[]
    for q in qs:
        docs+=vs.similarity_search(q,k=3)
        if len(docs)>=n: break
    text="\n\n".join(d.page_content for d in docs[:n])
    return text[:100000]

def generate_qa_from_context(vs, n_questions=10):
    context=sample_corpus_for_brief(vs)
    llm=make_llm(0.2)
    sys=("You are a principal engineer conducting a rigorous technical interview about a specific GitHub repository. "
         "Ask targeted, realistic questions grounded in the repo's actual code and config, then give ideal, precise answers. "
         "Focus on architecture, trade-offs, complexity, testing, performance, security, deployment, data models, and pitfalls. "
         "Avoid generic fluff. Keep each answer concise but substantive.")
    fmt=("Using only the repo context below, produce {n} Q&A pairs.\n"
         "Context:\n\"\"\"\n{ctx}\n\"\"\"\n"
         "Format strictly:\nQ1: ...\nA1: ...\nQ2: ...\nA2: ...\n... up to Q{n}/A{n}.\n"
         "Do not add extra commentary.")
    prompt=fmt.format(n=n_questions,ctx=context)
    res=llm.invoke([{"role":"system","content":sys},{"role":"user","content":prompt}])
    return res.content

def ask_one(vs, topic):
    llm=make_llm(0.2)
    docs=vs.similarity_search(topic or "most critical part of this repository",k=6)
    ctx="\n\n".join(d.page_content for d in docs)[:6000]
    sys=("You are a senior interviewer. Ask ONE tough, repo-specific question, then give the ideal answer.")
    usr=(f"Repo context:\n\"\"\"\n{ctx}\n\"\"\"\n"
         "Output format:\nQ: <question>\nA: <answer>\nNo preamble.")
    res=llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}])
    return res.content

def analyze_repo(url):
    if not url or not re.match(r"^https?://",url.strip()): return None,"Invalid URL"
    repo_dir=None
    try:
        repo_dir=clone_repo(url.strip())
        text=read_repo_text(repo_dir)
        if not text.strip(): return None,"No readable text files found"
        chunks=chunk_text(text)
        idx_dir=url_to_dir(url.strip())
        Path(idx_dir).parent.mkdir(parents=True,exist_ok=True)
        build_vs(chunks, idx_dir)
        return idx_dir,"Ready. Vector index built."
    except Exception as e:
        return None,f"Error: {e}"
    finally:
        if repo_dir and Path(repo_dir).exists(): shutil.rmtree(repo_dir,ignore_errors=True)

def on_analyze(url):
    idx_dir,status=analyze_repo(url)
    return idx_dir,status

def on_generate(vs_state, n):
    if not vs_state: return "Please analyze a repo first."
    vs=load_vs(vs_state)
    return generate_qa_from_context(vs,int(n))

def on_ask_one(vs_state, topic):
    if not vs_state: return "Please analyze a repo first."
    vs=load_vs(vs_state)
    return ask_one(vs, topic or "")

with gr.Blocks(title="Repo Interview Prep · Azure OpenAI") as demo:
    gr.Markdown("# Repo Interview Prep\nPaste a GitHub repo URL. Get **real interview questions** grounded in its code, with **model answers**.")
    vs_state=gr.State()
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

    analyze_btn.click(on_analyze,inputs=[repo_url],outputs=[vs_state,analyze_status])
    gen_btn.click(on_generate,inputs=[vs_state,nq],outputs=[qa_out])
    ask_btn.click(on_ask_one,inputs=[vs_state,topic],outputs=[single_out])

if __name__=="__main__":
    demo.launch(debug=True)