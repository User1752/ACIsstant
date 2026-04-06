
import os
import re
import logging
from pathlib import Path
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore

# Silence noisy loaders
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

logger = logging.getLogger("acisstant.rag")

# ---------------------------------------------------------------------------
# Document-type priority (lower number = higher priority in results)
# ---------------------------------------------------------------------------
DOC_TYPE_PRIORITY = {
    "lecture":      0,  # Aula slides — primary theory source
    "notes":        1,  # Apontamentos / summary notes
    "reference":    2,  # Formula tables, transform tables
    "presentation": 3,  # Course presentations / overview
    "exercises":    4,  # Exercise sheets, tests, solutions
    "other":        5,
}

def _classify_filename(filename: str) -> str:
    """Return a doc_type string based on filename patterns."""
    name = filename.lower()

    # Lecture slides: "Aula" followed by a number
    if re.search(r"aula[\s_#]?\d+", name):
        return "lecture"

    # Notes / apontamentos
    if "apontamento" in name:
        return "notes"

    # Formula / transform reference tables
    if any(k in name for k in ["formulário", "formulario", "tabela", "transform"]):
        return "reference"

    # Course presentation / overview
    if "apresenta" in name:
        return "presentation"

    # Exercise sheets, tests, resolutions
    if any(k in name for k in ["exerc", "teste", "resolução", "resolucao", "rms"]):
        return "exercises"

    return "other"


class RAGManager:
    def __init__(self, data_dir: str = "data/uploads", index_dir: str = "data/vectordb"):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / data_dir
        self.index_dir = self.base_dir / index_dir

        # Self-contained model cache inside the project
        self.models_dir = self.base_dir / "data" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(self.models_dir)
        print(f"[RAG] Using local model storage: data/models")

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder=str(self.models_dir)
            )
            self.vector_store = None
            # Auto-scan and build on startup to ensure RAG is never empty
            if self.data_dir.exists():
                valid_files = [
                    f for f in self.data_dir.rglob("*")
                    if f.is_file() and f.suffix in [".pdf", ".md", ".txt"] and f.name != "README.md"
                ]
                if valid_files:
                    logger.info(f"[RAG] Startup scan: {len(valid_files)} documents found in data/uploads.")
                    self.process_documents()
            self.load_index()
        except Exception as e:
            logger.error(f"[RAG] Initialization failed: {e}")
            self.vector_store = None

    # ------------------------------------------------------------------
    def load_index(self):
        try:
            if (self.index_dir / "index.faiss").exists():
                logger.info("[RAG] Loading existing FAISS index from data/vectordb...")
                try:
                    self.vector_store = FAISS.load_local(
                        str(self.index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("[RAG] FAISS index loaded successfully.")
                except Exception as e:
                    logger.error(f"[RAG] Failed to load index ({e}). Rebuilding...")
                    import shutil
                    shutil.rmtree(str(self.index_dir), ignore_errors=True)
                    self.index_dir.mkdir(parents=True, exist_ok=True)
                    self.vector_store = None
                    if self.data_dir.exists():
                        valid_files = [
                            f for f in self.data_dir.rglob("*")
                            if f.is_file() and f.suffix in [".pdf", ".md", ".txt"]
                            and f.name != "README.md"
                        ]
                        if valid_files:
                            self.process_documents()
            else:
                logger.info("[RAG] No index found.")
                if self.data_dir.exists():
                    valid_files = [
                        f for f in self.data_dir.rglob("*")
                        if f.is_file() and f.suffix in [".pdf", ".md", ".txt"] and f.name != "README.md"
                    ]
                    if valid_files:
                        logger.info(
                            f"[RAG] Found {len(valid_files)} unindexed documents in data/uploads. Auto-indexing now..."
                        )
                        self.process_documents()
        except Exception as e:
            logger.error(f"[RAG] load_index failed: {e}")
            self.vector_store = None

    # ------------------------------------------------------------------
    def process_documents(self):
        """Load, classify, chunk and index all documents."""
        logger.info("[RAG] Processing documents in data/uploads...")

        # Collect all files individually so we can tag them with doc_type
        documents = []

        pdf_files = list(self.data_dir.rglob("*.pdf"))
        md_files  = [f for f in self.data_dir.rglob("*.md") if f.name != "README.md"]
        txt_files = list(self.data_dir.rglob("*.txt"))

        for path in pdf_files:
            try:
                loader = PyPDFLoader(str(path))
                pages = loader.load()
                doc_type = _classify_filename(path.name)
                for page in pages:
                    page.metadata["doc_type"] = doc_type
                    page.metadata["filename"] = path.name
                documents.extend(pages)
                logger.info(f"[RAG] Loaded '{path.name}' → type='{doc_type}'")
            except Exception as e:
                logger.error(f"[RAG] Failed to load PDF '{path.name}': {e}")

        for path in md_files + txt_files:
            try:
                loader = TextLoader(str(path), autodetect_encoding=True)
                pages = loader.load()
                doc_type = _classify_filename(path.name)
                for page in pages:
                    page.metadata["doc_type"] = doc_type
                    page.metadata["filename"] = path.name
                documents.extend(pages)
            except Exception as e:
                logger.error(f"[RAG] Failed to load text file '{path.name}': {e}")

        if not documents:
            logger.warning("[RAG] No documents found to index.")
            return

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(str(self.index_dir))  # type: ignore
            logger.info(f"[RAG] Index saved to data/vectordb ({len(docs)} chunks from {len(documents)} pages)")
        except Exception as e:
            logger.error(f"[RAG] Failed to process documents: {e}")
            self.vector_store = None

    # ------------------------------------------------------------------
    def query(self, text: str, k: int = 4) -> Tuple[str, List[str]]:
        """
        Retrieve the top-k most relevant chunks, but prioritise by document
        type so that lecture slides and notes are always preferred over
        exercise sheets and test solutions.

        Strategy:
          1. Fetch a larger candidate pool (k * 5) by raw similarity.
          2. Re-rank by (doc_type_priority, similarity_distance).
          3. Return the best k from the re-ranked list.
        """
        if not self.vector_store:
            logger.warning("[RAG] Query attempted but vector_store is not loaded.")
            return "", []

        try:
            # Fetch a large pool so we have plenty of candidates to re-rank
            candidate_k = min(k * 5, 30)
            raw_results = self.vector_store.similarity_search_with_score(text, k=candidate_k)  # type: ignore
            # raw_results: List[Tuple[Document, float]]  (score = L2 distance — lower is better)

            # Re-rank: primary key = doc_type priority, secondary key = distance (ascending)
            def rank_key(item):
                doc, distance = item
                dtype = doc.metadata.get("doc_type", "other")
                priority = DOC_TYPE_PRIORITY.get(dtype, DOC_TYPE_PRIORITY["other"])
                return (priority, distance)

            ranked = sorted(raw_results, key=rank_key)
            top_docs = [doc for doc, _score in ranked[:k]]

            context = "\n---\n".join([doc.page_content for doc in top_docs])
            sources = list(dict.fromkeys(
                os.path.basename(doc.metadata.get("source", doc.metadata.get("filename", "Unknown")))
                for doc in top_docs
            ))

            type_summary = ", ".join(
                f"{doc.metadata.get('filename', '?')} [{doc.metadata.get('doc_type', '?')}]"
                for doc in top_docs
            )
            logger.info(f"[RAG] Query '{text[:40]}...' → selected: {type_summary}")
            return context, sources

        except Exception as e:
            logger.error(f"[RAG] Query failed: {e}")
            return "", []
