import os
import streamlit as st
from crewai import Task, Crew, Agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import fitz
from tqdm import tqdm
from dotenv import load_dotenv
import gc
from datetime import datetime
from docx import Document
import re

# Set page config first
st.set_page_config(
    page_title="Agent 1 - Scientist",
    page_icon="🧠",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize LLM
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

# Initialize embeddings
@st.cache_resource
def init_embeddings():
    return OpenAIEmbeddings()

def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def process_single_pdf(file_path, chunk_size=300):
    try:
        with fitz.open(file_path) as doc:
            text_chunks = []
            for page in doc:
                text = page.get_text()
                words = text.split()
                for i in range(0, len(words), chunk_size//2):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if len(chunk.strip()) > 50:
                        text_chunks.append(chunk)
                gc.collect()
        return {"filename": os.path.basename(file_path), "chunks": text_chunks}
    except Exception as e:
        st.error(f"Error processing {file_path}: {str(e)}")
        return None

@st.cache_data
def process_pdfs():
    pdf_directory = "neurology-knowledge-base"
    processed_texts = []
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, filename in enumerate(pdf_files):
        status_text.text(f"Processing {filename}...")
        file_path = os.path.join(pdf_directory, filename)
        result = process_single_pdf(file_path)
        if result:
            processed_texts.append(result)
        progress_bar.progress((idx + 1) / len(pdf_files))
    
    status_text.text("Processing complete!")
    return processed_texts

@st.cache_data
def search_knowledge_base(query):
    # Limit the number of results returned
    docs = st.session_state.db.similarity_search(query, k=3)
    return "\n".join(doc.page_content for doc in docs)

def generate_citation(text):
    try:
        docs = st.session_state.db.similarity_search(text, k=1)
        if docs:
            metadata = docs[0].metadata
            source = metadata.get('source', '').replace('.pdf', '')
            today = datetime.now().strftime("%d %B %Y")
            citation = f"{source}. Accessed {today}."
            return citation
    except Exception as e:
        return f"Unable to generate citation: {str(e)}"

def summarize_text(text, max_length=150):
    try:
        docs = st.session_state.db.similarity_search(text)
        if docs:
            summary_prompt = f"""Summarize the following scientific text while maintaining accuracy 
                and key technical details (maximum {max_length} words):
                
                {text}
                
                Context from knowledge base:
                {docs[0].page_content}
                """
            summary = st.session_state.llm.predict(summary_prompt)
            return summary
    except Exception as e:
        return f"Unable to summarize text: {str(e)}"

def initialize_vector_store():
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False

    if not st.session_state.db_initialized:
        with st.spinner("Processing PDFs and initializing knowledge base..."):
            documents = process_pdfs()
            all_chunks = []
            all_metadata = []
            for doc in documents:
                for chunk in doc['chunks']:
                    all_chunks.append(chunk)
                    all_metadata.append({'source': doc['filename']})
            
            embeddings = init_embeddings()
            
            st.session_state.db = Chroma.from_texts(
                texts=all_chunks,
                embedding=embeddings,
                metadatas=all_metadata,
                persist_directory="./chroma_db"
            )
            st.session_state.db_initialized = True

def create_scientist(llm):
    return Agent(
        role="Scientist",
        goal="Act as a PhD-level expert in neurobiology to verify factual accuracy of content.",
        backstory="""You are a highly knowledgeable scientist specializing in neurobiology, 
            cognitive neuroscience, and behavioral neuroscience. Your primary task is to fact-check 
            content against a verified knowledge base, correcting inaccuracies and illogical, unfactful statements""",
        constraints=[
            "Do not fabricate citations—only use references from the provided knowledge base.",
            "Maintain scientific accuracy",
            "Avoid personal opinions or speculative conclusions.",
            "Cite sources explicitly when correcting or verifying claims.",
        ],
        input_format="""Any form of content that may contain neurobiology-related claims. 
            The text could reference brain functions, neuroscience experiments, 
            cognitive processes, or psychological phenomena, some of which may be inaccurate.""",
        output_format="""Highlight the incorrect and inaccurate stetement and give correct and factual statement.
            Along with the corrections, provide a concise report listing:
            - **Corrected statements** should be wrapped in **bold**.
            - **Incorrect statements** should be wrapped in ~~strikethrough~~
            - Any speculative/unverified claims that were flagged.
            Scientific concepts should be clearly explained with proper references. 
            Keep the tone academic, ensuring all revisions enhance clarity and accuracy.""",
        system_prompt="""You are an expert neurobiologist reviewing content that contains scientific 
            elements. Your goal is to fact-check and point out the incorrect statement and correctly revise it. 
            For each scientific claim:
            - If correct, leave it unchanged but mention, the statement is c.
            - If incorrect, replace it with an accurate and right explanation.
            - If unverifiable, flag it and suggest a more rigorous way to phrase it.
            - If a key scientific detail is missing, propose an addition with supporting evidence.
            - Ensure all neurobiological terminology is precise and aligns with academic conventions.
            Use a neutral, academic tone when correcting errors but ensure that the content remains 
            readable.""",
    
    )

def highlight_corrections(text):
    """Highlight incorrect statements in red and corrected statements in green."""
    # Highlight corrected statements in green
    text = re.sub(r'\*\*(.*?)\*\*', r'<span style="color: green;"><b>\1</b></span>', text)  # Bold + Green for corrections
    # Highlight incorrect statements in red
    text = re.sub(r'\~\~(.*?)\~\~', r'<span style="color: red;"><b>\1</b></span>', text)  # Bold + Red for incorrect
    return text

def main():
    st.title("Agent 1 - Scientist")
    
    # Initialize components
    llm = init_llm()
    embeddings = init_embeddings()
    
    # Initialize vector store
    initialize_vector_store()
    
    # Create scientist agent
    scientist = create_scientist(llm)
    
    # Correctly create the tab
    tab1 = st.tabs(["Transcript Analysis"])[0]  # Get the first tab

    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # File uploader for transcript
            uploaded_file = st.file_uploader("Upload your transcript (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
            text = ""

            if uploaded_file is not None:
                # Read the content of the uploaded file
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    text = uploaded_file.read().decode("utf-8")
                else:
                    st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
                
                st.text_area("Transcript Content", value=text, height=300)

            # Text area for manual input
            manual_input = st.text_area("Or enter your scientific content for validation:", height=300, placeholder="Paste your neuroscience-related content here...")

            analyze_button = st.button(" Analyze Content", type="primary")
        
        with col2:
            if analyze_button:
                # Use text from uploaded file or manual input
                content_to_analyze = text if text else manual_input
                
                if content_to_analyze:
                    with st.spinner("Analyzing content..."):
                        task = Task(
                            description=f"Analyze this content: {content_to_analyze}",
                            expected_output="Corrected text with corrections and citations",
                            agent=scientist
                        )
                        
                        crew = Crew(
                            agents=[scientist],
                            tasks=[task],
                            verbose=True
                        )
                        
                        result = crew.kickoff()
                        
                        # Print the result to inspect its structure
                        print(result)  # Add this line to see the structure of the CrewOutput object

                        # Determine how to extract the corrected text based on the type of result
                        if isinstance(result, str):
                            styled_result = highlight_corrections(result)  # If result is a string
                        elif hasattr(result, "content"):
                            styled_result = highlight_corrections(result.content)  # If result has a 'content' attribute
                        elif hasattr(result, "text"):
                            styled_result = highlight_corrections(result.text)  # If result has a 'text' attribute
                        else:
                            styled_result = highlight_corrections(str(result))  # Fallback to string conversion

                        st.markdown(styled_result, unsafe_allow_html=True)  # Render styled text
                else:
                    st.warning("Please upload a transcript to analyze.")
            else:
                st.info("Upload a transcript or enter content and click 'Analyze Content' to receive feedback.")

if __name__ == "__main__":
    main()
