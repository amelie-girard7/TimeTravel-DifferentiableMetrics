import uuid
import logging
from typing import List, Dict

import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.mle.utils.config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_docs(df: pd.DataFrame) -> List[Document]:
    """Convert DataFrame to documents with robust metadata handling"""
    
    docs = []
    required_fields = ['premise', 'initial', 'original_ending', 'counterfactual', 'edited_ending', 'story_id']
    
    for idx, row in df.iterrows():
        try:
                
            # Clean and prepare content
            premise = str(row['premise']).strip()
            initial = str(row['initial']).strip()
            original_ending = str(row['original_ending']).strip()
            counterfactual = str(row['counterfactual']).strip()
            edited_ending = str(row['edited_ending']).strip()
            story_id = str(row['story_id']).strip()

            
            # Create document with consistent structure
            doc_content = (
                f"Premise: {premise}\n"
                f"Initial: {initial}\n"
                f"Original ending: {original_ending}\n"
                f"Counterfactual: {counterfactual}\n"
                f"Edited ending: {edited_ending}"
            )
            
            doc_metadata = {
                "story_id": story_id,
                "premise": premise,
                "initial": initial,
                "original_ending": original_ending,
                "counterfactual": counterfactual,
                "edited_ending": edited_ending
            }
            
            doc = Document(
                page_content=doc_content,
                metadata=doc_metadata
            )
                
            docs.append(doc)
            
        except Exception as e:
            print(f"!! Error loading story {row.get('story_id', '<no-id>')}: {str(e)}")
            continue
            
    
    return docs

def build_vector_store(docs: List[Document], persist_path: str):
    """
    Creates and persists a Chroma vector store with extensive metadata validation.
    """
    
    # 1. METADATA VALIDATION PHASE 
    required_fields = ['story_id', 'premise', 'initial', 
                     'original_ending', 'counterfactual', 'edited_ending']
    
    metadata_issues = 0
    for i, doc in enumerate(docs):
        # Check for missing fields
        missing_fields = [f for f in required_fields if f not in doc.metadata]
        if missing_fields:
            metadata_issues += 1
            for field in missing_fields:
                doc.metadata[field] = ""
        else:
            print("  All required fields present")


    # 2. STORE CREATION PHASE 
    try:
        embeddings = OpenAIEmbeddings(model=CONFIG["rag"]["embedding_model"])
        
        store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_path
        )
    except Exception as e:
        print("\n!! CRITICAL ERROR IN STORE CREATION !!")
        raise

    # 3. VERIFICATION PHASE 
    print("\n[PHASE 3] Verifying metadata persistence...")
    try:
        test_retriever = store.as_retriever(search_kwargs={"k": 1})
        test_docs = test_retriever.get_relevant_documents("test")
        
        if not test_docs:
            print("!! WARNING: No documents retrieved in test query")
        else:
            print("\nRetrieved test document metadata:")
            for field in required_fields:
                exists = field in test_docs[0].metadata
                print(f"- {field}: {'PRESENT' if exists else 'MISSING'}")
                if exists:
                    val = str(test_docs[0].metadata[field])
            
    except Exception as e:
        print("\n!! VERIFICATION FAILED (but store was created)")
    return store

def make_rag_chain(persist_path: str, k: int = 1):
    """
    Creates a RAG chain that:
    1. Retrieves similar examples from vector store
    2. Formats them with the current story into a prompt
    3. Generates an adapted ending using the LLM
    
    Args:
        persist_path: Path to persisted Chroma vector store
        k: Number of examples to retrieve
        
    Returns:
        A function that takes an input dict and returns the generated text
    """
    
    # 1) Retriever Setup 
    embeddings = OpenAIEmbeddings(model=CONFIG["rag"]["embedding_model"])
    
    try:
        store = Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"!! CRITICAL ERROR: Failed to initialize Chroma store")
        raise
    
    retriever = store.as_retriever(search_kwargs={"k": k})

    # 2) Prompt Template Setup 
    prompt_template = """Generate the adapted ending to fill these three aspects:
1. Minimal Intervention: Adjust the story's original ending with the minimal changes required to align it with the counterfactual event. The edited ending should remain as close as possible to the original ending.
2. Narrative Insight: Understand the story structure and make changes essential for maintaining the story's coherence and thematic consistency, avoiding unnecessary alterations.
3. Counterfactual Adaptability: Adapt the story's course in response to the counterfactual event that diverges from the initial event.

Here are relevant examples from our dataset:
{context}

Current Story:
Premise: {premise}
Initial event: {initial}
Original ending: {original_ending}
Counterfactual event: {counterfactual}

Now, generate the adapted ending:
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "premise",
            "initial",
            "original_ending",
            "counterfactual"
        ]
    )

    # 3) LLM Setup 
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.0,
        max_tokens=250
    )

    # Wrapped Chain Function 
    def wrapped_chain(input_dict: Dict[str, str]) -> str:
        """
        Inner function that handles the actual RAG process for a single story
        
        Args:
            input_dict: Contains story components with keys:
                - story_id
                - premise
                - initial
                - original_ending
                - counterfactual
                
        Returns:
            Generated adapted ending or error message
        """
        # Input Validation and Logging
        story_id = input_dict.get("story_id", "<no-id>")


        # Build Retrieval Query
        query = (
            f"{input_dict['premise']} "
            f"{input_dict['initial']} "
            f"{input_dict['original_ending']} "
            f"{input_dict['counterfactual']}"
        )

        try:
            # Document Retrieval 
            docs = retriever.get_relevant_documents(query)
            
            examples = []
            for i, doc in enumerate(docs, 1):
                
                # Initialize fields with default missing values
                fields = {
                    'premise': "<missing>",
                    'initial': "<missing>", 
                    'original_ending': "<missing>",
                    'counterfactual': "<missing>",
                    'edited_ending': "<missing>",
                    'story_id': "<no-id>"
                }
                
                # Extract from Metadata 
                if hasattr(doc, 'metadata') and doc.metadata:
                    for field in fields:
                        if field in doc.metadata:
                            fields[field] = doc.metadata[field]
                            print(f"  {field}: {str(doc.metadata[field])[:50]}...")
                        else:
                            print(f"  !! Missing {field} in metadata")
                else:
                    print("!! No metadata available in document")
                
                # Fallback to Content Parsing
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    print(f"\nCONTENT ANALYSIS (length: {len(content)} chars)")
                    
                    # Parse fields from content if missing from metadata
                    content_fields = {
                        'Premise:': 'premise',
                        'Initial:': 'initial',
                        'Original ending:': 'original_ending',
                        'Counterfactual:': 'counterfactual',
                        'Edited ending:': 'edited_ending'
                    }
                    
                else:
                    print("!! Document has no page_content")
                
                # Store Example 
                examples.append(
                    f"Example {i} (ID: {fields['story_id']}):\n"
                    f"Premise: {fields['premise']}\n"
                    f"Initial: {fields['initial']}\n"
                    f"Original ending: {fields['original_ending']}\n"
                    f"Counterfactual: {fields['counterfactual']}\n"
                    f"Edited ending: {fields['edited_ending']}"
                )
                

            context_str = "\n\n".join(examples)
            

            # Prompt Construction 
            try:
                full_prompt = PROMPT.format(
                    context=context_str,
                    premise=input_dict["premise"],
                    initial=input_dict["initial"],
                    original_ending=input_dict["original_ending"],
                    counterfactual=input_dict["counterfactual"]
                )
            except Exception as e:
                return f"ERROR: {str(e)}"

            # LLM Generation 
            try:
                response = llm.invoke(full_prompt)
                generated_text = response.content if hasattr(response, 'content') else str(response)
                return generated_text
            except Exception as e:
                return f"ERROR: {str(e)}"

        except Exception as e:
            return f"ERROR: {str(e)}"
    return wrapped_chain

def run_rag_inference(chain, test_data: pd.DataFrame) -> List[Dict]:
    """
    Run inference with the provided chain and collect results.
    """
    
    results = []
    
    for idx, row in test_data.iterrows():
        story_id = row.get("story_id", str(uuid.uuid4()))
        
        # Check for required fields
        required_fields = ['premise', 'initial', 'original_ending', 'counterfactual']
        missing_fields = [f for f in required_fields if f not in row]
        
        inputs = {
            "story_id": story_id,
            "premise": row["premise"],
            "initial": row["initial"],
            "original_ending": row["original_ending"],
            "counterfactual": row["counterfactual"]
        }
        
        generated = chain(inputs)
        
        result = {
            "story_id": story_id,
            "premise": row["premise"],
            "initial": row["initial"],
            "original_ending": row["original_ending"],
            "counterfactual": row["counterfactual"],
            "edited_ending": row.get("edited_ending", ""),
            "generated_text": generated
        }
        
        results.append(result)

    return results