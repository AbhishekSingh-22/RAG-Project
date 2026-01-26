"""
Unified Pipeline Orchestrator for RAG Ingestion

This script automates the complete ingestion flow:
1. PDF Extraction (Text & Images) using DBSCAN
2. Image Description Generation using Gemini
3. Content Chunking (Text & Markdown Descriptions)
4. Vector Embedding & Indexing
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path to import modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "src" / "ingestion"))

# Import pipeline components
from ingestion.pdf_extractor import extract_pdf
from image_description_generator import process_images_directory
from ingestion.text_chunker import chunk_documents
from ingestion.image_description_chunker import chunk_image_descriptions, save_image_chunks_to_json
from ingestion.vector_store import load_chunks_from_json, create_vector_store, VectorStoreConfig, ChromaVectorStore
from ingestion.embeddings import LangChainJinaEmbeddings, EmbeddingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    persist_dir: Path,
    reset_db: bool = False,
    clustering_method: str = "dbscan",
    limit_images: int = None
):
    """
    Run the end-to-end ingestion pipeline.
    """
    pipeline_start = datetime.now()
    
    # ----------------------------------------------------------------------
    # Step 1: PDF Extraction
    # ----------------------------------------------------------------------
    logger.info("="*60)
    logger.info("STEP 1: PDF EXTRACTION")
    logger.info("="*60)
    
    if not pdf_path.exists():
        logger.error(f"PDF not found at: {pdf_path}")
        return False
        
    try:
        extraction_summary = extract_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            clustering_method=clustering_method
        )
        logger.info("PDF Extraction completed successfully.")
    except Exception as e:
        logger.error(f"PDF Extraction failed: {e}")
        return False
        
    # ----------------------------------------------------------------------
    # Step 2: Image Description Generation
    # ----------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("STEP 2: IMAGE DESCRIPTION GENERATION")
    logger.info("="*60)
    
    images_dir = output_dir / "images"
    descriptions_dir = output_dir / "image_descriptions"
    
    if images_dir.exists() and any(images_dir.iterdir()):
        try:
            desc_results = process_images_directory(
                images_dir=str(images_dir),
                output_dir=str(descriptions_dir),
                max_images=limit_images
            )
            logger.info(f"Generated descriptions for {desc_results['successful']} images.")
        except Exception as e:
             logger.error(f"Image description generation failed: {e}")
             # We might want to continue even if image descriptions fail, 
             # but strictly speaking this is a failure for "multimodal" pipeline.
             # forcing return False to ensure quality.
             return False
    else:
        logger.warning("No images extracted or directory empty. Skipping description generation.")
        
    # ----------------------------------------------------------------------
    # Step 3: Content Chunking
    # ----------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("STEP 3: CONTENT CHUNKING")
    logger.info("="*60)
    
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # 3a. Text Chunking
    text_dir = output_dir / "text" # Note: extract_pdf saves text files here BUT we need to rename to .md or handle .txt
    # The text_chunker looks for *.md by default. 
    # Let's check where extract_pdf saves text. It saves as .txt in text_dir.
    # We need to make sure text_chunker works with .txt or rename them.
    # Looking at run_chunking.py it expects markdown files in input_dir.
    # But extract_pdf produces .txt files.
    # Let's handle .txt files in text_chunker invocation or rename here. 
    # For now, let's assume text_chunker can handle *.* if passed, or we rename.
    # To be safe, let's rename .txt to .md as the content is just text.
    
    if text_dir.exists():
        for txt_file in text_dir.glob("*.txt"):
            md_file = txt_file.with_suffix(".md")
            txt_file.replace(md_file)
            
    text_docs = []
    if text_dir.exists():
        text_docs, text_stats = chunk_documents(
            input_dir=text_dir,
            file_pattern="*.md"
        )
        logger.info(f"Chunked {len(text_docs)} text segments.")
    
    # 3b. Image Description Chunking
    image_docs = []
    if descriptions_dir.exists():
        image_docs, image_stats = chunk_image_descriptions(
            input_dir=descriptions_dir
        )
        # Save image chunks for review/debugging
        image_chunks_dir = output_dir / "image_chunks_review"
        save_image_chunks_to_json(image_docs, image_chunks_dir)
        logger.info(f"Chunked {len(image_docs)} image descriptions.")

    all_docs = text_docs + image_docs
    logger.info(f"Total chunks to embed: {len(all_docs)}")
    
    if not all_docs:
        logger.error("No content available for embedding. Aborting.")
        return False
        
    # ----------------------------------------------------------------------
    # Step 4: Vector Embedding & Indexing
    # ----------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("STEP 4: VECTOR EMBEDDING")
    logger.info("="*60)
    
    try:
        # Initialize Embeddings
        embed_config = EmbeddingConfig()
        embeddings = LangChainJinaEmbeddings(embed_config)
        
        # Initialize Vector Store
        store_config = VectorStoreConfig(
            collection_name="rag_documents",
            persist_directory=str(persist_dir),
            embedding_dimension=768 # MPNet
        )
        
        vector_store = ChromaVectorStore(
            config=store_config, 
            embeddings=embeddings
        )
        vector_store.initialize()
        
        if reset_db:
            logger.info("Resetting vector collection...")
            vector_store.delete_collection()
            # Re-initialize after deletion
            vector_store = ChromaVectorStore(
                config=store_config, 
                embeddings=embeddings
            )
            vector_store.initialize()
            
        # Add documents
        vector_store.add_documents(
            documents=all_docs,
            batch_size=8,
            show_progress=True
        )
        
        final_stats = vector_store.get_collection_stats()
        logger.info(f"Embedding complete. Collection size: {final_stats['document_count']}")
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return False
        
    total_duration = datetime.now() - pipeline_start
    logger.info("\n" + "="*60)
    logger.info(f"PIPELINE COMPLETED in {total_duration}")
    logger.info("="*60)
    return True

def main():
    parser = argparse.ArgumentParser(description="Unified RAG Ingestion Pipeline")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to ingest")
    parser.add_argument("--output", type=str, default="temp_extraction_pipeline", help="Output directory for intermediate files")
    parser.add_argument("--db-path", type=str, default="data/vectorstore", help="Path to ChromaDB storage")
    parser.add_argument("--reset", action="store_true", help="Reset the vector database before indexing")
    parser.add_argument("--method", type=str, default="dbscan", choices=["dbscan", "proximity", "hierarchical"], help="Clustering method for image extraction")
    parser.add_argument("--limit", type=int, help="Limit number of images to process (for testing)")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path).resolve()
    project_root = Path(__file__).parent.parent
    
    # Handle relative paths for output/db
    if Path(args.output).is_absolute():
        output_dir = Path(args.output)
    else:
        output_dir = project_root / args.output
        
    if Path(args.db_path).is_absolute():
        persist_dir = Path(args.db_path)
    else:
        persist_dir = project_root / args.db_path
        
    logger.info(f"Starting pipeline for: {pdf_path}")
    
    run_pipeline(
        pdf_path=pdf_path,
        output_dir=output_dir,
        persist_dir=persist_dir,
        reset_db=args.reset,
        clustering_method=args.method,
        limit_images=args.limit
    )

if __name__ == "__main__":
    main()
