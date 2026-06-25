from pocketflow import Flow
# Import all node classes from nodes.py
from nodes import (
    FetchRepo,
    SplitDocument,
    SummarizeChunks,
    IdentifyAbstractions,
    AnalyzeRelationships,
    OrderChapters,
    WriteChapters,
    CombineTutorial
)

def create_tutorial_flow():
    """Creates and returns the codebase tutorial generation flow."""

    # Instantiate nodes
    fetch_repo = FetchRepo()
    identify_abstractions = IdentifyAbstractions(max_retries=5, wait=20)
    analyze_relationships = AnalyzeRelationships(max_retries=5, wait=20)
    order_chapters = OrderChapters(max_retries=5, wait=20)
    write_chapters = WriteChapters(max_retries=5, wait=20) # This is a BatchNode
    combine_tutorial = CombineTutorial()

    # Connect nodes in sequence based on the design
    fetch_repo >> identify_abstractions
    identify_abstractions >> analyze_relationships
    analyze_relationships >> order_chapters
    order_chapters >> write_chapters
    write_chapters >> combine_tutorial

    # Create the flow starting with FetchRepo
    tutorial_flow = Flow(start=fetch_repo)

    return tutorial_flow


def create_book_tutorial_flow():
    """Creates the tutorial generation flow for a single large document
    (book/report). It splits the document into structure-aware sections,
    summarizes each section, then reuses the standard abstraction/relationship/
    chapter pipeline."""

    # Instantiate nodes
    split_document = SplitDocument(max_retries=2, wait=5)
    summarize_chunks = SummarizeChunks(max_retries=5, wait=20)  # BatchNode
    identify_abstractions = IdentifyAbstractions(max_retries=5, wait=20)
    analyze_relationships = AnalyzeRelationships(max_retries=5, wait=20)
    order_chapters = OrderChapters(max_retries=5, wait=20)
    write_chapters = WriteChapters(max_retries=5, wait=20)  # BatchNode
    combine_tutorial = CombineTutorial()

    # Connect nodes in sequence
    split_document >> summarize_chunks
    summarize_chunks >> identify_abstractions
    identify_abstractions >> analyze_relationships
    analyze_relationships >> order_chapters
    order_chapters >> write_chapters
    write_chapters >> combine_tutorial

    return Flow(start=split_document)
