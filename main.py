import dotenv
import os
import argparse
# Import the function that creates the flow
from flow import create_tutorial_flow, create_book_tutorial_flow

dotenv.load_dotenv()

# Default file patterns
DEFAULT_INCLUDE_PATTERNS = {
    "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.java", "*.pyi", "*.pyx",
    "*.c", "*.cc", "*.cpp", "*.h", "*.md", "*.rst", "*Dockerfile",
    "*Makefile", "*.yaml", "*.yml",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "assets/*", "data/*", "images/*", "public/*", "static/*", "temp/*",
    "*docs/*",
    "*venv/*",
    "*.venv/*",
    "*test*",
    "*tests/*",
    "*examples/*",
    "v1/*",
    "*dist/*",
    "*build/*",
    "*experimental/*",
    "*deprecated/*",
    "*misc/*",
    "*legacy/*",
    ".git/*", ".github/*", ".next/*", ".vscode/*",
    "*obj/*",
    "*bin/*",
    "*node_modules/*",
    "*.log"
}

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Generate a tutorial for a GitHub codebase or local directory.")

    # Create mutually exclusive group for source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--repo", help="URL of the public GitHub repository.")
    source_group.add_argument("--dir", help="Path to local directory.")
    source_group.add_argument("--file", help="Path to a single large markdown document (book/report) to turn into a tutorial.")

    parser.add_argument("-n", "--name", help="Project name (optional, derived from repo/directory if omitted).")
    parser.add_argument("-t", "--token", help="GitHub personal access token (optional, reads from GITHUB_TOKEN env var if not provided).")
    parser.add_argument("-o", "--output", default="output", help="Base directory for output (default: ./output).")
    parser.add_argument("-i", "--include", nargs="+", help="Include file patterns (e.g. '*.py' '*.js'). Defaults to common code files if not specified.")
    parser.add_argument("-e", "--exclude", nargs="+", help="Exclude file patterns (e.g. 'tests/*' 'docs/*'). Defaults to test/build directories if not specified.")
    parser.add_argument("-s", "--max-size", type=int, default=100000, help="Maximum file size in bytes (default: 100000, about 100KB).")
    # Add language parameter for multi-language support
    parser.add_argument("--language", default="english", help="Language for the generated tutorial (default: english)")
    # Add use_cache parameter to control LLM caching
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching (default: caching enabled)")
    # Add max_abstraction_num parameter to control the number of abstractions.
    # Default None -> auto: 10 for codebases, scaled with size for documents.
    parser.add_argument("--max-abstractions", type=int, default=None, help="Maximum number of abstractions to identify. Default: 10 for codebases; auto-scaled with document size in --file mode.")
    # Chunk sizing for single-document (book/report) mode
    parser.add_argument("--min-chunk-tokens", type=int, default=1500, help="Minimum approx. tokens per section when splitting a document (default: 1500)")
    parser.add_argument("--max-chunk-tokens", type=int, default=10000, help="Maximum approx. tokens per section when splitting a document (default: 10000)")

    args = parser.parse_args()

    # Get GitHub token from argument or environment variable if using repo
    github_token = None
    if args.repo:
        github_token = args.token or os.environ.get('GITHUB_TOKEN')
        if not github_token:
            print("Warning: No GitHub token provided. You might hit rate limits for public repositories.")

    # Initialize the shared dictionary with inputs
    shared = {
        "repo_url": args.repo,
        "local_dir": args.dir,
        "doc_path": args.file,  # Path to a single large document (book/report mode)
        "project_name": args.name, # Can be None, FetchRepo will derive it
        "github_token": github_token,
        "output_dir": args.output, # Base directory for CombineTutorial output

        # Add include/exclude patterns and max file size
        "include_patterns": set(args.include) if args.include else DEFAULT_INCLUDE_PATTERNS,
        "exclude_patterns": set(args.exclude) if args.exclude else DEFAULT_EXCLUDE_PATTERNS,
        "max_file_size": args.max_size,

        # Add language for multi-language support
        "language": args.language,
        
        # Add use_cache flag (inverse of no-cache flag)
        "use_cache": not args.no_cache,
        
        # Add max_abstraction_num parameter (None = auto-decide in IdentifyAbstractions)
        "max_abstraction_num": args.max_abstractions,

        # Single-document (book/report) mode settings
        "content_label": "document" if args.file else "codebase",
        "min_chunk_tokens": args.min_chunk_tokens,
        "max_chunk_tokens": args.max_chunk_tokens,

        # Outputs will be populated by the nodes
        "files": [],
        "summaries": {},
        "abstractions": [],
        "relationships": {},
        "chapter_order": [],
        "chapters": [],
        "final_output_dir": None
    }

    # Display starting message with source and language
    source = args.repo or args.dir or args.file
    print(f"Starting tutorial generation for: {source} in {args.language.capitalize()} language")
    print(f"LLM caching: {'Disabled' if args.no_cache else 'Enabled'}")

    # Create the appropriate flow instance
    if args.file:
        tutorial_flow = create_book_tutorial_flow()
    else:
        tutorial_flow = create_tutorial_flow()

    # Run the flow
    tutorial_flow.run(shared)

if __name__ == "__main__":
    main()
