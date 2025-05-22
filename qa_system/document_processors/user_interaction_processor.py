import os
import glob
import logging
from qa_system.config import get_config
from qa_system.query.contextualizer import generate_context

logger = logging.getLogger(__name__)

class UserInteractionProcessor:
    """
    Processes user interaction logs to generate contextual summaries for each day's Q&A pairs.
    Reads all .md files in USER_INTERACTION_DIRECTORY and writes context files to DOCUMENT_PROCESSING.USER_CONTEXT_DIRECTORY.
    """
    def __init__(self, config=None):
        self.config = config or get_config()
        self.input_dir = self.config.get_nested('USER_INTERACTION_DIRECTORY', './data/user_interaction/')
        self.output_dir = self.config.get_nested('DOCUMENT_PROCESSING.USER_CONTEXT_DIRECTORY', './docs_user_context/')
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self):
        """Process all user interaction files and generate context files."""
        logger.info(f"Processing user interaction logs from: {self.input_dir}")
        all_files = sorted(glob.glob(os.path.join(self.input_dir, '*.md')))
        
        for input_file in all_files:
            logger.info(f"Processing file: {input_file}")
            output_file = os.path.join(self.output_dir, os.path.basename(input_file))
            
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Generate context using the contextualizer
            context = generate_context(text)
            
            # Write output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(context)
            
            logger.info(f"Wrote contextual entries to {output_file}") 