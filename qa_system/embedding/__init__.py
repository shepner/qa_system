import logging

class EmbeddingGenerator:
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called EmbeddingGenerator.__init__(config={config})")
    def generate_embeddings(self, texts, metadata):
        self.logger.debug(f"Called EmbeddingGenerator.generate_embeddings(texts=<len {len(texts)}>, metadata={metadata})")
        return {'vectors': [], 'texts': [], 'metadata': []}
