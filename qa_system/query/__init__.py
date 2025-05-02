import logging

class QueryProcessor:
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(f"Called QueryProcessor.__init__(config={config})")
    def process_query(self, query):
        self.logger.debug(f"Called QueryProcessor.process_query(query={query})")
        class DummyResponse:
            text = ""
            sources = []
        return DummyResponse()
