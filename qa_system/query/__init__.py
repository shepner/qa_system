class QueryProcessor:
    def __init__(self, config): pass
    def process_query(self, query):
        class DummyResponse:
            text = ""
            sources = []
        return DummyResponse()
