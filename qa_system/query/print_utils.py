from .models import Source, QueryResponse

def print_response(response: QueryResponse):
    print("\nAnswer:")
    print("-" * 80)
    print(response.text)
    print("\nSources:")
    print("-" * 80)
    for source in response.sources:
        print(f"- {source.document} (original: {source.original_similarity:.4f}, boost: {source.boost}, final: {source.similarity:.4f})") 