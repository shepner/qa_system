# Document Chunking Best Practices for Embeddings

## Overview

This document summarizes best practices and advanced strategies for chunking documents for embedding and retrieval-augmented generation (RAG) systems. The goal is to maximize the semantic coherence of each chunk—ideally, each chunk should represent a single topic or idea, which improves both embedding quality and downstream retrieval accuracy.

---

## 1. Semantic-Aware Chunking (Dynamic Chunking)

**Best Practice:**
Chunk documents at natural semantic boundaries (e.g., paragraphs, sections, or even sentences), not just by character or token count.

- **Why:** Embeddings are most effective when each chunk is topically coherent. Arbitrary splits (e.g., every 512 tokens) can break up ideas, leading to embeddings that are less meaningful and harder to retrieve accurately.
- **How:**
  - Use NLP techniques to detect topic shifts, section headers, or paragraph breaks.
  - Use sentence boundary detection and group sentences until a max token/character limit is reached, but never split a sentence.
  - For Markdown, LaTeX, or code, use structural cues (e.g., headings, code blocks, list items).

**References:**
- [LangChain RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
- [OpenAI Cookbook: Chunking strategies](https://cookbook.openai.com/examples/embedding_long_documents)

---

## 2. Adaptive Chunk Sizing

**Best Practice:**
Allow chunk size to vary within a range, optimizing for semantic completeness rather than a fixed size.

- **Why:** Some topics are expressed in a single short paragraph, others require several paragraphs. Forcing all to a fixed size either truncates meaning or pads with unrelated content.
- **How:**
  - Set a **min** and **max** chunk size (e.g., 256–1024 tokens).
  - Start a new chunk when a semantic boundary is reached and the current chunk is at least the minimum size.
  - If a chunk would exceed the max size, split at the nearest previous boundary.

---

## 3. Hierarchical Chunking

**Best Practice:**
Create overlapping or hierarchical chunks to capture context at multiple granularities.

- **Why:** Overlapping chunks (with `CHUNK_OVERLAP`) help preserve context for queries that span chunk boundaries. Hierarchical chunking (e.g., both paragraph and section-level embeddings) allows for multi-scale retrieval.
- **How:**
  - Use a sliding window with overlap (e.g., 20–30% of chunk size).
  - Optionally, store both fine-grained (sentence/paragraph) and coarse-grained (section/chapter) embeddings.

---

## 4. Topic Segmentation Algorithms

**Best Practice:**
Use topic segmentation algorithms (e.g., TextTiling, TopicTiling, or transformer-based models) to detect topic shifts and split accordingly.

- **Why:** These algorithms are designed to find natural topic boundaries, which aligns with the goal of single-topic chunks.
- **How:**
  - Use libraries like [segtok](https://github.com/fnl/segtok), [textsplit](https://github.com/alfredfrancis/textsplit), or transformer-based models for topic segmentation.
  - Integrate with your chunking pipeline to split at detected boundaries.

Notes:
- Chunking will still be required if a topic exceeds the max model input.  In this case, other methods such as Hierarchical Chunking will need to be applied to further subdivide.

---

## 5. Domain-Specific Heuristics

**Best Practice:**
Customize chunking logic for your document types (e.g., legal, scientific, code, Markdown).

- **Why:** Different domains have different structural cues (e.g., legal sections, code functions, Markdown headers).
- **How:**
  - For Markdown: split at headers, lists, code blocks.
  - For code: split at function/class boundaries.
  - For scientific papers: split at section/subsection headings.

---

## 6. Empirical Evaluation and Tuning

**Best Practice:**
Continuously evaluate retrieval quality and tune chunking parameters based on real search results and user feedback.

- **Why:** The optimal chunking strategy is often corpus- and use-case-specific.
- **How:**
  - Run retrieval experiments with different chunking strategies.
  - Measure precision/recall, user satisfaction, and embedding utilization.
  - Adjust chunking logic and parameters accordingly.

---

## Summary Table

| Strategy                  | Pros                                    | Cons/Challenges                |
|---------------------------|-----------------------------------------|-------------------------------|
| Fixed-size chunking       | Simple, fast                            | Breaks semantic units         |
| Semantic-aware chunking   | Preserves meaning, better retrieval     | More complex, needs NLP       |
| Adaptive chunk sizing     | Flexible, fits content                  | Needs careful tuning          |
| Overlapping chunks        | Preserves context                       | Increases storage/compute     |
| Topic segmentation        | Best semantic coherence                 | Harder to implement           |
| Domain-specific chunking  | Highly relevant for structured docs     | Needs custom logic            |

---

## Recommended Approach

1. **Start with sentence/paragraph-based chunking** with min/max size and overlap.
2. **Incorporate topic segmentation** if your documents are long and multi-topic.
3. **Tune min/max/overlap** based on retrieval experiments.
4. **Log and analyze** which chunks are most/least useful for search, and iterate.

---

## References for Further Reading

- [OpenAI Cookbook: Embedding Long Documents](https://cookbook.openai.com/examples/embedding_long_documents)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
- [Haystack: Preprocessing and Chunking](https://docs.haystack.deepset.ai/docs/preprocessing)
- [TextTiling: Text Segmentation](https://www.aclweb.org/anthology/J97-1003.pdf)

---

**TL;DR:**
The best practice is to chunk at semantic boundaries (not just by size), use adaptive chunk sizes, and tune based on retrieval quality. Static chunk sizes based on average file size are almost never optimal for semantic search or RAG. 





# DeepTiling

To begin with DeepTiling, clone the repository and install its required Python packages (e.g., sentence-transformers, numpy) via pip install -r requirements.txt; example usage is shown under "Running the Program" in its README 
GitHub
. For the Unsupervised Topic Segmentation with BERT Embeddings implementation by Damaskinos et al., simply clone the repo, install transformers, and invoke the eval.eval_topic_segmentation entry point on your text—no additional training is needed
GitHub
. The RoBERTa‑based unsupervised podcast segmentation tool provides a requirements.txt and example scripts; after installing dependencies, you can run evaluation.py on any transcript to detect topic boundaries 
GitHub
. The wtpsplit library (“Segment Any Text”) is installable via pip and offers Sentence- and Semantic-Unit segmentation across 85 languages, with clear runtime examples in its docs 
GitHub
; its lightweight counterpart, wtpsplit‑lite, enables fast ONNX-based inference for SaT models 
GitHub
. For multilingual meetings, GruffGemini’s topic_segmentation package lets you call segmentation.segment_text after installing from GitHub 
GitHub
. Finally, the original Unsupervised Topic Segmentation by BERT paper (arXiv:2106.12978) details the algorithm’s theory and evaluation results, providing valuable background before hands‑on experimentation 
arXiv
. Supplemental blog posts offer end‑to‑end code demonstrations and best practices for pipeline integration 
Naveed Afzal
Medium
.

Works Cited

Bird, Steven, Edward Loper, and Ewan Klein. “DeepTiling: A TextTiling‑Based Algorithm for Text Segmentation.” GitHub, 2024, https://github.com/Ighina/DeepTiling.

Damaskinos, Georgios. Unsupervised Topic Segmentation of Meetings with BERT Embeddings. GitHub, 2021, https://github.com/gdamaskinos/unsupervised_topic_segmentation.

Schimmerd. Unsupervised Topic Segmentation of Podcasts Using a Pre‑Trained Transformer Model. GitHub, 2020, https://github.com/schimmerd/unsupervised-topic-segmentation-roberta.

Frohmann, Markus, et al. Segment Any Text (SaT): A Universal Text Segmentation Model. wtpsplit, GitHub, 2024, https://github.com/segment-any-text/wtpsplit.

Superlinear‑AI. wtpsplit‑lite: ONNX‑Accelerated SaT Inference. GitHub, 2024, https://github.com/superlinear-ai/wtpsplit-lite.

GruffGemini. topic_segmentation: Unsupervised Topic Segmentation of Work Meetings. GitHub, 2024, https://github.com/GruffGemini/topic_segmentation.

“Unsupervised Topic Segmentation by BERT Embeddings.” arXiv, 30 June 2021, https://arxiv.org/abs/2106.12978.

Afzal, Naveed. “An Introduction to Unsupervised Topic Segmentation with Implementation.” NaveedAfzal.com, 2022, https://www.naveedafzal.com/posts/an-introduction-to-unsupervised-topic-segmentation-with-implementation/.

Mishra, Prakhar. “Unsupervised Topic Segmentation of Meetings with BERT Embeddings: Summary.” Medium, 2021, https://medium.com/towards-data-science/unsupervised-topic-segmentation-of-meetings-with-bert-embeddings-summary-46e1b7369755.