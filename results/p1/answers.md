# Problem 1: answer quality

Generator `groq/openai/gpt-oss-120b`, judge `gemini/gemini-3.1-flash-lite` (different model family), k = 5, hybrid retrieval.

| Metric | Value | Method |
|---|---|---|
| Faithfulness | 1.0 | claim-level: supported claims / all claims (LLM judge), 24 answers |
| Answers with every claim supported | 1.0 | same judge |
| Answer relevance | 1.0 | judge 1-5 rescaled to 0-1 |
| Exact match | 0.0 | SQuAD normalisation vs gold short answer |
| Contains-match | 0.2083 | normalised gold answer appears in the response |
| Token F1 | 0.3067 | SQuAD token F1 vs gold |
| Answered (answerable) | 24/25 | false abstentions: 1 |
| Declined (unanswerable) | 5/5 | guard: 2, model: 3 |
| Answers citing a valid passage | 1.0 | invalid citation markers: 0 |
| Judge control: planted wrong numbers caught | 11/11 | first number in each answer changed; caught = judge marks a claim unsupported |
| End-to-end p50 / p95 | 870.6 / 1646.1 ms | embed + retrieve + generation API latency |
