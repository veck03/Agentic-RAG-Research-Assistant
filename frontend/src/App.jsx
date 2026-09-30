import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();

    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error("The server returned an error.");
      }

      const data = await response.json();

      setResult(data);
    } catch (error) {
      console.error(error);
      setError(
        "Could not connect to the research assistant. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <p className="eyebrow">AGENTIC RESEARCH SYSTEM</p>

          <h1>Ocean Research Assistant</h1>

          <p className="subtitle">
            Ask questions about ocean science, ENSO, climate research,
            and the available ocean dataset.
          </p>
        </div>
      </header>

      <main className="main">
        <form className="query-form" onSubmit={handleSubmit}>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask a research question..."
            rows={4}
          />

          <div className="form-footer">
            <span className="hint">
              Ask about papers, ENSO, or the ocean dataset.
            </span>

            <button type="submit" disabled={loading || !query.trim()}>
              {loading ? "Researching..." : "Ask Research Assistant"}
            </button>
          </div>
        </form>

        {error && <div className="error">{error}</div>}

        {loading && (
          <div className="loading">
            <div className="loading-title">
              Researching your question
            </div>

            <div className="loading-text">
              Routing the query, retrieving evidence, and generating a
              grounded answer...
            </div>
          </div>
        )}

        {result && (
          <section className="result">
            <div className="route">
              <span>Agent route</span>
              <strong>{result.route}</strong>
            </div>

            <div className="answer-section">
              <h2>Answer</h2>

              <div className="answer">
                {result.answer}
              </div>
            </div>

            {result.paper_evidence?.length > 0 && (
              <div className="sources">
                <h2>Paper Evidence</h2>

                {result.paper_evidence.map((source, index) => (
                  <div className="source" key={index}>
                    <div>
                      <strong>{source.paper_id}</strong>
                    </div>

                    <span>Page {source.page}</span>
                  </div>
                ))}
              </div>
            )}

            {result.data_evidence &&
              Object.keys(result.data_evidence).length > 0 && (
                <div className="data-result">
                  <h2>Dataset Evidence</h2>

                  <pre>
                    {JSON.stringify(
                      result.data_evidence,
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;