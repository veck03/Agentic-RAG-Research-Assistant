
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const paperTitles = {
  pacific_subtropical_highs_enso:
    "Annually Modulated Pacific Subtropical Highs Pace the ENSO Cycle",
  enso_changes_global_warming:
    "Detectability of Forced ENSO Changes under Global Warming",
  deep_learning_sst_downscaling:
    "Deep Learning-Based Statistical Downscaling of Sea Surface Temperature",
  neptune_ai_ocean_prediction:
    "NEPTUNE: An AI Model for Global Ocean Subseasonal Prediction",
  global_ocean_temperature_observations_2013:
    "A Review of Global Ocean Temperature Observations",
};

const suggestions = [
  "What mechanisms drive ENSO?",
  "How does ENSO change under global warming?",
  "What is the average surface temperature?",
];

function getPaperTitle(paperId) {
  return paperTitles[paperId] || paperId?.replaceAll("_", " ") || "Research paper";
}

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [requestCount, setRequestCount] = useState(0);

  function startNewQuery() {
    setQuery("");
    setResult(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const cleanQuery = query.trim();
    if (!cleanQuery || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: cleanQuery }),
      });

      if (!response.ok) {
        let message = "The server returned an error.";
        try {
          const body = await response.json();
          message = body.detail || message;
        } catch {
          // Keep the default error message.
        }
        throw new Error(message);
      }

      const data = await response.json();
      setResult(data);
      setRequestCount((count) => count + 1);
    } catch (err) {
      console.error(err);
      setError(
        err.message ||
          "Could not connect to the research assistant. Check that the backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  const paperEvidence = result?.paper_evidence || [];
  const dataEvidence = result?.data_evidence || {};

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M3 15c3-3 6-3 9 0s6 3 9 0" />
              <path d="M3 19c3-3 6-3 9 0s6 3 9 0" />
              <path d="M12 3v8m-3-3 3 3 3-3" />
            </svg>
          </div>
          <div>
            <div className="brand-name">Pelagic</div>
            <div className="brand-caption">RESEARCH ASSISTANT</div>
          </div>
        </div>

        <button className="new-query-button" onClick={startNewQuery}>
          <span className="plus-icon">+</span>
          New research query
        </button>

        <div className="sidebar-section">
          <div className="sidebar-label">WORKSPACE</div>
          <div className="sidebar-item active">
            <span className="sidebar-symbol">⌕</span>
            Research assistant
          </div>
          <div className="sidebar-item">
            <span className="sidebar-symbol">▤</span>
            Paper library
            <span className="coming-soon">Soon</span>
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="status-indicator">
            <span className="status-dot" />
            <span>Research engine</span>
            <span className="status-text">Ready</span>
          </div>
          <div className="sidebar-footer">
            Ocean science · Climate research
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-divider">/</span>
            <strong>Research assistant</strong>
          </div>
          <div className="topbar-right">
            <span className="topbar-status">
              <span className="status-dot" />
              API connected
            </span>
          </div>
        </header>

        <div className="content">
          {!result && !loading && !error && (
            <section className="welcome">
              <div className="welcome-icon">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="12" r="8.5" />
                  <path d="M3.8 12h16.4M12 3.5c2.3 2.3 3.3 5.1 3.3 8.5s-1 6.2-3.3 8.5c-2.3-2.3-3.3-5.1-3.3-8.5s1-6.2 3.3-8.5Z" />
                </svg>
              </div>
              <div className="welcome-eyebrow">OCEAN & CLIMATE INTELLIGENCE</div>
              <h1>Explore the science<br />beneath the surface.</h1>
              <p className="welcome-description">
                Ask questions about ocean and climate research. Get answers
                grounded in scientific papers and the available ocean dataset.
              </p>

              <div className="suggestion-heading">TRY A RESEARCH QUESTION</div>
              <div className="suggestions">
                {suggestions.map((item) => (
                  <button
                    className="suggestion-card"
                    key={item}
                    onClick={() => setQuery(item)}
                  >
                    <span>{item}</span>
                    <span className="suggestion-arrow">↗</span>
                  </button>
                ))}
              </div>
            </section>
          )}

          <section className={`query-area ${result || loading || error ? "query-area-active" : ""}`}>
            <form className="query-form" onSubmit={handleSubmit}>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    handleSubmit(event);
                  }
                }}
                placeholder="Ask a question about ocean science..."
                rows={3}
                aria-label="Research question"
              />
              <div className="query-form-footer">
                <span className="input-hint">
                  <kbd>Enter</kbd> to submit · <kbd>Shift + Enter</kbd> for a new line
                </span>
                <button
                  type="submit"
                  className="submit-button"
                  disabled={loading || !query.trim()}
                >
                  {loading ? "Researching" : "Ask"}
                  {!loading && <span className="send-icon">↑</span>}
                  {loading && <span className="button-spinner" />}
                </button>
              </div>
            </form>
            <div className="privacy-note">
              Answers are generated from retrieved evidence. Verify important findings against the cited sources.
            </div>
          </section>

          {error && (
            <div className="error-panel" role="alert">
              <div className="error-icon">!</div>
              <div>
                <strong>Something went wrong</strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          {loading && (
            <div className="loading-panel">
              <div className="loading-orbit">
                <span />
                <span />
                <span />
              </div>
              <div>
                <strong>Working on your research query</strong>
                <p>Routing your question, retrieving evidence, and generating an answer.</p>
              </div>
            </div>
          )}

          {result && (
            <section className="result-panel">
              <div className="result-heading">
                <div>
                  <div className="result-eyebrow">RESEARCH RESPONSE</div>
                  <h2>{result.query}</h2>
                </div>
                <button className="text-button" onClick={startNewQuery}>
                  New query
                </button>
              </div>

              <div className="response-meta">
                <span className="route-pill">
                  <span className="route-dot" />
                  {result.route || "UNKNOWN"} route
                </span>
                <span className="meta-divider" />
                <span>{paperEvidence.length} paper source{paperEvidence.length === 1 ? "" : "s"}</span>
                {Object.keys(dataEvidence).length > 0 && (
                  <>
                    <span className="meta-divider" />
                    <span>Dataset evidence</span>
                  </>
                )}
              </div>

              <article className="answer-card">
                <div className="answer-card-header">
                  <div className="answer-heading-icon">✦</div>
                  <h3>Answer</h3>
                </div>
                <div className="markdown-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {result.answer || "No answer was returned."}
                  </ReactMarkdown>
                </div>
              </article>

              {paperEvidence.length > 0 && (
                <section className="evidence-section">
                  <div className="section-heading">
                    <div>
                      <div className="section-eyebrow">RETRIEVED LITERATURE</div>
                      <h3>Paper evidence</h3>
                    </div>
                    <span className="count-pill">{paperEvidence.length} sources</span>
                  </div>

                  <div className="source-list">
                    {paperEvidence.map((source, index) => (
                      <article className="source-card" key={`${source.paper_id}-${source.page}-${index}`}>
                        <div className="source-number">
                          {String(index + 1).padStart(2, "0")}
                        </div>
                        <div className="source-info">
                          <h4>{getPaperTitle(source.paper_id)}</h4>
                          <div className="source-meta">
                            <span>{source.paper_id}</span>
                            {source.page != null && (
                              <>
                                <span className="source-separator">·</span>
                                <span>Page {source.page}</span>
                              </>
                            )}
                          </div>
                          {source.text && (
                            <p className="source-excerpt">{source.text}</p>
                          )}
                        </div>
                      </article>
                    ))}
                  </div>
                </section>
              )}

              {Object.keys(dataEvidence).length > 0 && (
                <section className="evidence-section">
                  <div className="section-heading">
                    <div>
                      <div className="section-eyebrow">STRUCTURED DATA</div>
                      <h3>Dataset evidence</h3>
                    </div>
                  </div>
                  <div className="dataset-card">
                    <div className="dataset-card-title">
                      <span className="dataset-icon">▦</span>
                      Ocean dataset result
                    </div>
                    <pre>{JSON.stringify(dataEvidence, null, 2)}</pre>
                  </div>
                </section>
              )}
            </section>
          )}
        </div>

        <footer className="main-footer">
          <span>Pelagic Research Assistant</span>
          <span>Powered by retrieval-augmented generation</span>
        </footer>
      </main>
    </div>
  );
}

export default App;