import { FormEvent, useState } from "react";
import SearchResults from "./components/SearchResults";
import type { SearchResult } from "./types";
import "./styles.css";

type SearchResponse = {
  query: string;
  results: SearchResult[];
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: trimmedQuery, limit: 8 })
      });

      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`);
      }

      const data = (await response.json()) as SearchResponse;
      setResults(data.results);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Search failed.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Multimodal retrieval</p>
        <h1>Search PDF images with plain language.</h1>
        <p className="lede">
          Text and images are embedded into one vector space with Gemini Embedding 2,
          then ranked in Chroma by semantic distance.
        </p>
      </section>

      <form className="search-form" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="query">
          Search query
        </label>
        <input
          id="query"
          name="query"
          placeholder="Find charts about renewable energy adoption"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error ? <p className="error-banner">{error}</p> : null}

      <SearchResults results={results} />
    </main>
  );
}
