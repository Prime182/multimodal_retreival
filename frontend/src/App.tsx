import { FormEvent, useState } from "react";
import SearchResults from "./components/SearchResults";
import type { ContentType, SearchResult } from "./types";
import "./styles.css";

type SearchResponse = {
  query: string;
  results: SearchResult[];
};

const API_BASE_URL =
  (import.meta as any).env?.VITE_API_BASE_URL ?? "http://localhost:8000";
const CONTENT_TYPES: ContentType[] = ["text", "equation", "table", "image"];
const CONTENT_TYPE_LABELS: Record<ContentType, string> = {
  text: "Text",
  equation: "Equation",
  table: "Table",
  image: "Image"
};

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeContentTypes, setActiveContentTypes] = useState<ContentType[]>(
    CONTENT_TYPES
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const allContentTypesSelected =
    activeContentTypes.length === CONTENT_TYPES.length &&
    CONTENT_TYPES.every((contentType) => activeContentTypes.includes(contentType));

  function toggleContentType(contentType: ContentType) {
    setActiveContentTypes((current) => {
      if (current.includes(contentType)) {
        if (current.length === 1) {
          return current;
        }
        return current.filter((value) => value !== contentType);
      }

      return [...current, contentType];
    });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const contentTypes = allContentTypesSelected ? null : activeContentTypes;
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: trimmedQuery,
          limit: 8,
          content_types: contentTypes
        })
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
        <h1>Search PDFs across text, equations, tables, and images.</h1>
        <p className="lede">
          Text, equations, tables, and images are embedded into one vector space with
          Gemini Embedding 2, then ranked in Chroma by semantic distance.
        </p>
      </section>

      <form className="search-form" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="query">
          Search query
        </label>
        <div className="search-form__row">
          <input
            id="query"
            name="query"
            placeholder="Find equations, tables, or figures about renewable energy adoption"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </div>

        <div className="type-filter" role="group" aria-label="Filter result types">
          {CONTENT_TYPES.map((contentType) => {
            const isActive = activeContentTypes.includes(contentType);
            return (
              <button
                key={contentType}
                type="button"
                className={`type-chip type-chip--${contentType} ${
                  isActive ? "is-active" : ""
                }`}
                aria-pressed={isActive}
                onClick={() => toggleContentType(contentType)}
              >
                {CONTENT_TYPE_LABELS[contentType]}
              </button>
            );
          })}
        </div>
      </form>

      {error ? <p className="error-banner">{error}</p> : null}

      <SearchResults results={results} />
    </main>
  );
}
