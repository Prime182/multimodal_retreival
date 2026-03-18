import React from "react";
import "./search-results.css";

export type SearchResultMetadata = {
  imageUrl?: string;
  file_path?: string;
  caption?: string;
  title?: string;
  source?: string;
  mimeType?: string;
  [key: string]: unknown;
};

export type SearchResultItem = {
  id: string;
  distance?: number | null;
  score?: number | null;
  metadata?: SearchResultMetadata | null;
};

export type SearchResultsProps = {
  results: SearchResultItem[];
  rankingMode?: "auto" | "distance" | "score";
  title?: string;
  emptyState?: React.ReactNode;
  onResultClick?: (result: SearchResultItem) => void;
};

const formatMetric = (value: number | null | undefined) => {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }
  return value.toFixed(3);
};

const getImageSource = (metadata?: SearchResultMetadata | null) => {
  if (!metadata) {
    return "";
  }
  return metadata.imageUrl ?? metadata.file_path ?? "";
};

const compareResults = (mode: "distance" | "score", a: SearchResultItem, b: SearchResultItem) => {
  const aValue = mode === "distance" ? a.distance : a.score;
  const bValue = mode === "distance" ? b.distance : b.score;

  const aMissing = typeof aValue !== "number" || Number.isNaN(aValue);
  const bMissing = typeof bValue !== "number" || Number.isNaN(bValue);

  if (aMissing && bMissing) {
    return a.id.localeCompare(b.id);
  }
  if (aMissing) {
    return 1;
  }
  if (bMissing) {
    return -1;
  }

  if (aValue === bValue) {
    return a.id.localeCompare(b.id);
  }

  return mode === "distance" ? aValue - bValue : bValue - aValue;
};

const resolveMode = (results: SearchResultItem[], rankingMode: SearchResultsProps["rankingMode"]) => {
  if (rankingMode !== "auto") {
    return rankingMode;
  }

  const hasDistance = results.some((result) => typeof result.distance === "number");
  if (hasDistance) {
    return "distance";
  }

  const hasScore = results.some((result) => typeof result.score === "number");
  return hasScore ? "score" : "distance";
};

export function SearchResults({
  results,
  rankingMode = "auto",
  title = "Search Results",
  emptyState = "No matching images were found.",
  onResultClick,
}: SearchResultsProps) {
  const mode = resolveMode(results, rankingMode);
  const sortedResults = [...results].sort((a, b) => compareResults(mode, a, b));

  return (
    <section className="search-results" aria-labelledby="search-results-title">
      <div className="search-results__header">
        <div>
          <p className="search-results__eyebrow">Multimodal retrieval</p>
          <h2 id="search-results-title" className="search-results__title">
            {title}
          </h2>
        </div>
        <div className="search-results__meta">
          <span>{sortedResults.length} result{sortedResults.length === 1 ? "" : "s"}</span>
          <span>sorted by {mode}</span>
        </div>
      </div>

      {sortedResults.length === 0 ? (
        <div className="search-results__empty">{emptyState}</div>
      ) : (
        <div className="search-results__grid">
          {sortedResults.map((result, index) => {
            const metadata = result.metadata ?? {};
            const src = getImageSource(metadata);
            const caption = metadata.caption ?? metadata.title ?? result.id;
            const metricValue = mode === "distance" ? result.distance : result.score;
            const metricLabel = mode === "distance" ? "distance" : "score";
            const sourceLabel =
              typeof metadata.source === "string"
                ? metadata.source
                : typeof metadata.source_path === "string"
                  ? metadata.source_path.split("/").at(-1)
                  : null;

            return (
              <article
                key={result.id}
                className="search-results__card"
                onClick={onResultClick ? () => onResultClick(result) : undefined}
                role={onResultClick ? "button" : undefined}
                tabIndex={onResultClick ? 0 : undefined}
                onKeyDown={
                  onResultClick
                    ? (event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          onResultClick(result);
                        }
                      }
                    : undefined
                }
                style={{ animationDelay: `${Math.min(index, 8) * 60}ms` }}
              >
                <div className="search-results__image-wrap">
                  {src ? (
                    <img className="search-results__image" src={src} alt={caption} loading="lazy" />
                  ) : (
                    <div className="search-results__placeholder">
                      <span>No image source</span>
                    </div>
                  )}
                </div>

                <div className="search-results__body">
                  <div className="search-results__row">
                    <h3 className="search-results__caption">{caption}</h3>
                    <span className="search-results__rank">#{index + 1}</span>
                  </div>

                  <div className="search-results__details">
                    <span className="search-results__pill">ID {result.id}</span>
                    {sourceLabel ? <span className="search-results__pill">{sourceLabel}</span> : null}
                    {typeof metadata.page_number === "number" ? (
                      <span className="search-results__pill">page {metadata.page_number}</span>
                    ) : null}
                    {formatMetric(metricValue) ? (
                      <span className="search-results__pill">
                        {metricLabel}: {formatMetric(metricValue)}
                      </span>
                    ) : null}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default SearchResults;
