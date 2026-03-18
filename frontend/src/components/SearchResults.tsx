import type { ReactNode } from "react";
import type { ContentType, SearchResult } from "../types";
import "./search-results.css";

type SearchResultsProps = {
  results: SearchResult[];
  rankingMode?: "auto" | "distance" | "score";
  title?: string;
  emptyState?: ReactNode;
  onResultClick?: (result: SearchResult) => void;
};

const CONTENT_TYPES: ContentType[] = ["text", "equation", "table", "image"];

const KIND_LABELS: Record<ContentType, string> = {
  text: "Text",
  equation: "Equation",
  table: "Table",
  image: "Image"
};

const KIND_DESCRIPTIONS: Record<ContentType, string> = {
  text: "Paragraph and section matches",
  equation: "Formula and derivation matches",
  table: "Structured table matches",
  image: "Rendered figure and image matches"
};

function isContentType(value: unknown): value is ContentType {
  return value === "text" || value === "equation" || value === "table" || value === "image";
}

function resolveKind(metadata: SearchResult["metadata"]): ContentType {
  if (isContentType(metadata.kind)) {
    return metadata.kind;
  }

  if (metadata.imageUrl || metadata.file_path) {
    return "image";
  }

  if (metadata.latex || metadata.context) {
    return "equation";
  }

  if (metadata.csv_data || metadata.header) {
    return "table";
  }

  return "text";
}

function formatMetric(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }

  return value.toFixed(3);
}

function getImageSource(metadata: SearchResult["metadata"]) {
  const src = metadata.imageUrl ?? metadata.file_path ?? "";
  if (src.startsWith("/") && !src.startsWith("//")) {
    const env = (import.meta as any).env;
    const apiBaseUrl = env?.VITE_API_BASE_URL ?? "http://localhost:8000";
    return `${apiBaseUrl}${src}`;
  }

  return src;
}

function compareResults(
  mode: "distance" | "score",
  a: SearchResult,
  b: SearchResult
) {
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
}

function resolveMode(
  results: SearchResult[],
  rankingMode: SearchResultsProps["rankingMode"]
) {
  if (rankingMode !== "auto") {
    return rankingMode;
  }

  const hasDistance = results.some(
    (result) => typeof result.distance === "number" && !Number.isNaN(result.distance)
  );
  if (hasDistance) {
    return "distance";
  }

  const hasScore = results.some(
    (result) => typeof result.score === "number" && !Number.isNaN(result.score)
  );
  return hasScore ? "score" : "distance";
}

function truncate(text: string, maxLength: number) {
  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

function parseCsvRows(csvData: string) {
  const normalized = csvData.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentCell = "";
  let inQuotes = false;

  for (let index = 0; index < normalized.length; index += 1) {
    const char = normalized[index];

    if (char === '"') {
      if (inQuotes && normalized[index + 1] === '"') {
        currentCell += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (!inQuotes && char === ",") {
      currentRow.push(currentCell);
      currentCell = "";
      continue;
    }

    if (!inQuotes && char === "\n") {
      currentRow.push(currentCell);
      if (currentRow.some((value) => value.trim().length > 0)) {
        rows.push(currentRow);
      }
      currentRow = [];
      currentCell = "";
      continue;
    }

    currentCell += char;
  }

  currentRow.push(currentCell);
  if (currentRow.some((value) => value.trim().length > 0)) {
    rows.push(currentRow);
  }

  return rows;
}

function parseCsvLine(line: string | null | undefined) {
  if (!line) {
    return [];
  }

  return parseCsvRows(line)[0] ?? [];
}

function normalizeTableRows(rows: string[][], columnCount: number) {
  return rows.map((row) => {
    const nextRow = [...row];
    while (nextRow.length < columnCount) {
      nextRow.push("");
    }
    return nextRow.slice(0, columnCount);
  });
}

function buildSourceLabel(metadata: SearchResult["metadata"]) {
  if (typeof metadata.source === "string" && metadata.source.trim()) {
    return metadata.source;
  }

  if (typeof metadata.source_path === "string" && metadata.source_path.trim()) {
    return metadata.source_path.split("/").pop() ?? metadata.source_path;
  }

  return null;
}

function getTextPreview(metadata: SearchResult["metadata"]) {
  const rawText =
    metadata.text ?? metadata.document ?? metadata.caption ?? metadata.title ?? "";
  return truncate(rawText.trim() || "No text preview available.", 320);
}

function renderImageCard(
  result: SearchResult,
  metadata: SearchResult["metadata"],
  caption: string
) {
  const src = getImageSource(metadata);
  return (
    <div className="search-results__media search-results__media--image">
      {src ? (
        <img className="search-results__image" src={src} alt={caption} loading="lazy" />
      ) : (
        <div className="search-results__placeholder">
          <span>Image unavailable</span>
        </div>
      )}
    </div>
  );
}

function renderTextCard(metadata: SearchResult["metadata"]) {
  return (
    <div className="search-results__media search-results__media--text">
      <p className="search-results__preview">{getTextPreview(metadata)}</p>
      <span className="search-results__media-label">Text excerpt</span>
    </div>
  );
}

function renderEquationCard(metadata: SearchResult["metadata"]) {
  const latex = metadata.latex ?? metadata.document ?? metadata.caption ?? "";
  const context = metadata.context ?? metadata.text ?? "";

  return (
    <div className="search-results__media search-results__media--equation">
      <pre className="sr-eq-block">{latex.trim() || "No equation text available."}</pre>
      <p className="search-results__preview search-results__preview--context">
        {context.trim() || "No surrounding context available."}
      </p>
      <span className="search-results__media-label">Equation excerpt</span>
    </div>
  );
}

function renderTableCard(metadata: SearchResult["metadata"]) {
  const csvRows = metadata.csv_data ? parseCsvRows(metadata.csv_data) : [];
  const headerCells = parseCsvLine(metadata.header);
  const hasExplicitHeader = headerCells.length > 0;
  const inferredHeader = hasExplicitHeader ? headerCells : csvRows[0] ?? [];
  const dataRows = hasExplicitHeader ? csvRows : csvRows.slice(1);
  const columnCount = Math.max(
    inferredHeader.length,
    ...dataRows.map((row) => row.length),
    0
  );
  const normalizedHeader = normalizeTableRows([inferredHeader], columnCount)[0] ?? [];
  const normalizedRows = normalizeTableRows(dataRows, columnCount);

  return (
    <div className="search-results__media search-results__media--table">
      <div className="search-results__table-scroll">
        {columnCount > 0 ? (
          <table className="sr-table">
            <thead>
              <tr>
                {normalizedHeader.map((cell, index) => (
                  <th key={`header-${index}`}>{cell || `Column ${index + 1}`}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {normalizedRows.length > 0 ? (
                normalizedRows.map((row, rowIndex) => (
                  <tr key={`row-${rowIndex}`}>
                    {row.map((cell, cellIndex) => (
                      <td key={`cell-${rowIndex}-${cellIndex}`}>{cell}</td>
                    ))}
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={Math.max(columnCount, 1)}>No table rows available.</td>
                </tr>
              )}
            </tbody>
          </table>
        ) : (
          <div className="search-results__placeholder">
            <span>No table data available</span>
          </div>
        )}
      </div>
      <span className="search-results__media-label">
        {typeof metadata.row_index === "number" ? `Row ${metadata.row_index}` : "Table excerpt"}
      </span>
    </div>
  );
}

function renderCardMedia(result: SearchResult, kind: ContentType) {
  const metadata = result.metadata;
  const caption = metadata.caption ?? metadata.title ?? result.id;

  if (kind === "image") {
    return renderImageCard(result, metadata, caption);
  }

  if (kind === "equation") {
    return renderEquationCard(metadata);
  }

  if (kind === "table") {
    return renderTableCard(metadata);
  }

  return renderTextCard(metadata);
}

function renderKindPills(kindCounts: Record<ContentType, number>) {
  return CONTENT_TYPES.map((kind) => (
    <span key={kind} className={`search-results__kind-count search-results__kind-count--${kind}`}>
      <span className="search-results__kind-count-label">{KIND_LABELS[kind]}</span>
      <span className="search-results__kind-count-value">{kindCounts[kind]}</span>
    </span>
  ));
}

export function SearchResults({
  results,
  rankingMode = "auto",
  title = "Search Results",
  emptyState = "No matching results were found.",
  onResultClick
}: SearchResultsProps) {
  const mode = resolveMode(results, rankingMode) || "distance";
  const sortedResults = [...results].sort((a, b) => compareResults(mode, a, b));
  const kindCounts = sortedResults.reduce<Record<ContentType, number>>(
    (counts, result) => {
      const kind = resolveKind(result.metadata);
      counts[kind] += 1;
      return counts;
    },
    { text: 0, equation: 0, table: 0, image: 0 }
  );

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
          <div className="search-results__meta-line">
            <span>{sortedResults.length} result{sortedResults.length === 1 ? "" : "s"}</span>
            <span>sorted by {mode}</span>
          </div>
          <div className="search-results__kind-summary" aria-label="Result counts by type">
            {renderKindPills(kindCounts)}
          </div>
        </div>
      </div>

      {sortedResults.length === 0 ? (
        <div className="search-results__empty">{emptyState}</div>
      ) : (
        <div className="search-results__grid">
          {sortedResults.map((result, index) => {
            const metadata = result.metadata;
            const kind = resolveKind(metadata);
            const metricValue = mode === "distance" ? result.distance : result.score;
            const metricLabel = mode === "distance" ? "distance" : "score";
            const sourceLabel = buildSourceLabel(metadata);
            const caption = metadata.caption ?? metadata.title ?? result.id;

            return (
              <article
                key={result.id}
                className={`search-results__card search-results__card--${kind}`}
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
                {renderCardMedia(result, kind)}

                <div className="search-results__body">
                  <div className="search-results__row">
                    <h3 className="search-results__caption">{caption}</h3>
                    <span className="search-results__rank">#{index + 1}</span>
                  </div>

                  <div className="search-results__details">
                    <span className="search-results__pill">ID {result.id}</span>
                    <span className={`search-results__pill search-results__pill--kind search-results__pill--${kind}`}>
                      {KIND_LABELS[kind]}
                    </span>
                    {sourceLabel ? <span className="search-results__pill">{sourceLabel}</span> : null}
                    {typeof metadata.section === "string" && metadata.section.trim() ? (
                      <span className="search-results__pill">{metadata.section}</span>
                    ) : null}
                    {typeof metadata.page_number === "number" ? (
                      <span className="search-results__pill">page {metadata.page_number}</span>
                    ) : null}
                    {typeof metadata.page_start === "number" && typeof metadata.page_end === "number" ? (
                      <span className="search-results__pill">
                        pages {metadata.page_start}-{metadata.page_end}
                      </span>
                    ) : null}
                    {formatMetric(metricValue) ? (
                      <span className="search-results__pill">
                        {metricLabel}: {formatMetric(metricValue)}
                      </span>
                    ) : null}
                  </div>

                  <footer className="search-results__footer">
                    <span className={`search-results__kind-badge search-results__kind-badge--${kind}`}>
                      {KIND_LABELS[kind]}
                    </span>
                    <span className="search-results__kind-note">{KIND_DESCRIPTIONS[kind]}</span>
                  </footer>
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
