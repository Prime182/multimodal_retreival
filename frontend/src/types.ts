export type SearchResultMetadata = {
  imageUrl?: string | null;
  file_path?: string | null;
  caption?: string | null;
  title?: string | null;
  source?: string | null;
  source_path?: string | null;
  page_number?: number | null;
  kind?: string | null;
  mimeType?: string | null;
};

export type SearchResult = {
  id: string;
  distance: number;
  score?: number | null;
  metadata: SearchResultMetadata;
};
