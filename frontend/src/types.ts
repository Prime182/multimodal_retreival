export type ContentType = "text" | "equation" | "table" | "image";

export type SearchResultMetadata = {
  kind?: ContentType | string | null;
  content_type?: ContentType | string | null;
  journal_id?: string | null;
  article_id?: string | null;
  section?: string | null;
  text?: string | null;
  document?: string | null;
  latex?: string | null;
  context?: string | null;
  csv_data?: string | null;
  header?: string | null;
  row_index?: number | null;
  imageUrl?: string | null;
  file_path?: string | null;
  caption?: string | null;
  title?: string | null;
  source?: string | null;
  source_path?: string | null;
  page_number?: number | null;
  page_start?: number | null;
  page_end?: number | null;
  mimeType?: string | null;
  token_count?: number | null;
};

export type SearchResult = {
  id: string;
  distance: number;
  score?: number | null;
  metadata: SearchResultMetadata;
};
