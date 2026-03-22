from unstructured.partition.pdf import partition_pdf

def extract_pdf_elements(path, fname):
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )

def categorize_elements(raw_pdf_elements):
    tables = []
    texts = []
    images = []

    for element in raw_pdf_elements:
        etype = str(type(element))

        if "unstructured.documents.elements.Table" in etype:
            caption = getattr(element.metadata, "text_as_html", None) or ""
            # Try to find a caption from nearby text (stored in metadata)
            table_caption = getattr(element.metadata, "caption", None) or "No caption found"
            tables.append({
                "content": str(element),
                "caption": table_caption,
                "html": caption
            })

        elif "unstructured.documents.elements.Image" in etype:
            caption = getattr(element.metadata, "caption", None) or str(element).strip() or "No caption found"
            image_path = getattr(element.metadata, "image_path", None) or "Path not available"
            images.append({
                "caption": caption,
                "image_path": image_path
            })

        elif "unstructured.documents.elements.CompositeElement" in etype:
            texts.append(str(element))

    return texts, tables, images


def print_results(texts, tables, images):
    # ── Text Chunks ──────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  TEXT CHUNKS  ({len(texts)} found)")
    print("=" * 70)
    for i, text in enumerate(texts, 1):
        print(f"\n[Chunk {i}]")
        print("-" * 50)
        print(text)

    # ── Images ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  IMAGES  ({len(images)} found)")
    print("=" * 70)
    for i, img in enumerate(images, 1):
        print(f"\n[Image {i}]")
        print("-" * 50)
        print(f"  Caption    : {img['caption']}")
        print(f"  Image Path : {img['image_path']}")

    # ── Tables ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  TABLES  ({len(tables)} found)")
    print("=" * 70)
    for i, tbl in enumerate(tables, 1):
        print(f"\n[Table {i}]")
        print("-" * 50)
        print(f"  Caption : {tbl['caption']}")
        print(f"  Content :\n{tbl['content']}")
        if tbl["html"]:
            print(f"  HTML    :\n{tbl['html']}")


# ── Main ──────────────────────────────────────────────────────────────────────
folder_path = "/content/"
file_name   = "BJ_100833.pdf"

try:
    raw_pdf_elements          = extract_pdf_elements(folder_path, file_name)
    texts, tables, images     = categorize_elements(raw_pdf_elements)
    print_results(texts, tables, images)

except Exception as e:
    print(f"An error occurred: {e}")