from docx import Document
from docx.oxml.ns import qn

path = r"C:\Users\jatto\Research_project\Lokoja_Flood_Thesis_Draft.docx"
doc = Document(path)


def highlighted_text(paragraph):
    hits = []
    for run in paragraph.runs:
        for highlight in run._r.xpath('.//w:highlight'):
            if highlight.get(qn('w:val')):
                hits.append((run.text, highlight.get(qn('w:val'))))
    return hits


for pi, para in enumerate(doc.paragraphs):
    hits = highlighted_text(para)
    if hits:
        print(f"PARAGRAPH {pi}: {para.text}")
        for text, colour in hits:
            print("  HIGHLIGHT:", repr(text), colour)

for ti, table in enumerate(doc.tables):
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            for pi, para in enumerate(cell.paragraphs):
                hits = highlighted_text(para)
                if hits:
                    print(f"TABLE {ti} ROW {ri} CELL {ci} P {pi}: {para.text}")
                    for text, colour in hits:
                        print("  HIGHLIGHT:", repr(text), colour)
