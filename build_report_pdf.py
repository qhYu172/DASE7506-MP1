"""Render REPORT.md to REPORT.pdf with ReportLab (LaTeX-free fallback)."""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, LongTable, PageBreak, PageTemplate,
    Paragraph, Spacer, TableStyle,
)


ROOT = Path(__file__).resolve().parent
FONT_PAIRS = (
    (Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')),
    (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
     Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')),
    (Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'),
     Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf')),
)
FONT_PAIR = next(((regular, bold) for regular, bold in FONT_PAIRS
                  if regular.exists() and bold.exists()), None)
if FONT_PAIR is None:
    raise RuntimeError('Install Arial, DejaVu Sans, or Liberation Sans to build the PDF.')
pdfmetrics.registerFont(TTFont('ArialReport', str(FONT_PAIR[0])))
pdfmetrics.registerFont(TTFont('ArialReportBold', str(FONT_PAIR[1])))
pdfmetrics.registerFontFamily('ArialReport', normal='ArialReport', bold='ArialReportBold')

NAVY = colors.HexColor('#15324f')
BLUE = colors.HexColor('#246596')
PALE = colors.HexColor('#eaf1f7')
GRAY = colors.HexColor('#4e5e6b')

STYLES = {
    'title': ParagraphStyle('title', fontName='ArialReportBold', fontSize=16.5,
                            leading=20.5, textColor=NAVY, spaceAfter=6),
    'meta': ParagraphStyle('meta', fontName='ArialReport', fontSize=9,
                           leading=12, textColor=GRAY, spaceAfter=2),
    'h2': ParagraphStyle('h2', fontName='ArialReportBold', fontSize=11,
                         leading=14, textColor=NAVY, spaceBefore=11, spaceAfter=5,
                         keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='ArialReportBold', fontSize=9.5,
                         leading=12.5, textColor=BLUE, spaceBefore=8, spaceAfter=4,
                         keepWithNext=True),
    'body': ParagraphStyle('body', fontName='ArialReport', fontSize=9,
                           leading=12.6, textColor=colors.HexColor('#202b33'),
                           spaceAfter=6.5, splitLongWords=True),
    'reference': ParagraphStyle('reference', fontName='ArialReport', fontSize=8.3,
                                leading=11.3, textColor=colors.HexColor('#202b33'),
                                leftIndent=14, firstLineIndent=-14, spaceAfter=3),
    'equation': ParagraphStyle('equation', fontName='ArialReport', fontSize=8.5,
                               leading=12.5, alignment=TA_CENTER, textColor=NAVY,
                               spaceBefore=3, spaceAfter=9, splitLongWords=True),
    'table_header': ParagraphStyle('table_header', fontName='ArialReportBold',
                                   fontSize=7.3, leading=9.5, textColor=colors.white,
                                   alignment=TA_LEFT, splitLongWords=True),
    'table_cell': ParagraphStyle('table_cell', fontName='ArialReport',
                                 fontSize=7.4, leading=9.7, textColor=colors.HexColor('#202b33'),
                                 splitLongWords=True),
}


def inline(text):
    result = escape(text)
    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                    lambda m: f'<link href="{m.group(2)}" color="#246596">{m.group(1)}</link>',
                    result)
    result = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', result)
    result = re.sub(r'`([^`]+)`', r'<font color="#15324f">\1</font>', result)
    return result


def page_frame(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor('#cad5df'))
    canvas.setLineWidth(.5)
    canvas.line(46, height - 39, width - 46, height - 39)
    canvas.setFont('ArialReport', 7.7)
    canvas.setFillColor(GRAY)
    canvas.drawString(46, height - 31, 'DASE7506 MP1  |  Final technical report')
    canvas.line(46, 38, width - 46, 38)
    canvas.drawString(46, 26, 'Frozen checkpoint: Iteration 13  |  Test BPB: 1.609155')
    canvas.drawRightString(width - 46, 26, f'{doc.page}')
    canvas.restoreState()


def table_from_markdown(lines, available_width):
    rows = [[cell.strip() for cell in line.strip().strip('|').split('|')] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r':?-{3,}:?', cell) for cell in row)]
    cols = len(rows[0])
    if cols == 5:
        widths = [available_width * n for n in (.38, .09, .12, .20, .21)]
    elif cols == 4:
        widths = [available_width * n for n in (.37, .21, .22, .20)]
    else:
        widths = [available_width / cols] * cols
    table_rows = []
    for ri, row in enumerate(rows):
        style = STYLES['table_header'] if ri == 0 else STYLES['table_cell']
        table_rows.append([Paragraph(inline(cell), style) for cell in row])
    table = LongTable(table_rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, 0), .7, NAVY),
    ]))
    return table


def build():
    output = ROOT / 'REPORT.pdf'
    width, height = A4
    margin = 46
    frame = Frame(margin, 49, width - 2 * margin, height - 100,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(str(output), pagesize=A4,
                          leftMargin=margin, rightMargin=margin,
                          topMargin=50, bottomMargin=49,
                          title='A Scaled Causal Decoder with Exact Within-Window Memory',
                          author='DASE7506 Mini Project 1')
    doc.addPageTemplates(PageTemplate(id='report', frames=frame, onPage=page_frame))

    lines = (ROOT / 'REPORT.md').read_text(encoding='utf-8').splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith('|'):
            block = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                block.append(lines[index])
                index += 1
            story.extend([Spacer(1, 3), table_from_markdown(block, width - 2 * margin), Spacer(1, 7)])
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), STYLES['title']))
            index += 1
            continue
        if line.startswith('## '):
            if line.startswith('## 5. Frozen result'):
                story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]), STYLES['h2']))
            index += 1
            continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), STYLES['h3']))
            index += 1
            continue
        if line.startswith('**DASE7506') or line.startswith('**Protocol:'):
            story.append(Paragraph(inline(line.rstrip('  ')), STYLES['meta']))
            index += 1
            continue
        if re.match(r'^\d+\. ', line):
            story.append(Paragraph(inline(line), STYLES['reference']))
            index += 1
            continue
        paragraph = []
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(('#', '|')):
            paragraph.append(lines[index].strip())
            index += 1
        body = ' '.join(paragraph)
        style = STYLES['equation'] if body.startswith('`') and body.endswith('`') else STYLES['body']
        story.append(Paragraph(inline(body), style))

    doc.build(story)
    print(output)


if __name__ == '__main__':
    build()
