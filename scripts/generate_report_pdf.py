"""
Generate Professional PDF Report using WeasyPrint
Creates beautifully formatted PDF from Markdown report
"""

import os
import sys
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

# Professional CSS styling for the report
CSS_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
    size: A4;
    margin: 2cm 2.5cm;
    @top-center {
        content: "Fraud Detection System - Final Report";
        font-size: 9pt;
        color: #666;
        font-family: 'Inter', sans-serif;
    }
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #666;
        font-family: 'Inter', sans-serif;
    }
    @bottom-right {
        content: "Adey Innovations Inc.";
        font-size: 8pt;
        color: #888;
        font-family: 'Inter', sans-serif;
    }
}

@page :first {
    @top-center { content: none; }
    @bottom-center { content: none; }
    @bottom-right { content: none; }
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a2e;
    max-width: 100%;
}

h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #0f3460;
    margin-top: 0;
    margin-bottom: 0.5em;
    padding-bottom: 0.3em;
    border-bottom: 3px solid #e94560;
    page-break-after: avoid;
}

h2 {
    font-size: 18pt;
    font-weight: 600;
    color: #16213e;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    padding-bottom: 0.2em;
    border-bottom: 2px solid #0f3460;
    page-break-after: avoid;
}

h3 {
    font-size: 14pt;
    font-weight: 600;
    color: #1a1a2e;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}

h4 {
    font-size: 12pt;
    font-weight: 600;
    color: #333;
    margin-top: 1em;
    margin-bottom: 0.3em;
}

p {
    margin: 0.8em 0;
    text-align: justify;
}

strong {
    font-weight: 600;
    color: #0f3460;
}

em {
    font-style: italic;
    color: #555;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

thead {
    background: linear-gradient(135deg, #0f3460, #16213e);
    color: white;
}

th {
    padding: 12px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 10pt;
}

td {
    padding: 10px;
    border-bottom: 1px solid #e0e0e0;
}

tbody tr:nth-child(even) {
    background-color: #f8f9fa;
}

tbody tr:hover {
    background-color: #e8f4f8;
}

/* Code blocks */
pre {
    background: #1a1a2e;
    color: #e8e8e8;
    padding: 1em;
    border-radius: 8px;
    overflow-x: auto;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 9pt;
    line-height: 1.5;
    margin: 1em 0;
    page-break-inside: avoid;
}

code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 9pt;
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 4px;
    color: #e94560;
}

pre code {
    background: none;
    padding: 0;
    color: inherit;
}

/* Lists */
ul, ol {
    margin: 0.8em 0;
    padding-left: 1.5em;
}

li {
    margin: 0.3em 0;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #e94560;
    margin: 1em 0;
    padding: 0.5em 1em;
    background: #f8f9fa;
    font-style: italic;
}

/* Images */
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 2em 0;
}

/* Links */
a {
    color: #0f3460;
    text-decoration: none;
}

a:hover {
    color: #e94560;
    text-decoration: underline;
}

/* Special classes */
.highlight {
    background: linear-gradient(120deg, #ffeaa7 0%, #ffeaa7 100%);
    padding: 2px 4px;
    border-radius: 3px;
}

.success {
    color: #27ae60;
    font-weight: 600;
}

.warning {
    color: #f39c12;
    font-weight: 600;
}

.danger {
    color: #e74c3c;
    font-weight: 600;
}

/* Cover page styling */
.cover-page {
    text-align: center;
    padding-top: 5cm;
    page-break-after: always;
}

.cover-page h1 {
    font-size: 36pt;
    border: none;
    color: #0f3460;
    margin-bottom: 0.5em;
}

.cover-page .subtitle {
    font-size: 16pt;
    color: #666;
    margin-bottom: 3cm;
}

.cover-page .author {
    font-size: 14pt;
    color: #333;
    margin-top: 2cm;
}

.cover-page .date {
    font-size: 12pt;
    color: #666;
    margin-top: 0.5cm;
}

/* Table of contents */
.toc {
    page-break-after: always;
}

.toc h2 {
    text-align: center;
    border: none;
}

.toc ul {
    list-style: none;
    padding: 0;
}

.toc li {
    margin: 0.5em 0;
    padding-left: 1em;
}

/* Metrics highlight boxes */
.metric-box {
    display: inline-block;
    background: linear-gradient(135deg, #0f3460, #16213e);
    color: white;
    padding: 1em 1.5em;
    border-radius: 8px;
    margin: 0.5em;
    text-align: center;
}

.metric-box .value {
    font-size: 24pt;
    font-weight: 700;
}

.metric-box .label {
    font-size: 10pt;
    opacity: 0.9;
}

/* Print-specific */
@media print {
    body {
        print-color-adjust: exact;
        -webkit-print-color-adjust: exact;
    }
}
"""


def convert_md_to_pdf(md_file: str, pdf_file: str, title: str = "Fraud Detection Report"):
    """Convert Markdown file to styled PDF."""
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'tables',
        'fenced_code',
        'codehilite',
        'toc',
        'attr_list',
        'md_in_html'
    ])
    html_content = md.convert(md_content)
    
    # Create full HTML document
    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Configure fonts
    font_config = FontConfiguration()
    
    # Generate PDF
    html = HTML(string=html_doc, base_url=os.path.dirname(os.path.abspath(md_file)))
    css = CSS(string=CSS_STYLE, font_config=font_config)
    
    html.write_pdf(pdf_file, stylesheets=[css], font_config=font_config)
    
    print(f"✓ PDF generated: {pdf_file}")


def main():
    """Generate PDF reports."""
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Generate final report PDF
    print("Generating Final Report PDF...")
    convert_md_to_pdf(
        'reports/final-report.md',
        'reports/final-report.pdf',
        'Fraud Detection System - Final Report'
    )
    
    # Also regenerate interim-2 report if it exists
    if os.path.exists('reports/interim-2-report.md'):
        print("Generating Interim-2 Report PDF...")
        convert_md_to_pdf(
            'reports/interim-2-report.md',
            'reports/interim-2-report.pdf',
            'Fraud Detection - Interim 2 Report'
        )
    
    print("\n✅ All PDF reports generated successfully!")


if __name__ == "__main__":
    main()

