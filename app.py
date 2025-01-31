import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportLabImage,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from markdown2 import markdown
from bs4 import BeautifulSoup
import datetime

# ----------------------------------------------------------------------------------
# DISC Type Mappings
# ----------------------------------------------------------------------------------
#
# We'll map text forms in the CSV (e.g. "Drive/Support") to slash-lowercase codes (e.g. "D/s").
# Single words: "Drive" => "D", "Influence" => "I", "Support" => "S", "Clarity" => "C"
# Hybrids: "Drive/Influence" => "D/i", "Influence/Support" => "I/s", etc.
csv_to_disc_map = {
    "Drive": "D",
    "Influence": "I",
    "Support": "S",
    "Clarity": "C"
}

def parse_disc_type(text: str) -> str:
    """
    Convert a CSV type like "Drive", "Influence", "Drive/Support" into slash-lowercase form (e.g. "D/s").
    Return an empty string if it can't be parsed.
    """
    if not text:
        return ""
    parts = text.split('/')
    codes = []
    for p in parts:
        p = p.strip()
        if p in csv_to_disc_map:
            codes.append(csv_to_disc_map[p])
        else:
            return ""
    if len(codes) == 1:
        return codes[0]
    elif len(codes) == 2:
        return f"{codes[0]}/{codes[1].lower()}"
    else:
        return ""

# ----------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------
initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments using the DISC framework.

DISC stands for Drive (D), Influence (I), Support (S), Clarity (C).
We also have hybrid types that combine two primary styles (e.g., D/i, I/d, S/c),
where the second letter is lowercase and separated by a slash.

**Team Size:** {TEAM_SIZE}

**Team Members and their DISC Results:**

{TEAM_MEMBERS_LIST}

Your goal:
Generate a comprehensive team personality report based on DISC, adhering to these requirements:
1. Refer to DISC as Drive, Influence, Support, Clarity (not Dominance, Influence, Steadiness, Conscientiousness).
2. Use "type" or "style" rather than "dimension."
3. For hybrid types, always use slash-lowercase for the second letter (e.g., D/i, C/s).
4. No mention of any other frameworks (e.g., MBTI, Enneagram).
5. Round all percentages to the nearest whole number.
6. Maintain a professional, neutral tone.
7. Use **Markdown headings**: `##` for main sections (1,2,3,4) and `###` (or smaller) for subheadings.
8. Provide blank lines between paragraphs and bullets for clarity.
"""

prompts = {
    "Team Profile": """
{INITIAL_CONTEXT}

**Your Role:**

Write the **Team Profile** section of the report (Section 1).

## Section 1: Team Profile

- Briefly introduce the DISC framework as Drive (D), Influence (I), Support (S), Clarity (C).
- Mention hybrid types with slash-lowercase (e.g., D/i, I/s, S/c).
- Outline the core characteristics of each DISC type/style actually present on the team (primary or hybrid).
- Describe how this combination of types affects foundational team dynamics.
- Use appropriate subheadings and blank lines.
- Required length: ~500 words.

**Begin your section below:**
""",
    # We add a placeholder {DISTRIBUTION_TABLE} so the LLM sees exact counts.
    "Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Actual Type Counts & Percentages**:
{DISTRIBUTION_TABLE}

**Your Role:**

Write the **Type Distribution** section of the report (Section 2).

## Section 2: Type Distribution

Include the following items in your text (in Markdown):
1. A **textual table** of each DISC type present in the overall set (primary + hybrid), with count and percentage (round to nearest whole) for each. DO NOT LEAVE THIS OUT!
2. `### Types on the Team` - List types actually present, from most common to least, with 1–2 bullet points about each type, and show count & percentage on separate lines.
3. `### Types Not on the Team` - Same format, but count=0, 0% (still give a short description).
4. `### Bar Chart` - Simply mention that we have included a bar chart below (the code handles displaying it).
5. `### Dominant Types` - Explain how the most common styles influence communication/decision-making.
6. `### Less Represented Types` - How the scarcity of certain styles impacts the team.
7. `### Summary` - Wrap up key points in 2–3 sentences.

Maintain heading hierarchy (`##` for this section, `###` for subtopics, bullet points as needed).

Required length: ~500 words.

**Continue the report below:**
""",
    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Team Insights** section of the report (Section 3).

## Section 3: Team Insights

Include these subheadings (`###`):
1. **Strengths**
   - At least four strengths from the dominant DISC styles.
   - Each strength in **bold** on a single line, followed by a blank line, then a paragraph explanation.

2. **Potential Blind Spots**
   - At least four challenges or areas of improvement.
   - Same formatting approach (bold line, blank line, paragraph).

3. **Communication**
   - 1–2 paragraphs.

4. **Teamwork**
   - 1–2 paragraphs.

5. **Conflict**
   - 1–2 paragraphs.

Required length: ~700 words. Add blank lines for readability.

**Continue the report below:**
""",
    "Actions and Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Actions and Next Steps** section of the report (Section 4).

## Section 4: Actions and Next Steps

- Provide actionable recommendations for team leaders, referencing the DISC composition.
- Use subheadings (`###`) for each major recommendation area.
- Offer a brief justification for each recommendation, linking it to the specific DISC types/styles involved.
- Present the recommendations as bullet points or numbered lists of specific actions, with blank lines between each item.
- End your output immediately after the last bullet (no concluding paragraph).
- Required length: ~400 words.

**Conclude the report below:**
"""
}

# ----------------------------------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------------------------------

st.title('DISC Team Report Generator')

st.subheader("Cover Page Details")
logo_path = "truity_logo.png"
company_name = st.text_input("Company Name (for cover page)", "")
team_name = st.text_input("Team Name (for cover page)", "")
today_str = datetime.date.today().strftime("%B %d, %Y")
custom_date = st.text_input("Date (for cover page)", today_str)

st.subheader("Upload CSV")
uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

if st.button("Generate Report from CSV"):
    if not uploaded_csv:
        st.error("Please upload a valid CSV file first.")
    else:
        with st.spinner("Processing CSV..."):
            df = pd.read_csv(uploaded_csv)
            valid_rows = []
            for i, row in df.iterrows():
                nm = str(row.get("User Name", "")).strip()
                raw_dt = str(row.get("DISC Type", "")).strip()
                dt_parsed = parse_disc_type(raw_dt)
                if nm and dt_parsed:
                    valid_rows.append((nm, dt_parsed))

            if not valid_rows:
                st.error("No valid DISC types found in CSV.")
            else:
                team_size = len(valid_rows)
                # Prepare a user list
                disc_lines = []
                for idx, (u, d) in enumerate(valid_rows, start=1):
                    disc_lines.append(f"{idx}. {u}: {d}")

                team_members_list = "\n".join(disc_lines)

                from collections import Counter
                type_counts = Counter([v[1] for v in valid_rows])
                total_members = len(valid_rows)
                type_percentages = {
                    t: round((count/total_members)*100)
                    for t, count in type_counts.items()
                }

                # Make bar chart
                sns.set_style('whitegrid')
                plt.rcParams.update({'font.family': 'serif'})
                plt.figure(figsize=(10,6))
                sorted_types = sorted(type_counts.keys())
                sorted_counts = [type_counts[t] for t in sorted_types]
                sns.barplot(x=sorted_types, y=sorted_counts, palette='viridis')
                plt.title('DISC Type Distribution', fontsize=16)
                plt.xlabel('DISC Types/Styles', fontsize=14)
                plt.ylabel('Number of Team Members', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                type_distribution_plot = buf.getvalue()
                plt.close()

                # Build a textual table of counts for injection
                # so the LLM sees the real data and doesn't guess
                # We'll include all possible DISC types (including subtypes)
                disc_primaries = ['D','I','S','C']
                disc_subtypes = ['D/i','I/d','I/s','S/i','S/c','C/s','D/c','C/d']
                all_disc = disc_primaries + disc_subtypes

                distribution_table_md = "Type | Count | Percentage\n---|---|---\n"
                for dstyle in all_disc:
                    c = type_counts.get(dstyle, 0)
                    p = type_percentages.get(dstyle, 0)
                    distribution_table_md += f"{dstyle} | {c} | {p}%\n"

                # LLM
                chat_model = ChatOpenAI(
                    openai_api_key=st.secrets['API_KEY'],
                    model_name='gpt-4o-2024-08-06',
                    temperature=0.2
                )

                # initial context
                from langchain.prompts import PromptTemplate
                init_template = PromptTemplate.from_template(initial_context)
                formatted_init = init_template.format(
                    TEAM_SIZE=str(team_size),
                    TEAM_MEMBERS_LIST=team_members_list
                )

                # Generate sections
                section_order = [
                    "Team Profile",
                    "Type Distribution",
                    "Team Insights",
                    "Actions and Next Steps"
                ]
                report_sections = {}
                report_so_far = ""

                for sec_name in section_order:
                    prompt_template = PromptTemplate.from_template(prompts[sec_name])
                    if sec_name == "Type Distribution":
                        # pass the distribution_table to LLM
                        prompt_vars = {
                            "INITIAL_CONTEXT": formatted_init.strip(),
                            "REPORT_SO_FAR": report_so_far.strip(),
                            "DISTRIBUTION_TABLE": distribution_table_md
                        }
                    else:
                        prompt_vars = {
                            "INITIAL_CONTEXT": formatted_init.strip(),
                            "REPORT_SO_FAR": report_so_far.strip()
                        }

                    chain = LLMChain(prompt=prompt_template, llm=chat_model)
                    sec_text = chain.run(**prompt_vars)
                    report_sections[sec_name] = sec_text.strip()
                    report_so_far += f"\n\n{sec_text.strip()}"

                # Output
                for sec_name in section_order:
                    st.markdown(report_sections[sec_name])
                    if sec_name == "Type Distribution":
                        st.header("DISC Type Distribution Plot")
                        st.image(type_distribution_plot, use_column_width=True)

                # PDF with cover
                def build_cover_page(logo_path, company_name, team_name, date_str):
                    elems = []
                    styles = getSampleStyleSheet()

                    cover_title_style = ParagraphStyle(
                        'CoverTitle',
                        parent=styles['Title'],
                        fontName='Times-Bold',
                        fontSize=24,
                        leading=28,
                        alignment=TA_CENTER,
                        spaceAfter=20
                    )
                    cover_text_style = ParagraphStyle(
                        'CoverText',
                        parent=styles['Normal'],
                        fontName='Times-Roman',
                        fontSize=14,
                        alignment=TA_CENTER,
                        spaceAfter=8
                    )

                    elems.append(Spacer(1,80))
                    try:
                        logo = ReportLabImage(logo_path, width=140, height=52)
                        elems.append(logo)
                    except:
                        pass

                    elems.append(Spacer(1,50))
                    title_para = Paragraph("DISC For The Workplace<br/>Team Report", cover_title_style)
                    elems.append(title_para)
                    elems.append(Spacer(1,50))
                    sep = HRFlowable(width="70%", color=colors.darkgoldenrod)
                    elems.append(sep)
                    elems.append(Spacer(1,20))

                    cpara = Paragraph(company_name, cover_text_style)
                    elems.append(cpara)
                    tpara = Paragraph(team_name, cover_text_style)
                    elems.append(tpara)
                    dpara = Paragraph(date_str, cover_text_style)
                    elems.append(dpara)

                    elems.append(Spacer(1,60))
                    elems.append(PageBreak())
                    return elems

                def convert_markdown_to_pdf_with_cover(
                    report_dict,
                    distribution_plot,
                    logo_path,
                    company_name,
                    team_name,
                    date_str
                ):
                    pdf_buf = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
                    elements = []

                    # cover page
                    cover = build_cover_page(logo_path, company_name, team_name, date_str)
                    elements.extend(cover)

                    styles = getSampleStyleSheet()
                    styleH1 = ParagraphStyle(
                        'Heading1Custom',
                        parent=styles['Heading1'],
                        fontName='Times-Bold',
                        fontSize=18,
                        leading=22,
                        spaceAfter=10,
                    )
                    styleH2 = ParagraphStyle(
                        'Heading2Custom',
                        parent=styles['Heading2'],
                        fontName='Times-Bold',
                        fontSize=16,
                        leading=20,
                        spaceAfter=8,
                    )
                    styleH3 = ParagraphStyle(
                        'Heading3Custom',
                        parent=styles['Heading3'],
                        fontName='Times-Bold',
                        fontSize=14,
                        leading=18,
                        spaceAfter=6,
                    )
                    styleH4 = ParagraphStyle(
                        'Heading4Custom',
                        parent=styles['Heading4'],
                        fontName='Times-Bold',
                        fontSize=12,
                        leading=16,
                        spaceAfter=4,
                    )
                    styleN = ParagraphStyle(
                        'Normal',
                        parent=styles['Normal'],
                        fontName='Times-Roman',
                        fontSize=12,
                        leading=14,
                    )
                    styleList = ParagraphStyle(
                        'List',
                        parent=styles['Normal'],
                        fontName='Times-Roman',
                        fontSize=12,
                        leading=14,
                        leftIndent=20,
                    )

                    def process_markdown(md_text):
                        html = markdown(md_text, extras=['tables'])
                        soup = BeautifulSoup(html, 'html.parser')
                        for elem in soup.contents:
                            if isinstance(elem, str):
                                continue
                            if elem.name == 'h1':
                                elements.append(Paragraph(elem.text, styleH1))
                                elements.append(Spacer(1,12))
                            elif elem.name == 'h2':
                                elements.append(Paragraph(elem.text, styleH2))
                                elements.append(Spacer(1,12))
                            elif elem.name == 'h3':
                                elements.append(Paragraph(elem.text, styleH3))
                                elements.append(Spacer(1,12))
                            elif elem.name == 'h4':
                                elements.append(Paragraph(elem.text, styleH4))
                                elements.append(Spacer(1,12))
                            elif elem.name == 'p':
                                elements.append(Paragraph(elem.decode_contents(), styleN))
                                elements.append(Spacer(1,12))
                            elif elem.name == 'ul':
                                for li in elem.find_all('li', recursive=False):
                                    elements.append(Paragraph('• ' + li.text, styleList))
                                    elements.append(Spacer(1,6))
                            elif elem.name == 'table':
                                table_data = []
                                thead = elem.find('thead')
                                if thead:
                                    header_row = []
                                    for th in thead.find_all('th'):
                                        header_row.append(th.get_text(strip=True))
                                    if header_row:
                                        table_data.append(header_row)
                                tbody = elem.find('tbody')
                                if tbody:
                                    rows = tbody.find_all('tr')
                                else:
                                    rows = elem.find_all('tr')
                                for row in rows:
                                    cols = row.find_all(['td','th'])
                                    row_data = [c.get_text(strip=True) for c in cols]
                                    table_data.append(row_data)

                                if table_data:
                                    t = Table(table_data, hAlign='LEFT')
                                    t.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                                        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                    ]))
                                    elements.append(t)
                                    elements.append(Spacer(1,12))
                            else:
                                elements.append(Paragraph(elem.get_text(strip=True), styleN))
                                elements.append(Spacer(1,12))

                    for s_name in ["Team Profile","Type Distribution","Team Insights","Actions and Next Steps"]:
                        process_markdown(report_dict[s_name])
                        if s_name == "Type Distribution":
                            elements.append(Spacer(1,12))
                            distbuf = io.BytesIO(distribution_plot)
                            distimg = ReportLabImage(distbuf, width=400, height=240)
                            elements.append(distimg)
                            elements.append(Spacer(1,12))

                    doc.build(elements)
                    pdf_buf.seek(0)
                    return pdf_buf

                # Build PDF
                pdf_data = convert_markdown_to_pdf_with_cover(
                    report_dict=report_sections,
                    distribution_plot=type_distribution_plot,
                    logo_path=logo_path,
                    company_name=company_name,
                    team_name=team_name,
                    date_str=custom_date
                )

                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_data,
                    file_name="team_disc_report.pdf",
                    mime="application/pdf"
                )
