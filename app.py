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
import json
import numpy as np  # NEW: Import numpy as np

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
# Improved Plotting Functions (for DISC)
# ----------------------------------------------------------------------------------
def _generate_pie_chart(data, slices, scaling_factor=1.6, label_multiplier=1.3):
    """Generates an exploded pie chart that does NOT double-count people. Each key in 'data' is counted once."""
    # Filter out slices with 0 counts
    filtered_slices = [s for s in slices if data.get(s['label'], 0) > 0]
    
    total = sum(data.get(s['label'], 0) for s in filtered_slices)
    if total == 0:
        print("No data to plot.")
        return None

    current_angle = 0
    for s in filtered_slices:
        count = data.get(s['label'], 0)
        proportion = count / total
        angle = proportion * 360
        s['theta1'] = current_angle
        s['theta2'] = current_angle + angle
        s['radius'] = 1.0 + proportion * scaling_factor
        current_angle += angle

    fig, ax = plt.subplots(figsize=(8,8))
    for s in filtered_slices:
        wedge = plt.matplotlib.patches.Wedge(
            center=(0, 0),
            r=s['radius'],
            theta1=s['theta1'],
            theta2=s['theta2'],
            color=s['color'],
            edgecolor='white'
        )
        ax.add_patch(wedge)

    # Move labels farther out using label_multiplier
    for s in filtered_slices:
        theta_mid = (s['theta1'] + s['theta2']) / 2
        x = label_multiplier * s['radius'] * np.cos(np.radians(theta_mid))
        y = label_multiplier * s['radius'] * np.sin(np.radians(theta_mid))
        ax.text(
            x, y, s['label'],
            ha='center', va='center', fontsize=12, color=s['color'], fontweight='bold'
        )

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def plot_disc_chart(data):
    """
    Pie chart slices covering all 4 single-letter styles (D,I,S,C) plus
    8 hybrids (D/i, I/d, I/s, S/i, S/c, C/s, D/c, C/d), ensuring
    no double-counting in the final display.
    """
    slices = [
        {'label': 'D',    'color': '#3E7279'},
        {'label': 'I',    'color': '#D2A26C'},
        {'label': 'S',    'color': '#C5744B'},
        {'label': 'C',    'color': '#4D8570'},
        {'label': 'D/i',  'color': '#B6823E'},
        {'label': 'I/d',  'color': '#B8A04C'},  # new color for I/d
        {'label': 'I/s',  'color': '#DD7C65'},
        {'label': 'S/i',  'color': '#D35858'},
        {'label': 'S/c',  'color': '#87AC73'},
        {'label': 'C/s',  'color': '#6E9973'},
        {'label': 'D/c',  'color': '#355760'},
        {'label': 'C/d',  'color': '#7B8A2A'}
    ]
    return _generate_pie_chart(data, slices, scaling_factor=1.6, label_multiplier=1.3)

# ----------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------
initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments using the DISC framework.

DISC stands for Drive (D), Influence (I), Support (S), Clarity (C).
We also have hybrid types that combine two primary styles (e.g., D/i, I/s, S/c),
where the second letter is lowercase and separated by a slash.

**Team Size:** {TEAM_SIZE}

**Team Members and their DISC Results:**

{TEAM_MEMBERS_LIST}

**Team Members by DISC Type:**

{TYPE_PEOPLE_DICT}

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
9. IMPORTANT: always refer to it as 'your team' or 'the team' and NEVER as 'our team' (you're not on their team!)
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
    "Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Actual Type Counts & Percentages**:
{DISTRIBUTION_TABLE}

**Your Role:**

Write the **Type Distribution** section of the report (Section 2).

## Section 2: Type Distribution

Please follow these steps in your text (in Markdown):

1. **Primary Styles Table**  
   Present the four primary styles (D, I, S, C) with their count and percentage. Provide short descriptive text for each style.  

2. **Hybrid Types Table**  
   Present each hybrid type (e.g., D/i, I/d, I/s, etc.) with its count and percentage, including a concise description for each.  

3. **Dominant Types**  
   Explain how the most common styles or hybrids influence communication, decision-making, etc.

4. **Less Represented Types**  
   Highlight what might be missing from the team if certain styles/hybrids are low or at 0% count.

5. **Summary**  
   Conclude in 2–3 sentences about the overall distribution and how it may shape the team's approach to collaboration and problem-solving.

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

                # -------------------------------------------------------------
                # 1) Define core styles + hybrid types
                disc_primaries = ['D','I','S','C']
                disc_hybrids = ['D/i','I/d','I/s','S/i','S/c','C/s','D/c','C/d']
                all_disc = disc_primaries + disc_hybrids

                # 2) Build separate counters for "core styles" vs. "full types"
                style_counts = {s: 0 for s in disc_primaries}
                type_counts = {t: 0 for t in all_disc}

                # 3) Count each row's DISC code in both style_counts and type_counts
                for nm, d in valid_rows:
                    # Always increment the specific type (e.g., D/i) in type_counts
                    if d in type_counts:
                        type_counts[d] += 1
                    else:
                        type_counts[d] = 1

                    # For style_counts, increment only the single-letter "primary" style.
                    # If it's hybrid (e.g., 'D/i'), split and use the first uppercase letter.
                    if '/' in d:
                        primary = d.split('/')[0]
                        if primary in style_counts:
                            style_counts[primary] += 1
                    else:
                        if d in style_counts:
                            style_counts[d] += 1

                total_members = len(valid_rows)

                # 4) Build a table for the four core styles
                core_style_table_md = "## Team Composition by Core Style\n"
                core_style_table_md += "Style | Count | Percentage\n"
                core_style_table_md += "---|---|---\n"
                for style in disc_primaries:
                    c = style_counts[style]
                    pct = round((c / total_members)*100) if total_members > 0 else 0
                    core_style_table_md += f"{style} | {c} | {pct}%\n"

                # 5) Build a table for all DISC hybrid types
                type_table_md = "## Team Composition by Hybrid Type\n"
                type_table_md += "Type | Count | Percentage\n"
                type_table_md += "---|---|---\n"
                for t in disc_hybrids:
                    c = type_counts[t]
                    pct = round((c / total_members)*100) if total_members > 0 else 0
                    type_table_md += f"{t} | {c} | {pct}%\n"

                # 6) Combine them into one Markdown string for the prompt or display
                distribution_table_md = core_style_table_md + "\n\n" + type_table_md
                # -------------------------------------------------------------

                # Build dictionary mapping DISC types to list of people
                # (For hybrid types, also add the person to the primary style.)
                type_to_people = { d: [] for d in all_disc }
                for name, d in valid_rows:
                    if '/' in d:
                        primary = d.split('/')[0]
                        if primary in type_to_people:
                            type_to_people[primary].append(name)
                        if d in type_to_people:
                            type_to_people[d].append(name)
                        else:
                            type_to_people[d] = [name]
                    else:
                        if d in type_to_people:
                            type_to_people[d].append(name)
                        else:
                            type_to_people[d] = [name]

                type_people_json = json.dumps(type_to_people, indent=2)

                # Build pie chart of total types (no double-counting)
                type_distribution_plot = plot_disc_chart(type_counts)

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
                    TEAM_MEMBERS_LIST=team_members_list,
                    TYPE_PEOPLE_DICT=type_people_json
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

                # Output to Streamlit
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
                            distimg = ReportLabImage(distbuf, width=400, height=400)
                            elements.append(distimg)
                            elements.append(Spacer(1,12))

                    doc.build(elements)
                    pdf_buf.seek(0)
                    return pdf_buf

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
