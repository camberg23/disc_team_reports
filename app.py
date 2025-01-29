import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
# DISC Types (with slash-lowercase for UI, plus code mapping)
# ----------------------------------------------------------------------------------

# Internally, let's keep original code versions for randomization logic
disc_primaries_code = ['D', 'I', 'S', 'C']
disc_subtypes_code = ['DI', 'ID', 'IS', 'SI', 'SC', 'CS', 'DC', 'CD']
disc_types_code = disc_primaries_code + disc_subtypes_code

# For user display in Streamlit, we use slash-lowercase
disc_code_to_ui = {
    'D':  'D',
    'I':  'I',
    'S':  'S',
    'C':  'C',
    'DI': 'D/i',
    'ID': 'I/d',
    'IS': 'I/s',
    'SI': 'S/i',
    'SC': 'S/c',
    'CS': 'C/s',
    'DC': 'D/c',
    'CD': 'C/d'
}
disc_ui_to_code = {v: k for k, v in disc_code_to_ui.items()}

# We'll build a UI list for selection
disc_types_ui = list(disc_code_to_ui.values())  # e.g., ['D','I','S','C','D/i','I/d',...]

def randomize_types_callback():
    """Randomly assign DISC types (in code form), then store slash-lowercase in session_state."""
    randomized_code_types = [random.choice(disc_types_code) for _ in range(int(st.session_state['team_size']))]
    for i in range(int(st.session_state['team_size'])):
        key = f'disc_{i}'
        # Convert code to slash-lowercase for display
        st.session_state[key] = disc_code_to_ui[randomized_code_types[i]]

# ----------------------------------------------------------------------------------
# Updated Prompts
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
    "Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

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

# -- Cover Page Info
st.subheader("Cover Page Details")
logo_path = "truity_logo.png"  # Hardcode your logo name or path
company_name = st.text_input("Company Name (for cover page)", "Example Corp")
team_name = st.text_input("Team Name (for cover page)", "Sales Team")
today_str = datetime.date.today().strftime("%B %d, %Y")
custom_date = st.text_input("Date (for cover page)", today_str)

if 'team_size' not in st.session_state:
    st.session_state['team_size'] = 5

team_size = st.number_input(
    'Enter the number of team members (up to 30)', 
    min_value=1, max_value=30, value=5, key='team_size'
)

st.button('Randomize Types', on_click=randomize_types_callback)

st.subheader('Enter DISC results for each team member')

for i in range(int(team_size)):
    if f'disc_{i}' not in st.session_state:
        st.session_state[f'disc_{i}'] = 'Select DISC Type'

team_disc_types = []
for i in range(int(team_size)):
    user_choice = st.selectbox(
        f'Team Member {i+1}',
        options=['Select DISC Type'] + disc_types_ui,
        key=f'disc_{i}'
    )
    if user_choice != 'Select DISC Type':
        # Convert slash-lowercase UI choice back to code form
        code_choice = disc_ui_to_code[user_choice]
        team_disc_types.append(code_choice)
    else:
        team_disc_types.append(None)

if st.button('Generate Report'):
    if None in team_disc_types:
        st.error('Please select DISC types for all team members.')
    else:
        with st.spinner('Generating report, please wait...'):
            # Build a list string for LLM context, but convert code to slash-lowercase in the text
            team_members_list = ""
            for i, code_type in enumerate(team_disc_types):
                slash_type = disc_code_to_ui[code_type]  # e.g., 'D/i'
                team_members_list += f"{i+1}. Team Member {i+1}: {slash_type}\n"

            from collections import Counter
            type_counts = Counter(team_disc_types)
            total_members = len(team_disc_types)
            type_percentages = {
                t: round((c / total_members) * 100)
                for t, c in type_counts.items()
            }

            # Make bar chart (label the x-axis with slash-lowercase for clarity)
            slash_labels = [disc_code_to_ui[t] for t in type_counts.keys()]

            sns.set_style('whitegrid')
            plt.rcParams.update({'font.family': 'serif'})
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=slash_labels,
                y=list(type_counts.values()),
                palette='viridis'
            )
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

            # LLM
            chat_model = ChatOpenAI(
                openai_api_key=st.secrets['API_KEY'],
                model_name='gpt-4o-2024-08-06',
                temperature=0.2
            )

            # Format context
            initial_context_template = PromptTemplate.from_template(initial_context)
            formatted_initial_context = initial_context_template.format(
                TEAM_SIZE=str(team_size),
                TEAM_MEMBERS_LIST=team_members_list
            )

            section_order = [
                "Team Profile",
                "Type Distribution",
                "Team Insights",
                "Actions and Next Steps"
            ]
            report_sections = {}
            report_so_far = ""

            # Generate each section
            for section_name in section_order:
                prompt_template = PromptTemplate.from_template(prompts[section_name])
                prompt_vars = {
                    "INITIAL_CONTEXT": formatted_initial_context.strip(),
                    "REPORT_SO_FAR": report_so_far.strip()
                }
                chain = LLMChain(prompt=prompt_template, llm=chat_model)
                section_text = chain.run(**prompt_vars)
                report_sections[section_name] = section_text.strip()
                report_so_far += f"\n\n{section_text.strip()}"

            # Streamlit display
            for section_name in section_order:
                st.markdown(report_sections[section_name])
                if section_name == "Type Distribution":
                    st.header("DISC Type Distribution Plot")
                    st.image(type_distribution_plot, use_column_width=True)

            # ----------------------------------------------------------
            # Cover Page + PDF Generation
            # ----------------------------------------------------------

            def build_cover_page(logo_path, company_name, team_name, date_str):
                cover_elems = []
                styles = getSampleStyleSheet()

                # Title style
                cover_title_style = ParagraphStyle(
                    'CoverTitle',
                    parent=styles['Title'],
                    fontName='Times-Bold',
                    fontSize=24,
                    leading=28,
                    alignment=TA_CENTER,
                    spaceAfter=20
                )
                # Normal center
                cover_text_style = ParagraphStyle(
                    'CoverText',
                    parent=styles['Normal'],
                    fontName='Times-Roman',
                    fontSize=14,
                    alignment=TA_CENTER,
                    spaceAfter=8
                )

                # Some vertical space
                cover_elems.append(Spacer(1, 80))

                # Logo ~ half size
                try:
                    logo = ReportLabImage(logo_path, width=140, height=52)
                    cover_elems.append(logo)
                except:
                    pass

                cover_elems.append(Spacer(1, 50))

                # "DISC For The Workplace" + "Team Report"
                title_para = Paragraph("DISC For The Workplace<br/>Team Report", cover_title_style)
                cover_elems.append(title_para)

                cover_elems.append(Spacer(1, 50))

                # Horizontal line
                sep = HRFlowable(width="70%", color=colors.darkgoldenrod)
                cover_elems.append(sep)
                cover_elems.append(Spacer(1, 20))

                # Company, Team, Date
                comp_para = Paragraph(company_name, cover_text_style)
                cover_elems.append(comp_para)
                tm_para = Paragraph(team_name, cover_text_style)
                cover_elems.append(tm_para)
                dt_para = Paragraph(date_str, cover_text_style)
                cover_elems.append(dt_para)

                cover_elems.append(Spacer(1, 60))
                cover_elems.append(PageBreak())

                return cover_elems

            def convert_markdown_to_pdf_with_cover(report_sections_dict, distribution_plot,
                                                   logo_path, company_name, team_name, date_str):
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                elements = []

                # 1) Cover
                cover_page = build_cover_page(logo_path, company_name, team_name, date_str)
                elements.extend(cover_page)

                # 2) Main content
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

                def process_md(md_text):
                    html = markdown(md_text, extras=['tables'])
                    soup = BeautifulSoup(html, 'html.parser')
                    for elem in soup.contents:
                        if isinstance(elem, str):
                            continue

                        if elem.name == 'h1':
                            elements.append(Paragraph(elem.text, styleH1))
                            elements.append(Spacer(1, 12))
                        elif elem.name == 'h2':
                            elements.append(Paragraph(elem.text, styleH2))
                            elements.append(Spacer(1, 12))
                        elif elem.name == 'h3':
                            elements.append(Paragraph(elem.text, styleH3))
                            elements.append(Spacer(1, 12))
                        elif elem.name == 'h4':
                            elements.append(Paragraph(elem.text, styleH4))
                            elements.append(Spacer(1, 12))
                        elif elem.name == 'p':
                            elements.append(Paragraph(elem.decode_contents(), styleN))
                            elements.append(Spacer(1, 12))
                        elif elem.name == 'ul':
                            for li in elem.find_all('li', recursive=False):
                                elements.append(Paragraph('• ' + li.text, styleList))
                                elements.append(Spacer(1, 6))
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
                                table_row = [c.get_text(strip=True) for c in cols]
                                table_data.append(table_row)

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
                                elements.append(Spacer(1, 12))
                        else:
                            elements.append(Paragraph(elem.get_text(strip=True), styleN))
                            elements.append(Spacer(1, 12))

                # Process each of the 4 sections
                for sec in ["Team Profile", "Type Distribution", "Team Insights", "Actions and Next Steps"]:
                    process_md(report_sections_dict[sec])
                    if sec == "Type Distribution":
                        elements.append(Spacer(1, 12))
                        img_buf = io.BytesIO(distribution_plot)
                        img = ReportLabImage(img_buf, width=400, height=240)
                        elements.append(img)
                        elements.append(Spacer(1, 12))

                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            # Build PDF with cover page
            pdf_data = convert_markdown_to_pdf_with_cover(
                report_sections,
                type_distribution_plot,
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
