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
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from markdown2 import markdown
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------------
# IMPORTANT: Updated guidance integrated into LLM prompts so that it is reflected
#            in the generated text. 
# ----------------------------------------------------------------------------------

# -------------------------------
# DISC Categories and Hybrid Mapping
# -------------------------------
# Primary DISC Types:
#   D (Drive), I (Influence), S (Support), C (Clarity)
#
# Hybrid Types (second letter lowercase, slash notation for final text):
#   DI -> D/i
#   ID -> I/d
#   IS -> I/s
#   SI -> S/i
#   SC -> S/c
#   CS -> C/s
#   DC -> D/c
#   CD -> C/d
#
# For functionality within the app, the user can select uppercase. 
# We'll instruct the LLM to reference them in slash-lowercase format.

disc_primaries = ['D', 'I', 'S', 'C']
disc_subtypes = ['DI', 'ID', 'IS', 'SI', 'SC', 'CS', 'DC', 'CD']
disc_types = disc_primaries + disc_subtypes

# -------------------------------
# Conversion Function (Optional)
# -------------------------------
# If you want to automatically convert the uppercase hybrid types to slash-lowercase
# in the final text, you could post-process the LLM output. 
# However, we are instructing the LLM to do it directly via prompts.

hybrid_map = {
    'DI': 'D/i',
    'ID': 'I/d',
    'IS': 'I/s',
    'SI': 'S/i',
    'SC': 'S/c',
    'CS': 'C/s',
    'DC': 'D/c',
    'CD': 'C/d'
}

def randomize_types_callback():
    randomized_types = [random.choice(disc_types) for _ in range(int(st.session_state['team_size']))]
    for i in range(int(st.session_state['team_size'])):
        key = f'disc_{i}'
        st.session_state[key] = randomized_types[i]

# -------------------------------
# Updated LLM Prompts
# -------------------------------

initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments using the DISC framework.

DISC stands for Drive (D), Influence (I), Support (S), and Clarity (C). 
In addition to these four primary types, we also have hybrid types that combine two primary styles (e.g., D/i, I/d, S/c, etc.), 
where the second letter is lowercase and separated by a slash.

**Team Size:** {TEAM_SIZE}

**Team Members and their DISC Results:**

{TEAM_MEMBERS_LIST}

Your goal:
Generate a comprehensive team personality report based on DISC, adhering to these requirements:
1. Refer to DISC as Drive, Influence, Support, Clarity (not Dominance, Influence, Steadiness, Conscientiousness).
2. Use "type" or "style" instead of "dimension."
3. For hybrid types, always use slash-lowercase for the second letter (e.g., D/i, C/s).
4. Include a section listing "Types Not on the Team" with the same brief info as "Types on the Team."
5. Emphasize "Dominant Types" and "Less Represented Types."
6. Provide a brief summary of the distribution in the same section.
7. No mention of any other frameworks (e.g., MBTI, Enneagram).
8. Round all percentages to the nearest whole number.
9. Maintain a professional, neutral tone.

Below is the structure you should follow for this DISC report:
1. Team Profile
2. Type Distribution
3. Team Insights
4. Actions and Next Steps
"""

prompts = {
    "Team Profile": """
{INITIAL_CONTEXT}

**Your Role:**

Write the **Team Profile** section of the report (Section 1).

**Section 1: Team Profile**
- Briefly introduce the DISC framework as Drive (D), Influence (I), Support (S), Clarity (C).
- Mention that there are hybrid types, and whenever you refer to them, use slash-lowercase for the second letter (e.g., D/i, S/c).
- Outline the core characteristics of each DISC type/style present on the team (primary or hybrid).
- Describe how this combination of types affects foundational team dynamics.
- Required length: Approximately 500 words.

**Begin your section below:**
""",
    "Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Type Distribution** section of the report (Section 2).

**Section 2: Type Distribution**
- Provide a breakdown (table or list) of how many team members fall into each DISC type/style.
- Convert these counts into percentages (rounded to nearest whole number).
- Under a subheading "Types on the Team," list each type/style present with 1–2 bullet points describing it, plus count and percentage.
- Under a subheading "Types Not on the Team," list any type/style not represented, with the same 1–2 bullet points (and note count = 0).
- Include a brief note on how these distributions influence communication and decision-making.
- Provide separate subheadings for "Dominant Types" and "Less Represented Types," discussing how each impacts the team.
- End this section with a brief "Summary" subheading summarizing key insights.
- Required length: Approximately 500 words.

**Continue the report by adding your section below:**
""",
    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Team Insights** section of the report (Section 3).

**Section 3: Team Insights**
Create the following subheadings:

1. **Strengths**  
   - Identify at least four key strengths emerging from the dominant DISC types/styles.
   - Each strength should be in **bold** as a single sentence, followed by a paragraph explanation.

2. **Potential Blind Spots**  
   - Identify at least four areas of improvement or challenges based on the DISC composition.
   - Each blind spot should be in **bold** as a single sentence, followed by a paragraph explanation.

3. **Communication**  
   - Describe any notable communication patterns relevant to the team’s mix of DISC types/styles.

4. **Teamwork**  
   - Discuss how presence/absence of certain DISC types/styles can shape collaboration and delegation.

5. **Conflict**  
   - Explain potential sources of conflict given the DISC composition, and offer suggestions for healthy conflict resolution.

- Required length: Approximately 700 words total.

**Continue the report by adding your section below:**
""",
    "Actions and Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write the **Actions and Next Steps** section of the report (Section 4).

**Section 4: Actions and Next Steps**
- Provide actionable recommendations for team leaders, referencing the DISC composition.
- Use subheadings for each major recommendation area.
- Offer a brief justification for each recommendation, linking it to the specific DISC types/styles involved.
- Present the recommendations as bullet points or numbered lists of specific actions.
- End your output immediately after the last bullet (no concluding paragraph).
- Required length: Approximately 400 words.

**Conclude the report by adding your section below:**
"""
}

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title('DISC Team Report Generator')

# Initialize or retrieve team_size from session
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
    d_type = st.selectbox(
        f'Team Member {i+1}',
        options=['Select DISC Type'] + disc_types,
        key=f'disc_{i}'
    )
    if d_type != 'Select DISC Type':
        team_disc_types.append(d_type)
    else:
        team_disc_types.append(None)

if st.button('Generate Report'):
    if None in team_disc_types:
        st.error('Please select DISC types for all team members.')
    else:
        with st.spinner('Generating report, please wait...'):
            # Prepare the string listing team members and their DISC results
            team_members_list = "\n".join([
                f"{i+1}. Team Member {i+1}: {d_type}"
                for i, d_type in enumerate(team_disc_types)
            ])

            # Calculate counts and percentages
            type_counts = Counter(team_disc_types)
            total_members = len(team_disc_types)
            type_percentages = {
                t: round((c / total_members) * 100)
                for t, c in type_counts.items()
            }

            # Generate bar plot for distribution
            sns.set_style('whitegrid')
            plt.rcParams.update({'font.family': 'serif'})

            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=list(type_counts.keys()),
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

            # Initialize LLM
            chat_model = ChatOpenAI(
                openai_api_key=st.secrets['API_KEY'],
                model_name='gpt-4o-2024-08-06',
                temperature=0.2
            )

            # Format the initial context
            initial_context_template = PromptTemplate.from_template(initial_context)
            formatted_initial_context = initial_context_template.format(
                TEAM_SIZE=str(team_size),
                TEAM_MEMBERS_LIST=team_members_list
            )

            # Generate sections in order
            section_order = [
                "Team Profile",
                "Type Distribution",
                "Team Insights",
                "Actions and Next Steps"
            ]
            report_sections = {}
            report_so_far = ""

            for section_name in section_order:
                prompt_template = PromptTemplate.from_template(prompts[section_name])
                prompt_variables = {
                    "INITIAL_CONTEXT": formatted_initial_context.strip(),
                    "REPORT_SO_FAR": report_so_far.strip()
                }
                chat_chain = LLMChain(prompt=prompt_template, llm=chat_model)
                section_text = chat_chain.run(**prompt_variables)
                report_sections[section_name] = section_text.strip()
                report_so_far += f"\n\n{section_text.strip()}"

            final_report = "\n\n".join([report_sections[s] for s in section_order])

            # Display the final report in Streamlit
            for section_name in section_order:
                st.markdown(report_sections[section_name])
                if section_name == "Type Distribution":
                    st.header("DISC Type Distribution Plot")
                    st.image(type_distribution_plot, use_column_width=True)

            # -------------------------------
            # PDF Generation
            # -------------------------------
            def convert_markdown_to_pdf(report_sections_dict, distribution_plot):
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()

                styleN = ParagraphStyle(
                    'Normal',
                    parent=styles['Normal'],
                    fontName='Times-Roman',
                    fontSize=12,
                    leading=14,
                )
                styleH = ParagraphStyle(
                    'Heading',
                    parent=styles['Heading1'],
                    fontName='Times-Bold',
                    fontSize=18,
                    leading=22,
                    spaceAfter=10,
                )
                styleList = ParagraphStyle(
                    'List',
                    parent=styles['Normal'],
                    fontName='Times-Roman',
                    fontSize=12,
                    leading=14,
                    leftIndent=20,
                )

                def process_markdown(text):
                    html = markdown(text, extras=['tables'])
                    soup = BeautifulSoup(html, 'html.parser')
                    for elem in soup.contents:
                        if isinstance(elem, str):
                            continue

                        # Process any tables in the text
                        if elem.name == 'table':
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
                                cols = row.find_all(['td', 'th'])
                                table_row = [col.get_text(strip=True) for col in cols]
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

                        # Process headings
                        elif elem.name in ['h1', 'h2', 'h3']:
                            elements.append(Paragraph(elem.text, styleH))
                            elements.append(Spacer(1, 12))

                        # Process paragraph text
                        elif elem.name == 'p':
                            elements.append(Paragraph(elem.decode_contents(), styleN))
                            elements.append(Spacer(1, 12))

                        # Process lists
                        elif elem.name == 'ul':
                            for li in elem.find_all('li', recursive=False):
                                elements.append(Paragraph('• ' + li.text, styleList))
                                elements.append(Spacer(1, 12))
                        else:
                            # Fallback for anything else
                            elements.append(Paragraph(elem.get_text(strip=True), styleN))
                            elements.append(Spacer(1, 12))

                # Build PDF content from each report section
                for sec in section_order:
                    process_markdown(report_sections_dict[sec])
                    if sec == "Type Distribution":
                        # Insert the distribution plot image
                        elements.append(Spacer(1, 12))
                        img_buffer = io.BytesIO(distribution_plot)
                        img = ReportLabImage(img_buffer, width=400, height=240)
                        elements.append(img)
                        elements.append(Spacer(1, 12))

                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            pdf_data = convert_markdown_to_pdf(report_sections, type_distribution_plot)

            st.download_button(
                label="Download Report as PDF",
                data=pdf_data,
                file_name="team_disc_report.pdf",
                mime="application/pdf"
            )
