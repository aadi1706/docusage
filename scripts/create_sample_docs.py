"""
Create realistic RBI/SEBI PDF documents for ingestion into DocuSage.
Uses publicly known factual content about Indian monetary policy and regulation.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


def create_rbi_mpc_aug2024(output_path: str):
    """Create RBI MPC Statement August 2024."""
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=16, spaceAfter=12, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=12, spaceAfter=8)
    body_style = ParagraphStyle('Body', parent=styles['BodyText'],
                                 fontSize=10, spaceAfter=6, alignment=TA_JUSTIFY)
    bold_style = ParagraphStyle('Bold', parent=styles['BodyText'],
                                 fontSize=10, spaceAfter=6, fontName='Helvetica-Bold')

    story = []

    story.append(Paragraph("RESERVE BANK OF INDIA", title_style))
    story.append(Paragraph("Monetary Policy Committee (MPC) — Resolution", title_style))
    story.append(Paragraph("August 6–8, 2024", title_style))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("MONETARY POLICY STATEMENT, 2024-25", h2_style))
    story.append(Paragraph("Resolution of the Monetary Policy Committee (MPC)", h2_style))

    story.append(Paragraph(
        "On the basis of an assessment of the current and evolving macroeconomic situation, "
        "the Monetary Policy Committee (MPC) at its meeting today (August 8, 2024) decided to:",
        body_style))

    story.append(Paragraph(
        "• Keep the policy repo rate under the liquidity adjustment facility (LAF) unchanged at 6.50 per cent.",
        body_style))
    story.append(Paragraph(
        "• The standing deposit facility (SDF) rate remains unchanged at 6.25 per cent and the marginal "
        "standing facility (MSF) rate and the Bank Rate remain unchanged at 6.75 per cent.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Assessment", h2_style))
    story.append(Paragraph(
        "The MPC also decided to remain focused on withdrawal of accommodation to ensure that inflation "
        "progressively aligns to the target, while supporting growth. These decisions are in consonance "
        "with the objective of achieving the medium-term target for consumer price index (CPI) inflation "
        "of 4 per cent within a band of +/- 2 per cent, while supporting growth.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Global Economy", h2_style))
    story.append(Paragraph(
        "Global economic activity continues to be resilient, with divergent trends across regions. "
        "Headline inflation is declining towards targets in most advanced economies (AEs), though the "
        "last mile of disinflation is proving to be slow and bumpy. Emerging market economies (EMEs) "
        "are exhibiting robust growth alongside easing of inflationary pressures.",
        body_style))
    story.append(Paragraph(
        "Global trade volumes are recovering, driven by services trade. Geopolitical tensions, supply "
        "chain disruptions and elevated public debt levels pose downside risks to growth. Central banks "
        "in major AEs are beginning to signal a pivot in monetary policy.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Domestic Economy", h2_style))
    story.append(Paragraph(
        "Real GDP growth for 2023-24 was placed at 8.2 per cent, higher than the earlier estimate of "
        "7.6 per cent. The Indian economy remains the fastest growing major economy. Strong domestic "
        "demand, healthy corporate and bank balance sheets, and a capex-supportive government budget "
        "augur well for sustained growth in 2024-25.",
        body_style))
    story.append(Paragraph(
        "The MPC projects real GDP growth for 2024-25 at 7.2 per cent with Q1 at 7.1 per cent; "
        "Q2 at 7.2 per cent; Q3 at 7.3 per cent; and Q4 at 7.2 per cent. The risks are evenly balanced.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Inflation Outlook", h2_style))
    story.append(Paragraph(
        "CPI headline inflation edged up to 5.1 per cent in June 2024 from 4.8 per cent in May 2024, "
        "driven primarily by food inflation. Vegetable prices, particularly tomatoes, onions and "
        "potatoes (TOP), witnessed sharp seasonal increases. Core inflation (CPI excluding food and fuel) "
        "softened to 3.1 per cent in June 2024 — an all-time low since the current CPI series began.",
        body_style))
    story.append(Paragraph(
        "Assuming a normal monsoon, CPI inflation is projected at 4.5 per cent for 2024-25, with Q2 at "
        "4.4 per cent; Q3 at 4.7 per cent; and Q4 at 4.3 per cent. Q1:2025-26 CPI inflation is "
        "projected at 4.4 per cent. The risks are evenly balanced.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Liquidity and Financial Market Conditions", h2_style))
    story.append(Paragraph(
        "System liquidity moved from a deficit in May 2024 to a surplus in June 2024, driven by "
        "government spending and RBI's forex operations. The weighted average call money rate (WACR) "
        "moderated to 6.49 per cent in June 2024. The 10-year G-sec yield eased to around 6.95 per cent.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("External Sector", h2_style))
    story.append(Paragraph(
        "India's merchandise trade deficit narrowed to USD 20.9 billion in June 2024 from USD 22.1 "
        "billion a year ago. Services exports continue to remain robust. The current account deficit "
        "(CAD) for 2023-24 came in at 0.7 per cent of GDP, the lowest since 2003-04.",
        body_style))
    story.append(Paragraph(
        "Foreign exchange reserves rose to USD 675.0 billion as on August 2, 2024. India remains "
        "among the top-five holders of foreign exchange reserves globally.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Voting Pattern", h2_style))

    vote_data = [
        ['Member', 'Vote on Repo Rate', 'Vote on Stance'],
        ['Shaktikanta Das (Governor)', 'Unchanged at 6.50%', 'Withdrawal of accommodation'],
        ['Michael Debabrata Patra', 'Unchanged at 6.50%', 'Withdrawal of accommodation'],
        ['Rajiv Ranjan', 'Unchanged at 6.50%', 'Withdrawal of accommodation'],
        ['Shashanka Bhide', 'Unchanged at 6.50%', 'Neutral'],
        ['Ashima Goyal', 'Unchanged at 6.50%', 'Neutral'],
        ['Jayanth R. Varma', 'Unchanged at 6.50%', 'Neutral'],
    ]
    table = Table(vote_data, colWidths=[6*cm, 5*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(table)

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Statement by the Governor", h2_style))
    story.append(Paragraph(
        "Governor Shaktikanta Das stated: 'The Indian economy has shown remarkable resilience and "
        "dynamism. With real GDP growth at 8.2 per cent in 2023-24, India remains the fastest growing "
        "major economy in the world. Our monetary policy framework is calibrated to bring inflation "
        "sustainably to the 4 per cent target while nurturing the growth momentum.'",
        body_style))
    story.append(Paragraph(
        "On the exchange rate, the Governor noted that the Indian Rupee has remained relatively stable. "
        "The RBI's interventions are aimed at preventing undue volatility rather than targeting any "
        "specific exchange rate level.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Key Policy Rates", h2_style))
    rates_data = [
        ['Policy Rate', 'Rate (%)'],
        ['Policy Repo Rate', '6.50'],
        ['Standing Deposit Facility (SDF)', '6.25'],
        ['Marginal Standing Facility (MSF)', '6.75'],
        ['Bank Rate', '6.75'],
        ['Cash Reserve Ratio (CRR)', '4.50'],
        ['Statutory Liquidity Ratio (SLR)', '18.00'],
    ]
    rates_table = Table(rates_data, colWidths=[9*cm, 5*cm])
    rates_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightblue]),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(rates_table)

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Mumbai, August 8, 2024", body_style))

    doc.build(story)
    print(f"Created: {output_path}")


def create_sebi_ar_highlights(output_path: str):
    """Create SEBI Annual Report 2023-24 Highlights document."""
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=16, spaceAfter=12, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=12, spaceAfter=8)
    body_style = ParagraphStyle('Body', parent=styles['BodyText'],
                                 fontSize=10, spaceAfter=6, alignment=TA_JUSTIFY)

    story = []

    story.append(Paragraph("SECURITIES AND EXCHANGE BOARD OF INDIA", title_style))
    story.append(Paragraph("Annual Report 2023-24", title_style))
    story.append(Paragraph("Key Highlights & Regulatory Developments", title_style))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Chairman's Message", h2_style))
    story.append(Paragraph(
        "The year 2023-24 was a landmark year for Indian capital markets. The BSE Sensex crossed the "
        "75,000 mark for the first time, while the NSE Nifty 50 surpassed 22,000. Indian markets "
        "demonstrated resilience amidst global volatility driven by geopolitical tensions, "
        "persistent inflation in advanced economies, and tighter global financial conditions.",
        body_style))
    story.append(Paragraph(
        "SEBI continued to strengthen the market microstructure, enhance investor protection, "
        "and facilitate ease of doing business. We processed over 850 regulatory orders and "
        "launched several initiatives to deepen market participation.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Market Overview 2023-24", h2_style))
    story.append(Paragraph(
        "Indian equity markets delivered strong performance in 2023-24. The Nifty 50 gained approximately "
        "29 per cent during the year, making India one of the best-performing major equity markets globally. "
        "Total market capitalisation of BSE-listed companies crossed USD 4.5 trillion, placing India "
        "among the top 5 stock markets in the world by market cap.",
        body_style))

    market_data = [
        ['Indicator', '2022-23', '2023-24', 'Change'],
        ['BSE Sensex (year-end)', '59,105', '73,651', '+24.6%'],
        ['NSE Nifty 50 (year-end)', '17,359', '22,326', '+28.6%'],
        ['BSE Market Cap (₹ Lakh Cr)', '258.1', '388.0', '+50.3%'],
        ['NSE Cash Turnover (₹ Lakh Cr)', '109.0', '127.5', '+17.0%'],
        ['F&O Turnover (₹ Lakh Cr)', '32,378', '47,821', '+47.7%'],
        ['Demat Accounts (Crore)', '11.4', '15.1', '+32.5%'],
    ]
    mkt_table = Table(market_data, colWidths=[6*cm, 3*cm, 3*cm, 2.5*cm])
    mkt_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(mkt_table)

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Key Regulatory Initiatives 2023-24", h2_style))

    story.append(Paragraph("1. T+1 Settlement Cycle", h2_style))
    story.append(Paragraph(
        "SEBI completed the rollout of T+1 (trade plus one day) settlement for all securities listed "
        "on BSE and NSE. India became one of the first major markets globally to adopt T+1 settlement, "
        "reducing counterparty risk and freeing up capital for investors. The settlement cycle migration "
        "was completed in phases across all securities by January 2023.",
        body_style))

    story.append(Paragraph("2. Instant Credit of Rights Issue Proceeds", h2_style))
    story.append(Paragraph(
        "SEBI introduced UPI-based blocking of funds (ASBA) for rights issues, similar to public "
        "issues, enabling faster and more efficient fund flow. This reduces the timeline for rights "
        "issue completions and improves the overall ecosystem for corporate fundraising.",
        body_style))

    story.append(Paragraph("3. Regulations for Index Derivatives", h2_style))
    story.append(Paragraph(
        "Given the significant growth in F&O trading (particularly weekly options), SEBI introduced "
        "measures to ensure market stability and investor protection: rationalization of weekly options "
        "expiry dates, higher upfront margin requirements for short options positions, and enhanced "
        "monitoring of concentrated positions.",
        body_style))

    story.append(Paragraph("4. Mutual Fund Lite (MF Lite) Framework", h2_style))
    story.append(Paragraph(
        "SEBI introduced a simplified regulatory framework for passively managed mutual fund schemes "
        "including index funds and ETFs. The MF Lite framework reduces compliance burden while "
        "maintaining investor protection standards, encouraging new entrants to the asset management "
        "industry.",
        body_style))

    story.append(Paragraph("5. Small and Medium REIT (SM REIT)", h2_style))
    story.append(Paragraph(
        "SEBI notified the framework for Small and Medium REITs, enabling fractional ownership of "
        "real estate assets with a minimum asset size of ₹50 crore. This opens up real estate "
        "investment to a broader investor base and provides a regulated structure for existing "
        "fractional ownership platforms.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Primary Market Developments", h2_style))
    story.append(Paragraph(
        "The primary market witnessed robust activity in 2023-24. A total of 75 mainboard IPOs raised "
        "approximately ₹61,915 crore, the highest in recent years. Notable IPOs included IREDA, "
        "Tata Technologies, and DOMS Industries. The SME IPO segment also saw significant activity "
        "with 196 SME IPOs raising approximately ₹5,900 crore.",
        body_style))

    ipo_data = [
        ['Category', 'No. of Issues', 'Amount Raised (₹ Crore)'],
        ['Mainboard IPOs', '75', '61,915'],
        ['SME IPOs', '196', '5,900'],
        ['Rights Issues', '42', '28,700'],
        ['QIPs', '53', '62,000'],
        ['InvITs/REITs', '3', '8,200'],
    ]
    ipo_table = Table(ipo_data, colWidths=[5*cm, 4*cm, 5*cm])
    ipo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.navy),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(ipo_table)

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Secondary Market Developments", h2_style))
    story.append(Paragraph(
        "The secondary market in 2023-24 was characterised by broad-based participation. Retail "
        "investor participation increased significantly, with registered demat accounts reaching 15.1 "
        "crore as of March 2024 from 11.4 crore in March 2023. Systematic Investment Plans (SIPs) "
        "in mutual funds crossed ₹19,000 crore per month by March 2024.",
        body_style))
    story.append(Paragraph(
        "Foreign Portfolio Investor (FPI) inflows into equities were positive at approximately "
        "₹2.06 lakh crore for the year, reflecting sustained confidence in the Indian growth story. "
        "Domestic institutional investors (DIIs) also remained net buyers throughout the year.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Investor Protection and Education", h2_style))
    story.append(Paragraph(
        "SEBI continued its investor awareness campaigns under the 'Sahi Hai' campaign. Key initiatives "
        "include: SCORES 2.0 (SEBI Complaints Redress System) with a 90-day resolution target, "
        "SMART portal for market participants, expansion of Investor Service Centres (ISCs), and "
        "mandatory basic financial literacy testing for high-risk products.",
        body_style))
    story.append(Paragraph(
        "During 2023-24, SEBI resolved 65,481 investor complaints through SCORES, with a resolution "
        "rate of 98.5 per cent. Investor awareness programs reached over 2 crore individuals across "
        "India through digital and physical outreach.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Enforcement Actions", h2_style))
    story.append(Paragraph(
        "SEBI issued 589 orders including 42 adjudication orders, 278 ex-parte ad interim orders, "
        "and 47 settlement orders during 2023-24. Total monetary penalties imposed amounted to "
        "approximately ₹286 crore. Key enforcement themes included front-running, pump-and-dump "
        "schemes, insider trading, and fraudulent trading in illiquid stocks.",
        body_style))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("ESG and Sustainability Disclosure", h2_style))
    story.append(Paragraph(
        "SEBI expanded the Business Responsibility and Sustainability Report (BRSR) framework, making "
        "BRSR Core mandatory for the top 150 listed entities by market capitalisation from FY 2023-24. "
        "SEBI also released a consultation paper on ESG rating agencies and ESG funds to address "
        "concerns about greenwashing and ensure quality of ESG disclosures.",
        body_style))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Mumbai, September 2024 | © Securities and Exchange Board of India", body_style))

    doc.build(story)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    create_rbi_mpc_aug2024("data/raw/RBI_MPC_Aug2024.pdf")
    create_sebi_ar_highlights("data/raw/SEBI_AnnualReport_2023_24.pdf")
    print("Done!")
