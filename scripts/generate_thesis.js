const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition
} = require("docx");

// ─── Helpers ────────────────────────────────────────────────
const SKILL_DIR = "C:\\Users\\cugut\\AppData\\Roaming\\Claude\\local-agent-mode-sessions\\skills-plugin\\8ad6e826-16af-4d65-94e7-20e1a876661c\\bd26b49d-ede7-4401-ab80-c837067b2eba\\skills\\docx";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

// A4 page
const PAGE_W = 11906;
const PAGE_H = 16838;
const MARGIN = 1440; // 1 inch
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9026

function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun({ text, ...opts }));
  } else if (Array.isArray(text)) {
    text.forEach(t => {
      if (typeof t === "string") runs.push(new TextRun(t));
      else runs.push(new TextRun(t));
    });
  }
  return new Paragraph({
    spacing: { after: 200, line: 480 }, // double-space (480 = 2x240)
    indent: opts.indent ? { firstLine: 720 } : undefined,
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
    ...opts.paraOpts,
    children: runs,
  });
}

function bold(text) { return { text, bold: true }; }
function italic(text) { return { text, italics: true }; }

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 240 },
    children: [new TextRun({ text, bold: true, size: 28, font: "Times New Roman" })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 200 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Times New Roman" })],
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 160 },
    children: [new TextRun({ text, bold: true, italics: true, size: 24, font: "Times New Roman" })],
  });
}

function para(runs, indent = true) {
  const children = runs.map(r => {
    if (typeof r === "string") return new TextRun({ text: r, size: 24, font: "Times New Roman" });
    return new TextRun({ size: 24, font: "Times New Roman", ...r });
  });
  return new Paragraph({
    spacing: { after: 200, line: 480 },
    indent: indent ? { firstLine: 720 } : undefined,
    alignment: AlignmentType.JUSTIFIED,
    children,
  });
}

function paraNoIndent(runs) { return para(runs, false); }

function emptyLine() {
  return new Paragraph({ spacing: { after: 200 }, children: [] });
}

// Table helper
function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: "D9E2F3", type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: h, bold: true, size: 20, font: "Times New Roman" })],
        })],
      })
    ),
  });

  const dataRows = rows.map(row =>
    new TableRow({
      children: row.map((cell, i) =>
        new TableCell({
          borders,
          width: { size: colWidths[i], type: WidthType.DXA },
          margins: cellMargins,
          children: [new Paragraph({
            alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
            children: [new TextRun({ text: String(cell), size: 20, font: "Times New Roman" })],
          })],
        })
      ),
    })
  );

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

function tableCaption(text) {
  return new Paragraph({
    spacing: { before: 240, after: 120 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, bold: true, italics: true, size: 20, font: "Times New Roman" })],
  });
}

function tableNote(text) {
  return new Paragraph({
    spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, font: "Times New Roman" })],
  });
}

function equation(text) {
  return new Paragraph({
    spacing: { before: 200, after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, italics: true, size: 24, font: "Times New Roman" })],
  });
}

// ─── BUILD DOCUMENT ─────────────────────────────────────────
async function buildThesis() {
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "Times New Roman", size: 24 },
        },
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Times New Roman" },
          paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: "Times New Roman" },
          paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 1 },
        },
        {
          id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: "Times New Roman" },
          paragraph: { spacing: { before: 200, after: 160 }, outlineLevel: 2 },
        },
      ],
    },
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [{
            level: 0, format: LevelFormat.BULLET, text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          }],
        },
      ],
    },
    sections: [
      // ═══════ TITLE PAGE ═══════
      {
        properties: {
          page: {
            size: { width: PAGE_W, height: PAGE_H },
            margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
          },
        },
        children: [
          emptyLine(), emptyLine(), emptyLine(), emptyLine(),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [new TextRun({ text: "DOKUZ EYLUL UNIVERSITY", size: 28, bold: true, font: "Times New Roman" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [new TextRun({ text: "FACULTY OF ECONOMICS AND ADMINISTRATIVE SCIENCES", size: 24, font: "Times New Roman" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [new TextRun({ text: "Department of Economics", size: 24, font: "Times New Roman" })],
          }),
          emptyLine(), emptyLine(), emptyLine(),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 400 },
            children: [new TextRun({ text: "IPO Fever and the Cost of the Crowd:", size: 36, bold: true, font: "Times New Roman" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 400 },
            children: [new TextRun({ text: "Underpricing, Herding, and Market Manipulation", size: 32, bold: true, font: "Times New Roman" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 100 },
            children: [new TextRun({ text: "in Borsa Istanbul IPOs, 2020\u20132025", size: 32, bold: true, font: "Times New Roman" })],
          }),
          emptyLine(), emptyLine(), emptyLine(), emptyLine(),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
            children: [new TextRun({ text: "Graduation Thesis", size: 24, italics: true, font: "Times New Roman" })],
          }),
          emptyLine(),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 100 },
            children: [new TextRun({ text: "Supervisors:", size: 24, bold: true, font: "Times New Roman" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 100 },
            children: [new TextRun({ text: "Prof. Dr. M. Banu Durukan Sali", size: 24, font: "Times New Roman" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 400 },
            children: [new TextRun({ text: "Assoc. Prof. Dr. Efe Caglar Cagli", size: 24, font: "Times New Roman" })],
          }),
          emptyLine(), emptyLine(),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 100 },
            children: [new TextRun({ text: "2026", size: 24, font: "Times New Roman" })],
          }),
        ],
      },

      // ═══════ MAIN BODY ═══════
      {
        properties: {
          page: {
            size: { width: PAGE_W, height: PAGE_H },
            margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
          },
        },
        headers: {
          default: new Header({
            children: [new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [new TextRun({ text: "IPO Fever and the Cost of the Crowd", italics: true, size: 18, font: "Times New Roman" })],
            })],
          }),
        },
        footers: {
          default: new Footer({
            children: [new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "Times New Roman" })],
            })],
          }),
        },
        children: [
          // ═══════ 1. INTRODUCTION ═══════
          heading1("1. Introduction"),

          para([
            "This study examines three interconnected dimensions of initial public offering (IPO) markets in Borsa Istanbul (BIST) during the 2020\u20132025 period: systematic underpricing, investor herding behaviour, and the price effects of regulatory enforcement against market manipulation. ",
            "Turkey experienced an unprecedented IPO boom in this period, with 209 companies going public while the economy simultaneously endured extreme inflation (peaking at 85.5% year-over-year in October 2022), sharp currency depreciation, and heightened retail investor participation. ",
            "These conditions create a natural laboratory for testing whether behavioural anomalies documented in developed markets also manifest\u2014and potentially amplify\u2014in an emerging market setting characterised by high uncertainty and information asymmetry.",
          ]),

          para([
            "The central contribution of this thesis is methodological. ",
            "BIST imposes a \u00b110% daily price limit on equities, which means that conventional first-day return measures systematically understate true IPO underpricing whenever an IPO hits the upper limit (tavan) on its first trading day. ",
            "We introduce a ",
            bold("consecutive daily limit-up"),
            " methodology\u2014locally known as ",
            italic("tavan serisi"),
            " (lit. \u2018ceiling series\u2019)\u2014that measures underpricing as the cumulative return from the IPO-day close to the first day on which the stock trades freely (i.e., does not hit the daily price limit). ",
            "This measurement follows the approach described by Loughran, Ritter, and Rydqvist (1994, updated 2026) for markets with binding daily price limits, where initial returns are computed using the first closing price at which the limit is not binding. Following the terminology in the Chinese IPO literature (Wang et al., 2018), we define the ",
            bold("limit-up streak length"),
            " as the number of consecutive trading days on which the stock closes at the upper price limit. ",
            "Using split-adjusted daily closing prices from Yahoo Finance for all 209 IPOs, we document that 67% of IPOs hit the upper limit on their first day, with an average streak of 4.4 consecutive limit-up days. ",
            "The resulting mean underpricing is 67.8%\u2014substantially higher than the 6.6% average first-day return that a naive single-day calculation would suggest.",
          ]),

          para([
            "We address five research questions that together form the narrative arc of our thesis title\u2014\u2018IPO Fever\u2019 and \u2018the Cost of the Crowd\u2019:",
          ]),

          para([
            bold("RQ1:"),
            " What is the magnitude and duration of IPO underpricing on Borsa Istanbul under the daily price limit regime (2020\u20132025), and how does the consecutive limit-up mechanism affect the measured level of initial returns? (Durukan, 2002; Ljungqvist, 2007; Wang et al., 2018)",
          ]),

          para([
            bold("RQ2:"),
            " Which firm-specific characteristics are significant cross-sectional determinants of IPO underpricing on BIST, and are these consistent with the predictions of asymmetric information theories? (Bildik and Yilmaz, 2008; Rock, 1986)",
          ]),

          para([
            bold("RQ3:"),
            " Is there evidence of investor herding behaviour in Borsa Istanbul during the 2020\u20132025 period, and does herding intensity differ between bull and bear market sub-periods? (Chang, Cheng, and Khorana, 2000; Durukan, Ozsu, and Ergun, 2017)",
          ]),

          para([
            bold("RQ4:"),
            " What is the stock price impact of SPK (Capital Markets Board of Turkey) market manipulation penalties on sanctioned firms? (MacKinlay, 1997; Ferris and Pritchard, 1994)",
          ]),

          para([
            bold("RQ5:"),
            " Does the post-IPO price pattern on BIST support the overreaction hypothesis, and can a contrarian investment strategy generate abnormal returns after the limit-up streak ends? (De Bondt and Thaler, 1985; Ritter, 1991; Kiymaz, 2000)",
          ]),

          para([
            "Our main findings are as follows. ",
            "First, BIST IPOs are significantly underpriced (t = 10.82, p < 0.001), with mean underpricing of 67.8% and median of 33.1% when measured using the consecutive limit-up methodology (RQ1). ",
            "Second, cross-sectional OLS regression reveals that price-to-book ratio is the only significant determinant of underpricing (p = 0.040), while P/E ratio and market capitalisation are not significant, suggesting that underpricing is primarily behavioural rather than fundamental (RQ2). ",
            "Third, CSAD herding analysis finds no statistically significant herding at the aggregate market level (\u03B32 = 0.645, p = 0.666), though this must be qualified by the low statistical power of the test (RQ3). ",
            "Fourth, event study analysis of 47 SPK manipulation penalties reveals a cumulative average abnormal return (CAAR) of \u22127.4% over the [\u221230, +30] event window, confirming that regulatory enforcement carries real price consequences (RQ4). ",
            "Fifth, a short-term contrarian sell strategy earns statistically significant returns (\u22123.3% at 5 days, t = \u22122.34, p = 0.021), consistent with post-IPO overreaction and mean reversion (RQ5). ",
            "Finally, inflation substantially erodes nominal IPO returns: the median 365-day real return falls to approximately half its nominal value.",
          ]),

          para([
            "The remainder of this paper is organised as follows. ",
            "Section 2 reviews the prior literature on IPO underpricing, herding, and market manipulation. ",
            "Section 3 describes our data and methodology. ",
            "Section 4 presents and discusses the results, including IPO underpricing (4.1), the contrarian strategy (4.2), SPK event study (4.3), CSAD herding analysis (4.4), cross-sectional determinants (4.5), and inflation effects (4.6). ",
            "Section 5 concludes with policy implications.",
          ]),

          // ═══════ 2. LITERATURE REVIEW ═══════
          heading1("2. Prior Literature on IPO Underpricing and Investor Behaviour"),

          heading2("2.1 IPO Underpricing: Theory and Turkish Evidence"),

          para([
            "IPO underpricing\u2014the phenomenon where shares are offered below their first-trading-day market value\u2014is one of the most robust findings in empirical finance. ",
            "Rock (1986) provides the foundational theoretical explanation through the Winner\u2019s Curse model: uninformed investors receive disproportionate allocations in overpriced IPOs because informed investors crowd into underpriced ones, so issuers must underprice to compensate uninformed investors for this adverse selection. ",
            "Benveniste and Spindt (1989) extend this framework through the bookbuilding model, arguing that underpricing serves as compensation for informed investors who truthfully reveal their private information during the pricing process. ",
            "Ritter (1991) documents the long-run underperformance of IPOs, suggesting that initial underpricing reflects temporary overvaluation driven by investor sentiment rather than a permanent equilibrium.",
          ]),

          para([
            "For the Turkish market, Durukan (2002) provides the foundational study, analysing 173 IPOs on the Istanbul Stock Exchange during 1990\u20131997. ",
            "She reports average underpricing of 14.16% using simple first-day returns and finds support for the Winner\u2019s Curse hypothesis. ",
            "Critically, Durukan (2002) uses single-day returns because the price limit regime in the 1990s was substantially different from today\u2019s \u00b110% rule, and short-squeeze dynamics were less prevalent. ",
            "Durukan and Onder (2006) subsequently examine the relationship between ownership structure and underpricing, finding a weak association. ",
            "Kiymaz (2000) extends the Turkish evidence by examining 163 IPOs on the Istanbul Stock Exchange during 1990\u20131996, finding average underpricing of 13.1% and identifying firm size, market conditions, and institutional ownership as significant determinants. ",
            "More recently, Ilbasmi (2023) analyses IPO underpricing during the COVID-19 pandemic on Borsa Istanbul and explicitly acknowledges the price-limit problem by employing multi-day cumulative abnormal returns (CARs), representing the closest methodological precedent to our consecutive limit-up approach. ",
            "Bildik and Yilmaz (2008) contribute to the Turkish IPO literature by examining the role of underwriter discretion in the pricing process, finding a median 21% underwriter discount. ",
            "Most recently, Tanyeri, Ozturkkal, and Tirtiroglu (2021) study 633 BIST IPOs during 1990\u20132020, reporting average underpricing of approximately 5% for the 2010\u20132020 sub-period and explicitly adjusting for binding price limits using one-week or one-month returns when daily limits are reached. ",
            "Our study extends this work to the 2020\u20132025 period, which experienced dramatically different market conditions and substantially higher underpricing.",
          ]),

          para([
            "Ljungqvist (2007) provides a comprehensive survey classifying underpricing theories into four categories: asymmetric information, institutional, ownership and control, and behavioural. ",
            "Our study extends this literature in three ways: (i) we cover the most recent period (2020\u20132025) with 209 IPOs, (ii) we develop the consecutive limit-up methodology to fully capture underpricing in a \u00b110% price-limit environment, and (iii) we use split-adjusted prices from Yahoo Finance with three-source verification (halkarz.com, KAP, SPK) to ensure data accuracy.",
          ]),

          heading2("2.2 Herding Behaviour in Emerging Markets"),

          para([
            "Herding occurs when investors suppress their own private information and follow the actions of others, leading to correlated trading patterns and potential mispricing. ",
            "Chang, Cheng, and Khorana (2000; hereafter CCK) introduce the Cross-Sectional Absolute Deviation (CSAD) methodology, which detects herding through a non-linear relationship between market-wide return dispersion and the absolute market return. ",
            "Under rational asset pricing, CSAD increases linearly with market stress; herding causes CSAD to increase at a decreasing rate (or decrease) during extreme market movements, captured by a significantly negative \u03B32 coefficient.",
          ]),

          para([
            "Durukan, Ozsu, and Ergun (2017) examine herding in BIST using a state-space model combined with the LSV herding measure, covering the 2006\u20132015 period. ",
            "They find evidence of herding and note that foreign investors herd less during financial crises. ",
            "Cimen and Ergun (2019) investigate herding in Turkish IPO markets specifically, using the CSSD (Cross-Sectional Standard Deviation) measure. ",
            "They find no evidence of herding, though CSSD is a less powerful test than CSAD. ",
            "Cagli (2022) employs the Patterson-Sharma runs-based herding measure for cryptocurrency markets, demonstrating Fourier-augmented Granger causality between herding intensity and returns, but does not apply CSAD-based methods to BIST equities.",
          ]),

          para([
            "Our study is the first to apply the CCK CSAD methodology specifically to BIST stock returns during IPO-intensive periods. ",
            "We note the important limitation identified by Zhang (2024), who demonstrates that the statistical power of CSAD-based herding tests averages only 59.37%, which means that failure to reject the null hypothesis of no herding should be interpreted with caution.",
          ]),

          heading2("2.3 IPO Aftermarket Performance and the Overreaction Hypothesis"),

          para([
            "De Bondt and Thaler (1985) propose the overreaction hypothesis: investors systematically overreact to dramatic news, causing past losers to outperform past winners over subsequent periods. ",
            "Applied to IPOs, this implies that initial enthusiasm inflates prices above fundamental value, followed by a correction as the market reassesses. ",
            "Ritter (1991) documents significant long-run underperformance of IPOs over three years, consistent with initial overvaluation. ",
            "Loughran and Ritter (1995) further show that IPO investors earned only 5% annually over five years post-issue, dramatically below matched-firm benchmarks.",
          ]),

          para([
            "In the Turkish context, the overreaction hypothesis is particularly relevant because the consecutive limit-up mechanism may amplify initial price distortions. ",
            "When demand exceeds the daily limit for multiple consecutive days, the stock cannot reach equilibrium, and the artificially constrained supply may create the illusion of extreme scarcity. ",
            "Once the limit-up streak ends and the stock trades freely, mean reversion should be expected if the streak prices exceeded fundamental value. ",
            "Kiymaz (2000) examines aftermarket performance of ISE IPOs and finds evidence consistent with this pattern.",
          ]),

          heading2("2.4 Market Manipulation and Regulatory Enforcement"),

          para([
            "Turkey\u2019s Capital Markets Board (Sermaye Piyasasi Kurulu, SPK) actively prosecutes market manipulation, primarily in two categories: transaction-based manipulation (islem bazli) involving coordinated trading to create artificial prices, and information-based manipulation (bilgi bazli) involving insider trading. ",
            "MacKinlay (1997) provides the standard event study methodology for measuring abnormal returns around corporate events, which we adapt to measure the stock price impact of SPK penalty announcements. ",
            "To our knowledge, no prior study has systematically examined the price effects of SPK manipulation penalties using a formal event study framework.",
          ]),

          // Summary table of prior literature
          tableCaption("Table 1. Summary of Key Prior Studies"),

          makeTable(
            ["Study", "Sample", "Method", "Key Finding"],
            [
              ["Durukan (2002)", "173 IPOs, ISE, 1990\u20131997", "First-day returns", "Mean underpricing 14.16%"],
              ["Durukan & Onder (2006)", "ISE IPOs", "OLS regression", "Weak ownership\u2013underpricing link"],
              ["Durukan et al. (2017)", "BIST, 2006\u20132015", "State-space + LSV", "Herding exists; less by foreigners in crises"],
              ["Cimen & Ergun (2019)", "BIST IPOs", "CSSD", "No herding detected"],
              ["Cagli (2022)", "Crypto markets", "Patterson-Sharma", "Herding in cryptocurrencies"],
              ["Ilbasmi (2023)", "BIST IPOs, COVID", "Multi-day CAR", "Acknowledged price-limit problem"],
              ["Zhang (2024)", "Simulation", "Monte Carlo", "CSAD test power avg. 59.37%"],
              ["De Bondt & Thaler (1985)", "NYSE", "Portfolio formation", "Overreaction: past losers outperform"],
              ["Ritter (1991)", "1,526 IPOs, NYSE", "BHR", "Long-run IPO underperformance"],
              ["Kiymaz (2000)", "163 IPOs, ISE, 1990\u20131996", "First-day returns + aftermarket", "13.1% underpricing"],
              ["Ljungqvist (2007)", "Survey", "Literature review", "Four theories of underpricing"],
              ["This study", "209 IPOs, BIST, 2020\u20132025", "Consecutive limit-up + CSAD + Event study + OLS", "67.8% underpricing, no herding, \u20137.4% CAR"],
            ],
            [1800, 2000, 2200, 3026]
          ),

          tableNote("Note: ISE = Istanbul Stock Exchange (now Borsa Istanbul). CSSD = Cross-Sectional Standard Deviation. CSAD = Cross-Sectional Absolute Deviation."),

          // ═══════ 3. DATA AND METHODOLOGY ═══════
          heading1("3. Data and Methodology"),

          heading2("3.1 Data Sources and Sample"),

          para([
            "Our dataset consists of 209 IPOs on Borsa Istanbul between February 2020 and December 2025. ",
            "We collect IPO data from three independent sources to ensure accuracy: (i) halkarz.com, which provides offer prices, dates, and oversubscription ratios; (ii) KAP (Kamuyu Aydinlatma Platformu, the Public Disclosure Platform), which provides official company announcements; and (iii) SPK bulletins, which provide regulatory confirmations. ",
            "Daily closing prices are obtained from Yahoo Finance. ",
            "All prices are adjusted for stock splits and other corporate actions using Yahoo Finance\u2019s adjusted close series.",
          ]),

          para([
            "Table 2 reports the distribution of IPOs by year. ",
            "The number of IPOs increased dramatically from 8 in 2020 to a peak of 56 in 2023, driven by retail investor enthusiasm, favourable market conditions, and the Turkish economy\u2019s inflationary environment which pushed investors toward equities as an inflation hedge.",
          ]),

          tableCaption("Table 2. IPO Distribution by Year"),

          makeTable(
            ["Year", "N IPOs", "Mean Underpricing (%)", "Median Underpricing (%)", "Avg Limit-Up Days"],
            [
              ["2020", "8", "105.2", "114.4", "7.1"],
              ["2021", "53", "41.5", "20.9", "2.7"],
              ["2022", "39", "54.7", "32.9", "3.7"],
              ["2023", "56", "104.2", "76.8", "6.6"],
              ["2024", "33", "61.6", "33.1", "4.2"],
              ["2025", "20", "51.8", "20.7", "3.2"],
              ["Total", "209", "67.8", "33.1", "4.4"],
            ],
            [1200, 1200, 2200, 2200, 2226]
          ),

          tableNote("Note: Underpricing is measured using the consecutive limit-up methodology (cumulative return from IPO-day close to first free-trading day close). Limit-up days = number of consecutive trading days closing at the +10% daily upper price limit."),

          para([
            "For the CSAD herding analysis, we use daily returns for all BIST-listed equities from January 2020 to December 2025, comprising 1,496 trading days. ",
            "For the event study on manipulation penalties, we collect 47 verified SPK penalty announcements from SPK weekly bulletins over the 2020\u20132025 period, involving 217 individuals and total fines of approximately 1.13 billion TL.",
          ]),

          heading2("3.2 Consecutive Limit-Up Methodology"),

          para([
            "BIST imposes a \u00b110% daily price limit on equity prices. ",
            "When a newly listed IPO stock hits the upper price limit on its first trading day, the observed first-day return is mechanically capped at approximately 10%, regardless of the true equilibrium price. ",
            "For 67% of our sample IPOs, the stock hits the upper limit on day one and continues to do so for multiple consecutive days\u2014a phenomenon known in Turkish market parlance as ",
            italic("tavan serisi"),
            " (lit. \u2018ceiling series\u2019). ",
            "Following the methodology described by Loughran, Ritter, and Rydqvist (1994, updated 2026) for markets with binding daily price limits, we refer to this pattern as ",
            bold("consecutive daily limit-up hits"),
            " and define the ",
            bold("limit-up streak length"),
            " as the number of consecutive trading days on which the stock closes at the upper price limit. ",
            "Measuring underpricing using only the first-day return would severely understate the actual return accruing to IPO investors.",
          ]),

          para([
            "We define the consecutive limit-up return as:",
          ]),

          equation("Underpricing_i = (P_free,i / P_close,i) \u2212 1"),

          para([
            "where P_free,i is the first unconstrained closing price\u2014the closing price on the first day that stock i does not hit the daily price limit (Loughran, Ritter, and Rydqvist, 1994)\u2014and P_close,i is the Yahoo Finance split-adjusted closing price on the IPO date. ",
            "We use the IPO-day adjusted close (rather than the nominal offer price) as the denominator because Yahoo Finance applies split adjustments retroactively, making the adjusted close the appropriate base for computing returns consistent with subsequent price data. ",
            "For IPOs that do not hit the upper price limit on their first trading day (33% of the sample), the limit-up return is zero by construction, since P_free equals P_close.",
          ]),

          para([
            "Multi-period buy-and-hold returns are computed as:",
          ]),

          equation("Return_d,i = (P_d,i / P_close,i) \u2212 1"),

          para([
            "where P_d,i is the adjusted closing price d calendar days after the IPO. ",
            "Buy-and-hold abnormal returns (BHARs) are computed by subtracting the BIST-100 index return over the same period.",
          ]),

          heading2("3.3 CSAD Herding Methodology"),

          para([
            "Following Chang, Cheng, and Khorana (2000), we estimate the cross-sectional absolute deviation (CSAD) of individual stock returns from the equally-weighted market return as:",
          ]),

          equation("CSAD_t = (1/N) \u03A3|R_i,t \u2212 R_m,t|"),

          para([
            "We then regress CSAD on the absolute market return and its square:",
          ]),

          equation("CSAD_t = \u03B1 + \u03B31|R_m,t| + \u03B32 R_m,t\u00B2 + \u03B5_t"),

          para([
            "Under rational expectations, both \u03B31 and \u03B32 should be positive (or at least non-negative), reflecting that return dispersion increases with market stress. ",
            "A significantly negative \u03B32 indicates herding: during extreme market movements, investors suppress heterogeneous beliefs and follow the crowd, causing individual returns to cluster around the market return. ",
            "Standard errors are computed using the Newey-West (1987) heteroskedasticity and autocorrelation consistent (HAC) estimator with automatic lag selection. ",
            "We additionally estimate the model separately for bull-market and bear-market days to test for asymmetric herding, and we compute 60-day rolling window estimates to examine time variation.",
          ]),

          heading2("3.4 Event Study Methodology"),

          para([
            "Following MacKinlay (1997), we estimate the market model for each stock subject to an SPK penalty:",
          ]),

          equation("R_i,t = \u03B1_i + \u03B2_i R_m,t + \u03B5_i,t"),

          para([
            "The estimation window spans [\u2212250, \u221231] trading days relative to the SPK penalty announcement date (day 0). ",
            "Abnormal returns in the event window [\u221230, +30] are:",
          ]),

          equation("AR_i,t = R_i,t \u2212 (\u03B1\u0302_i + \u03B2\u0302_i R_m,t)"),

          para([
            "Cumulative abnormal returns (CARs) are summed across the event window, and the cross-sectional average (CAAR) is tested for significance using the Boehmer, Musumeci, and Poulsen (1991) standardised cross-sectional test, where each CAR_i is standardised by its individual standard deviation \u03C3_i from the estimation window. ",
            "We additionally conduct a non-parametric sign test.",
          ]),

          heading2("3.5 Summary Statistics"),

          tableCaption("Table 3. Descriptive Statistics of Key Variables"),

          makeTable(
            ["Variable", "N", "Mean", "Median", "Std. Dev.", "Min", "Max"],
            [
              ["Underpricing (consecutive limit-up)", "204", "0.678", "0.331", "0.895", "0.000", "6.300"],
              ["First-day return", "204", "0.066", "0.100", "0.052", "\u22120.200", "0.100"],
              ["Limit-up streak (days)", "209", "4.41", "3.0", "4.67", "0", "21"],
              ["Return d30", "204", "0.510", "0.249", "0.917", "\u22120.783", "8.500"],
              ["Return d365", "198", "1.645", "0.748", "2.358", "\u22120.920", "18.200"],
              ["BHAR d365", "198", "0.638", "0.074", "2.142", "\u22121.200", "17.500"],
              ["CSAD (daily)", "1,496", "0.0155", "0.0141", "0.0058", "0.0028", "0.0855"],
              ["SPK fine (million TL)", "47", "23.99", "6.80", "89.12", "1.85", "615.00"],
            ],
            [2400, 800, 1000, 1000, 1000, 1200, 1626]
          ),

          tableNote("Note: Underpricing, returns, and BHARs are expressed as decimals (0.678 = 67.8%). CSAD is computed from daily cross-sectional return dispersion."),

          // ═══════ 4. RESULTS AND DISCUSSION ═══════
          heading1("4. Results and Discussion"),

          heading2("4.1 IPO Underpricing"),

          para([
            "Table 2 reports underpricing by year. ",
            "Across the full sample, mean underpricing (measured using the consecutive limit-up methodology) is 67.8% (t = 10.82, p < 0.001) and median underpricing is 33.1%. ",
            "The Wilcoxon signed-rank test confirms the result non-parametrically (p < 0.001). ",
            "The distribution is highly right-skewed: while the median is 33.1%, the mean is pulled upward by a small number of IPOs with extremely long limit-up streaks (up to 21 consecutive days, corresponding to underpricing of 630%).",
          ]),

          para([
            "These figures are substantially higher than those reported in prior Turkish IPO studies. ",
            "Durukan (2002) reports mean underpricing of 14.16% for the 1990\u20131997 period using single-day returns, while Kiymaz (2000) reports 13.1% for 1990\u20131996. ",
            "The difference is not simply a matter of methodology: the 2020\u20132025 period features significantly more retail investor participation, higher inflation driving equity demand, and a generally bull market for BIST. ",
            "Our 67.8% mean underpricing is economically large, suggesting substantial wealth transfer from issuers to initial investors.",
          ]),

          para([
            "To test whether underpricing is driven by fundamental valuation, we examine the relationship between P/E (price-to-earnings) ratios and underpricing. ",
            "For the 112 IPOs with valid P/E data, the median P/E is 18.2 and the mean is 40.3. ",
            "The Spearman rank correlation between P/E and underpricing is 0.147 (p = 0.122), which is not statistically significant. ",
            "This suggests that underpricing is a behavioural phenomenon\u2014driven by investor enthusiasm and the limit-up mechanism\u2014rather than a rational response to fundamental undervaluation.",
          ]),

          heading2("4.2 Contrarian Strategy: Post-Limit-Up Returns"),

          para([
            "Table 4 reports post-limit-up returns for the 137 IPOs that experienced at least one consecutive limit-up day. ",
            "We measure returns from the first free-trading day (the day after the limit-up streak ends) to examine whether the initial price surge reverses, consistent with the overreaction hypothesis of De Bondt and Thaler (1985).",
          ]),

          tableCaption("Table 4. Post-Limit-Up Contrarian Strategy Returns"),

          makeTable(
            ["Horizon", "N", "Mean Return (%)", "Median Return (%)", "t-statistic", "p-value", "% Positive", "Signal"],
            [
              ["5 days", "137", "\u22123.3", "\u22126.3", "\u22122.335", "0.021**", "32.8%", "SELL"],
              ["10 days", "137", "\u22123.4", "\u22129.3", "\u22121.606", "0.111", "30.7%", "SELL"],
              ["30 days", "137", "+4.8", "\u221210.1", "1.092", "0.277", "37.2%", "HOLD"],
              ["60 days", "137", "+16.6", "\u22123.4", "2.386", "0.018**", "43.1%", "HOLD"],
              ["90 days", "136", "+21.7", "\u22124.2", "2.783", "0.006***", "46.3%", "HOLD"],
              ["180 days", "131", "+46.0", "+1.1", "4.056", "0.000***", "51.9%", "HOLD"],
              ["365 days", "131", "+72.6", "+26.2", "5.022", "0.000***", "61.8%", "HOLD"],
            ],
            [1100, 600, 1400, 1500, 1100, 1100, 1000, 1226]
          ),

          tableNote("Note: Returns measured from first free-trading day after the limit-up streak ends. ***, **, * denote significance at 1%, 5%, and 10% levels, respectively."),

          para([
            "The results reveal a clear pattern consistent with the overreaction hypothesis. ",
            "Short-term mean reversion is statistically significant at the 5-day horizon (\u22123.3%, p = 0.021), with only 32.8% of stocks posting positive returns. ",
            "This is consistent with the hypothesis that the consecutive limit-up mechanism artificially inflates prices above equilibrium, creating an overreaction that corrects once free trading resumes. ",
            "A contrarian sell strategy at the end of the limit-up streak would have been profitable in the short term.",
          ]),

          para([
            "However, long-term returns are strongly positive, consistent with Ritter\u2019s (1991) observation that IPO aftermarket performance depends critically on the holding period. ",
            "By 365 days, the mean return is +72.6% (p < 0.001) with 61.8% of IPOs in positive territory. ",
            "This presents a nuanced picture: while there is genuine short-term overpricing at the end of the limit-up streak, the broader bull market and inflationary environment of 2020\u20132025 lifted most stocks over longer horizons. ",
            "Importantly, the median returns remain negative until 180 days, indicating that the mean is driven by a subset of high-performing IPOs, while the typical stock underperforms in the medium term.",
          ]),

          heading2("4.3 SPK Manipulation Penalties: Event Study Results"),

          para([
            "We analyse 47 SPK manipulation penalty cases involving unique stocks during the 2020\u20132025 period. ",
            "Two stocks (ARMDA and KONKA) were excluded from the price analysis due to delisting, yielding 45 stocks with available price data for the event study. ",
            "Table 5 presents the event study results.",
          ]),

          tableCaption("Table 5. Event Study Results: SPK Manipulation Penalties"),

          makeTable(
            ["Metric", "Value"],
            [
              ["Stocks analysed", "48"],
              ["Mean CAR [\u221230, +30]", "\u22127.4%"],
              ["Significantly negative CARs (5% level)", "40 / 48 (83.3%)"],
              ["Negative CARs", "32 / 48 (66.7%)"],
              ["CAAR at Day 0", "\u22124.8%"],
              ["CAAR at Day +30", "\u22127.4%"],
              ["Pre-event CAAR [\u221230, \u22121]", "\u22125.4%"],
              ["Post-event CAAR [+1, +30]", "\u22122.0%"],
            ],
            [5000, 4026]
          ),

          tableNote("Note: CAR = Cumulative Abnormal Return. CAAR = Cumulative Average Abnormal Return. Event window: [\u221230, +30] trading days around SPK penalty announcement date."),

          para([
            "The CAAR pattern reveals that manipulated stocks experience significant negative abnormal returns around SPK penalty announcements. ",
            "The pre-event CAAR of \u22125.4% suggests that the market anticipates enforcement action (or that the manipulation\u2019s unwinding begins before the formal announcement). ",
            "The post-event CAAR continues to decline, reaching \u22127.4% by day +30, indicating that the penalty announcement carries additional negative information.",
          ]),

          para([
            "Transaction-based manipulation (islem bazli) accounts for 37 of 47 cases (78.7%), reflecting SPK\u2019s focus on coordinated trading schemes. ",
            "Information-based manipulation (bilgi bazli, insider trading) accounts for 9 cases (19.1%). ",
            "The average investigation period is 307 days, and 77% of cases result in trading bans. ",
            "The largest single penalty in our sample is 615 million TL (Besiktas Futbol Yatirimlari, BJKAS, 2023), where the stock price was reportedly inflated from approximately 4 TL to 90 TL through coordinated transactions.",
          ]),

          heading2("4.4 CSAD Herding Analysis"),

          para([
            "Table 6 reports the CSAD regression results.",
          ]),

          tableCaption("Table 6. CSAD Herding Regression Results (CCK 2000)"),

          makeTable(
            ["Parameter", "Estimate", "Newey-West SE", "t-statistic", "p-value"],
            [
              ["\u03B1 (intercept)", "0.0139", "0.0005", "27.45", "<0.001"],
              ["\u03B31 (|R_m|)", "0.1064", "0.0658", "1.617", "0.106"],
              ["\u03B32 (R_m\u00B2)", "0.6450", "1.4931", "0.432", "0.666"],
              ["R\u00B2", "0.096", "", "", ""],
              ["N (trading days)", "1,496", "", "", ""],
            ],
            [2000, 1600, 1800, 1600, 2026]
          ),

          tableNote("Note: Dependent variable is CSAD. Standard errors computed using Newey-West HAC estimator. Herding requires \u03B32 < 0 (significantly negative)."),

          para([
            "The key parameter \u03B32 is positive (0.645) and statistically insignificant (p = 0.666), indicating ",
            bold("no evidence of herding"),
            " at the aggregate market level during the 2020\u20132025 period. ",
            "The positive coefficient suggests that, contrary to the herding hypothesis, return dispersion increases more than proportionally during extreme market movements, consistent with rational heterogeneous expectations.",
          ]),

          para([
            "We further examine bull versus bear market asymmetry. ",
            "In bull markets (positive market return days, N = 847), \u03B32 = 4.01, and in bear markets (negative market return days, N = 647), \u03B32 = \u22120.78. ",
            "Neither coefficient is statistically significant. ",
            "The slightly negative bear-market \u03B32 hints at possible herding during downturns, but the evidence is not strong enough to reject the null hypothesis.",
          ]),

          para([
            "This null result should be interpreted with the caveat from Zhang (2024), who demonstrates through Monte Carlo simulation that the power of the CSAD-based t-test averages only 59.37% and can drop to 37.62%. ",
            "In other words, even if herding exists at moderate intensity, the CSAD test may fail to detect it approximately 40% of the time. ",
            "Our finding of no herding is therefore consistent with either the genuine absence of herding or insufficient statistical power to detect it.",
          ]),

          heading2("4.5 Cross-Sectional Determinants of Underpricing"),

          para([
            "To identify the firm-specific and market-level characteristics that drive IPO underpricing on BIST, we estimate cross-sectional OLS regressions with heteroskedasticity-robust (HC3) standard errors. ",
            "Following the determinants framework of Durukan (2002), Kiymaz (2000), and Ljungqvist (2007), we specify four progressive models. ",
            "The dependent variable is underpricing as measured by the consecutive limit-up methodology.",
          ]),

          para([
            "Model 1 (Basic, N = 204) includes only the log offer price, year dummies (2020 as base), and a hot-market dummy (months with five or more IPOs). ",
            "This model yields an R-squared of 9.7% and a significant F-statistic (p < 0.001), driven primarily by the year effects. ",
            "Model 2 (Fundamentals, N = 112) adds log market capitalisation, trailing P/E ratio, and price-to-book ratio. ",
            "The sample shrinks to 112 IPOs with complete fundamental data, and R-squared increases to 19.3%.",
          ]),

          para([
            "The key finding across specifications is that ",
            bold("price-to-book ratio is the only consistently significant determinant"),
            " of underpricing. ",
            "In Model 4 (log-level robustness check using ln(1 + underpricing) as the dependent variable), the P/B coefficient is 0.017 (t = 2.05, p = 0.040), indicating that higher-valued firms experience greater underpricing. ",
            "This is consistent with the behavioural interpretation: overvalued firms (high P/B) attract more speculative demand, extending the limit-up streak. ",
            "In contrast, the P/E ratio is not significant in any specification (p > 0.50), suggesting that earnings-based valuation does not systematically predict underpricing.",
          ]),

          para([
            "Log market capitalisation shows a marginally significant negative coefficient in Model 3 (p = 0.072), consistent with the asymmetric information hypothesis of Rock (1986): larger firms have less information asymmetry, leading to lower underpricing. ",
            "Year dummies and the hot-market indicator are not individually significant, though they are jointly significant in Model 1. ",
            "Sector dummies (Industrials, Consumer Defensive, Basic Materials, Utilities) add limited explanatory power (adjusted R-squared increases only marginally from 11.3% to 10.5% after penalising for lost degrees of freedom).",
          ]),

          tableCaption("Table 7. Cross-Sectional OLS Regression: Determinants of IPO Underpricing"),

          makeTable(
            ["Variable", "Model 1 (Basic)", "Model 2 (Fund.)", "Model 4 (Log-level)"],
            [
              ["Constant", "1.251*** (4.82)", "4.069** (2.08)", "2.007** (2.23)"],
              ["ln(Offer Price)", "\u22120.113 (\u22121.38)", "\u22120.086 (\u22121.29)", "\u22120.055 (\u22121.49)"],
              ["ln(Market Cap)", "\u2014", "\u22120.148 (\u22121.67)", "\u22120.067 (\u22121.67)"],
              ["P/E Ratio", "\u2014", "0.001 (0.39)", "0.000 (0.65)"],
              ["P/B Ratio", "\u2014", "0.035 (1.53)", "0.017** (2.05)"],
              ["Hot Market", "\u22120.110 (\u22120.71)", "0.164 (0.94)", "0.063 (0.75)"],
              ["Year Dummies", "Included", "Included", "Included"],
              ["N", "204", "112", "112"],
              ["R-squared", "0.097", "0.193", "0.214"],
              ["Adj. R-squared", "0.065", "0.113", "0.137"],
              ["F-statistic (p)", "4.66 (<0.001)", "1.54 (0.137)", "2.68 (0.006)"],
            ],
            [2200, 2400, 2400, 2026]
          ),

          tableNote("Note: t-statistics in parentheses. HC3 robust standard errors. ***, **, * denote significance at 1%, 5%, and 10% levels. Dependent variable: underpricing (Models 1\u20132) or ln(1 + underpricing) (Model 4)."),

          para([
            "The low R-squared values (9.7\u201321.4%) indicate that observable firm characteristics explain a modest fraction of underpricing variation. ",
            "This finding is consistent with the behavioural view articulated by Ljungqvist (2007): in markets with high retail participation and constrained price discovery (due to daily limits), investor sentiment and speculative demand\u2014rather than fundamentals\u2014drive the bulk of underpricing. ",
            "The insignificance of P/E and the significance of P/B suggest that investors focus on balance sheet multiples rather than earnings when evaluating newly listed firms, possibly because trailing earnings are less informative for young growth companies.",
          ]),

          heading2("4.6 Inflation and Real Returns"),

          para([
            "Turkey\u2019s extreme inflationary environment during 2020\u20132025 substantially affects the interpretation of IPO returns. ",
            "Cumulative consumer price inflation exceeded 500% over this period (TUFE index rising from approximately 500 to 3,055, base 2003 = 100), with year-over-year inflation peaking at 85.5% in October 2022. ",
            "The BIST-100 index rose approximately 688% in nominal terms, but only approximately 40% in CPI-adjusted real terms.",
          ]),

          para([
            "For IPO investors, the inflation adjustment is meaningful. ",
            "Using the Fisher equation (Real Return = (1 + Nominal Return) / (1 + Inflation) \u2212 1), the median 365-day nominal IPO return of 74.8% translates to a substantially lower real return once period-specific inflation is applied. ",
            "An investor who earned 100% nominal return over 12 months in 2022 (when inflation averaged approximately 70% year-over-year) would have earned only approximately 18% in real terms. ",
            "This highlights the importance of distinguishing between nominal and real wealth creation in high-inflation emerging markets.",
          ]),

          // ═══════ 5. CONCLUSIONS ═══════
          heading1("5. Conclusions and Policy Implications"),

          para([
            "This study provides a comprehensive analysis of IPO market dynamics in Borsa Istanbul during the 2020\u20132025 period, a time of unprecedented retail investor participation, extreme inflation, and active regulatory enforcement. ",
            "Our five research questions are addressed as follows.",
          ]),

          para([
            bold("RQ1 (Underpricing):"),
            " BIST IPOs are significantly underpriced when measured using the consecutive limit-up methodology (mean 67.8%, median 33.1%, t = 10.82, p < 0.001). ",
            "The standard first-day return measure understates true underpricing by a factor of ten in this market. ",
            "This has direct implications for issuers, who leave substantial money on the table, and for regulators, who should consider whether the \u00b110% daily price limit, while intended to reduce volatility, actually exacerbates the underpricing phenomenon by creating artificial scarcity through the limit-up mechanism.",
          ]),

          para([
            bold("RQ2 (Cross-Sectional Determinants):"),
            " Cross-sectional OLS regression reveals that price-to-book ratio is the only significant determinant of underpricing (p = 0.040), while P/E ratio, market capitalisation, and hot-market conditions are not significant. ",
            "Observable fundamentals explain only 10\u201321% of underpricing variation, suggesting that the phenomenon is primarily behavioural\u2014driven by investor sentiment and the limit-up mechanism\u2014rather than a rational response to firm characteristics. ",
            "This is consistent with the behavioural theories surveyed by Ljungqvist (2007).",
          ]),

          para([
            bold("RQ3 (Herding):"),
            " We find no evidence of aggregate market-level herding using the CSAD methodology (\u03B32 = 0.645, p = 0.666). ",
            "This null result must be qualified by the low statistical power of the test (Zhang, 2024, reports average power of 59.37%). ",
            "Future research could employ alternative herding measures (such as the Patterson-Sharma runs test, as in Cagli, 2022, or the LSV measure, as in Durukan et al., 2017) to provide robustness checks.",
          ]),

          para([
            bold("RQ4 (SPK Manipulation Penalties):"),
            " SPK manipulation penalties are associated with significant negative cumulative abnormal returns (\u22127.4% CAAR over the [\u221230, +30] event window), confirming that regulatory enforcement carries real economic consequences. ",
            "The pre-event CAAR pattern suggests partial information leakage or anticipation of enforcement. ",
            "The high proportion of transaction-based manipulation cases (78.7%) and the large average investigation period (307 days) highlight both the prevalence of coordinated trading schemes and the challenges of timely detection.",
          ]),

          para([
            bold("RQ5 (Contrarian Strategy):"),
            " The contrarian analysis reveals clear short-term mean reversion following the limit-up streak (\u22123.3% at 5 days, p = 0.021), consistent with the overreaction hypothesis of De Bondt and Thaler (1985). ",
            "Investors who buy at the end of the limit-up streak face poor short-term prospects, though long-term returns are positive in absolute terms (+72.6% at 365 days). ",
            "This finding has practical value for retail investors, who should be cautioned against purchasing IPO stocks in the secondary market immediately after the limit-up streak ends.",
          ]),

          para([
            "Several limitations should be noted. ",
            "The consecutive limit-up methodology assumes that the first free-trading day reflects equilibrium pricing, which may not hold if speculative momentum persists beyond the limit-up period. ",
            "Our CSAD analysis uses market-wide data rather than IPO-specific stocks, so it captures general herding tendencies rather than IPO-specific herding. ",
            "The SPK event study sample of 47 cases, while representing the near-universe of publicly available penalty announcements, limits the power of cross-sectional tests. ",
            "The cross-sectional regression suffers from a reduced sample (N = 112) for models with fundamentals, due to missing P/E and market capitalisation data for 97 IPOs. ",
            "Finally, the extreme inflationary environment of 2020\u20132025 makes it difficult to disentangle genuine alpha from nominal gains driven by monetary debasement.",
          ]),

          para([
            "Future research directions include: (i) applying the consecutive limit-up methodology to other price-limit markets (e.g., China, Korea, Thailand) to assess generalisability; (ii) examining the relationship between IPO herding and SPK manipulation cases at the firm level; (iii) developing more powerful tests for herding that address the limitations identified by Zhang (2024); (iv) investigating the role of social media and online trading platforms in amplifying IPO demand during the Turkish IPO boom; and (v) employing alternative underpricing measures that incorporate the offer-to-close spread alongside the limit-up premium.",
          ]),

          // ═══════ REFERENCES ═══════
          new Paragraph({ children: [new PageBreak()] }),
          heading1("References"),

          paraNoIndent([
            "Benveniste, L.M. and Spindt, P.A. (1989) \u2018How investment bankers determine the offer price and allocation of new issues\u2019, ",
            italic("Journal of Financial Economics"),
            ", 24(2), pp. 343\u2013361.",
          ]),

          paraNoIndent([
            "Bildik, R. and Yilmaz, M.K. (2008) \u2018The market performance of initial public offerings in the Istanbul Stock Exchange\u2019, ",
            italic("Journal of Business Economics and Management"),
            ", 9(1), pp. 3\u201318.",
          ]),

          paraNoIndent([
            "Boehmer, E., Musumeci, J. and Poulsen, A.B. (1991) \u2018Event-study methodology under conditions of event-induced variance\u2019, ",
            italic("Journal of Financial Economics"),
            ", 30(2), pp. 253\u2013272.",
          ]),

          paraNoIndent([
            "Cagli, E.C. (2022) \u2018Herding in the cryptocurrency market\u2019, ",
            italic("Finance Research Letters"),
            ", 48, 102991.",
          ]),

          paraNoIndent([
            "Cagli, E.C., Ergun, Z.C. and Durukan Sali, M.B. (2020) \u2018The causal linkages between investor sentiment and excess returns on Borsa Istanbul\u2019, ",
            italic("Borsa Istanbul Review"),
            ", 20(3), pp. 214\u2013223.",
          ]),

          paraNoIndent([
            "Chang, E.C., Cheng, J.W. and Khorana, A. (2000) \u2018An examination of herd behavior in equity markets: An international perspective\u2019, ",
            italic("Journal of Banking & Finance"),
            ", 24(10), pp. 1651\u20131679.",
          ]),

          paraNoIndent([
            "Cimen, A. and Ergun, Z.C. (2019) \u2018Herding in Turkish IPO market\u2019, ",
            italic("Izmir Journal of Economics"),
            ", 34(3), pp. 393\u2013406.",
          ]),

          paraNoIndent([
            "De Bondt, W.F.M. and Thaler, R. (1985) \u2018Does the stock market overreact?\u2019, ",
            italic("Journal of Finance"),
            ", 40(3), pp. 793\u2013805.",
          ]),

          paraNoIndent([
            "Durukan, M.B. (2002) \u2018The relationship between IPO returns and factors influencing IPO performance: Case of Istanbul Stock Exchange\u2019, ",
            italic("Managerial Finance"),
            ", 28(2), pp. 18\u201338.",
          ]),

          paraNoIndent([
            "Durukan, M.B. and Onder, Z. (2006) \u2018IPO underpricing and ownership structure: Evidence from the Istanbul Stock Exchange\u2019, in ",
            italic("Initial Public Offerings: An International Perspective"),
            ". Amsterdam: Elsevier, pp. 263\u2013278.",
          ]),

          paraNoIndent([
            "Durukan, M.B., Ozsu, H.H. and Ergun, Z.C. (2017) \u2018Financial crisis and herd behavior: Evidence from the Borsa Istanbul\u2019, in Caliyurt, K.T. (ed.) ",
            italic("Finance and Corporate Governance Conference"),
            ". Amsterdam: Academic Press.",
          ]),

          paraNoIndent([
            "Ergun, Z.C., Cagli, E.C. and Durukan Sali, M.B. (2025) \u2018Sustainability-risk appetite nexus in Borsa Istanbul\u2019, ",
            italic("Risk Management"),
            ", 27, 5. Springer.",
          ]),

          paraNoIndent([
            "Ferris, S.P. and Pritchard, R.E. (1994) \u2018Stock price reactions to securities fraud class-action litigation\u2019, ",
            italic("Journal of Financial and Strategic Decisions"),
            ", 7(2), pp. 53\u201365.",
          ]),

          paraNoIndent([
            "Ilbasmi, S. (2023) \u2018IPO underpricing during the COVID-19 pandemic: Evidence from Borsa Istanbul\u2019, ",
            italic("Borsa Istanbul Review"),
            ", 23(4), pp. 835\u2013847.",
          ]),

          paraNoIndent([
            "Kiymaz, H. (2000) \u2018The initial and aftermarket performance of IPOs in an emerging market: Evidence from Istanbul Stock Exchange\u2019, ",
            italic("Journal of Multinational Financial Management"),
            ", 10(2), pp. 213\u2013227.",
          ]),

          paraNoIndent([
            "Ljungqvist, A. (2007) \u2018IPO underpricing\u2019, in Eckbo, B.E. (ed.) ",
            italic("Handbook of Empirical Corporate Finance"),
            ". Amsterdam: Elsevier, vol. 1, pp. 375\u2013422.",
          ]),

          paraNoIndent([
            "Loughran, T. and Ritter, J.R. (1995) \u2018The new issues puzzle\u2019, ",
            italic("Journal of Finance"),
            ", 50(1), pp. 23\u201351.",
          ]),

          paraNoIndent([
            "Loughran, T., Ritter, J.R. and Rydqvist, K. (1994) \u2018Initial public offerings: International insights\u2019, ",
            italic("Pacific-Basin Finance Journal"),
            ", 2(2\u20133), pp. 165\u2013199. [Updated March 2026].",
          ]),

          paraNoIndent([
            "MacKinlay, A.C. (1997) \u2018Event studies in economics and finance\u2019, ",
            italic("Journal of Economic Literature"),
            ", 35(1), pp. 13\u201339.",
          ]),

          paraNoIndent([
            "Newey, W.K. and West, K.D. (1987) \u2018A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix\u2019, ",
            italic("Econometrica"),
            ", 55(3), pp. 703\u2013708.",
          ]),

          paraNoIndent([
            "Ritter, J.R. (1991) \u2018The long-run performance of initial public offerings\u2019, ",
            italic("Journal of Finance"),
            ", 46(1), pp. 3\u201327.",
          ]),

          paraNoIndent([
            "Rock, K. (1986) \u2018Why new issues are underpriced\u2019, ",
            italic("Journal of Financial Economics"),
            ", 15(1\u20132), pp. 187\u2013212.",
          ]),

          paraNoIndent([
            "Tanyeri, B., Ozturkkal, B. and Tirtiroglu, D. (2021) \u2018Turkish IPO market: 1990\u20132020\u2019, ",
            italic("Borsa Istanbul Review"),
            ", 21(S1), pp. S27\u2013S33.",
          ]),

          paraNoIndent([
            "Wang, Q., Zhao, Z., He, J. and Feng, W. (2018) \u2018Price performance following stock\u2019s IPO in different price limit systems\u2019, ",
            italic("Physica A: Statistical Mechanics and its Applications"),
            ", 490, pp. 953\u2013966.",
          ]),

          paraNoIndent([
            "Xiong, X., Qu, C., He, F. and Zhang, Y.J. (2020) \u2018The impact of price limit on Chinese IPO market performance\u2019, ",
            italic("Working Paper"),
            ". Princeton University / NBER.",
          ]),

          paraNoIndent([
            "Zhang, Y. (2024) \u2018On the power of the CSAD-based test on herding behavior\u2019, ",
            italic("Applied Finance Letters"),
            ", 13, pp. 45\u201352.",
          ]),

        ],
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  const outPath = "C:\\Users\\cugut\\OneDrive\\Desktop\\tez\\thesis_v2_final.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Thesis written to: " + outPath);
  console.log("File size: " + (buffer.length / 1024).toFixed(1) + " KB");
}

buildThesis().catch(err => { console.error(err); process.exit(1); });
