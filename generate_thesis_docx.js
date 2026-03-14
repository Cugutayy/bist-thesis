const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, TabStopType, TabStopPosition
} = require('docx');

// ---- Parse the text file ----
const raw = fs.readFileSync('C:/Users/cugut/OneDrive/Desktop/tez/thesis_text_content.txt', 'utf-8');
const lines = raw.split('\n');

// ---- Helper: create a text run with formatting ----
function makeRun(text, opts = {}) {
  return new TextRun({
    text,
    font: opts.font || 'Times New Roman',
    size: opts.size || 24, // 12pt
    bold: opts.bold || false,
    italics: opts.italics || false,
    superScript: opts.super || false,
  });
}

// ---- Helper: parse inline formatting (handle special chars like γ, α, σ, ε, ±, −) ----
function parseInline(text, baseSize) {
  const sz = baseSize || 24;
  // Simple: return as single run with Times New Roman
  return [new TextRun({ text, font: 'Times New Roman', size: sz })];
}

// ---- Build table helper ----
function buildTable(headers, rows, noteText) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: '000000' };
  const borders = { top: border, bottom: border, left: border, right: border };
  const nCols = headers.length;
  const totalWidth = 9360; // US Letter with 1" margins
  const colWidth = Math.floor(totalWidth / nCols);
  const colWidths = Array(nCols).fill(colWidth);
  // Adjust last column for rounding
  colWidths[nCols - 1] = totalWidth - colWidth * (nCols - 1);

  const headerRow = new TableRow({
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: 'D9E2F3', type: ShadingType.CLEAR },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 0, line: 240 },
        children: [makeRun(h, { bold: true, size: 20 })],
      })],
    })),
  });

  const dataRows = rows.map(row => new TableRow({
    children: row.map((cell, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 0, line: 240 },
        children: [makeRun(cell, { size: 20 })],
      })],
    })),
  }));

  const elements = [
    new Table({
      width: { size: totalWidth, type: WidthType.DXA },
      columnWidths: colWidths,
      rows: [headerRow, ...dataRows],
    }),
  ];

  if (noteText) {
    elements.push(new Paragraph({
      spacing: { before: 60, after: 200 },
      children: [makeRun(noteText, { italics: true, size: 18 })],
    }));
  }

  return elements;
}

// ---- Parse tables from the text ----
// Table 1: Summary of Key Prior Studies
const table1Headers = ['Study', 'Sample', 'Method', 'Key Finding'];
const table1Rows = [
  ['Durukan (2002)', '173 IPOs, ISE, 1990\u20131997', 'First-day returns', 'Mean underpricing 14.16%'],
  ['Durukan & Onder (2006)', 'ISE IPOs', 'OLS regression', 'Weak ownership\u2013underpricing link'],
  ['Durukan et al. (2017)', 'BIST, 2006\u20132015', 'State-space + LSV', 'Herding exists; less by foreigners in crises'],
  ['Cimen & Ergun (2019)', 'BIST IPOs', 'CSSD', 'No herding detected'],
  ['Mandaci & Cagli (2022)', 'Crypto markets', 'Patterson-Sharma', 'Herding intensity and volatility'],
  ['Ilbasmi (2023)', 'BIST IPOs, COVID', 'Multi-day CAR', 'Acknowledged price-limit problem'],
  ['Zhang (2024)', 'Simulation', 'Monte Carlo', 'CSAD test power avg. 59.37%'],
  ['De Bondt & Thaler (1985)', 'NYSE', 'Portfolio formation', 'Overreaction: past losers outperform'],
  ['Ritter (1991)', '1,526 IPOs, NYSE', 'BHR', 'Long-run IPO underperformance'],
  ['Kiymaz (2000)', '163 IPOs, ISE, 1990\u20131996', 'First-day returns + aftermarket', '13.1% underpricing'],
  ['Ljungqvist (2007)', 'Survey', 'Literature review', 'Four theories of underpricing'],
  ['This study', '209 IPOs, BIST, 2020\u20132025', 'Consecutive limit-up + CSAD + Event study + OLS', '67.8% underpricing, no herding, \u20133.8% CAR'],
];
const table1Note = 'Note: ISE = Istanbul Stock Exchange (now Borsa Istanbul). CSSD = Cross-Sectional Standard Deviation. CSAD = Cross-Sectional Absolute Deviation.';

// Table 2: IPO Distribution by Year
const table2Headers = ['Year', 'N IPOs', 'Mean Underpricing (%)', 'Median Underpricing (%)', 'Avg Limit-Up Days'];
const table2Rows = [
  ['2020', '8', '105.2', '114.4', '7.1'],
  ['2021', '53', '41.5', '20.9', '2.7'],
  ['2022', '39', '54.7', '32.9', '3.7'],
  ['2023', '56', '104.2', '76.8', '6.6'],
  ['2024', '33', '61.6', '33.1', '4.2'],
  ['2025', '20', '51.8', '20.7', '3.2'],
  ['Total', '209', '67.8', '33.1', '4.4'],
];
const table2Note = 'Note: Underpricing is measured using the consecutive limit-up methodology (cumulative return from IPO-day close to first free-trading day close). Limit-up days = number of consecutive trading days closing at the +10% daily upper price limit.';

// Table 3: Descriptive Statistics
const table3Headers = ['Variable', 'N', 'Mean', 'Median', 'Std. Dev.', 'Min', 'Max'];
const table3Rows = [
  ['Underpricing (consecutive limit-up)', '204', '0.678', '0.331', '0.895', '0.000', '6.300'],
  ['First-day return', '204', '0.066', '0.100', '0.052', '\u22120.200', '0.100'],
  ['Limit-up streak (days)', '209', '4.41', '3.0', '4.67', '0', '21'],
  ['Return d30', '204', '0.510', '0.249', '0.917', '\u22120.783', '8.500'],
  ['Return d365', '198', '1.645', '0.748', '2.358', '\u22120.920', '18.200'],
  ['BHAR d365', '198', '0.638', '0.074', '2.142', '\u22121.200', '17.500'],
  ['CSAD (daily)', '1,496', '0.0155', '0.0141', '0.0058', '0.0028', '0.0855'],
  ['SPK fine (million TL)', '47', '23.99', '6.80', '89.12', '1.85', '615.00'],
];
const table3Note = 'Note: Underpricing, returns, and BHARs are expressed as decimals (0.678 = 67.8%). CSAD is computed from daily cross-sectional return dispersion.';

// Table 4: Contrarian Strategy
const table4Headers = ['Horizon', 'N', 'Mean Return (%)', 'Median Return (%)', 't-statistic', 'p-value', '% Positive', 'Signal'];
const table4Rows = [
  ['5 days', '137', '\u22123.3', '\u22126.3', '\u22122.335', '0.021**', '32.8%', 'SELL'],
  ['10 days', '137', '\u22123.4', '\u22129.3', '\u22121.606', '0.111', '30.7%', 'SELL'],
  ['30 days', '137', '+4.8', '\u221210.1', '1.092', '0.277', '37.2%', 'HOLD'],
  ['60 days', '137', '+16.6', '\u22123.4', '2.386', '0.018**', '43.1%', 'HOLD'],
  ['90 days', '136', '+21.7', '\u22124.2', '2.783', '0.006***', '46.3%', 'HOLD'],
  ['180 days', '131', '+46.0', '+1.1', '4.056', '0.000***', '51.9%', 'HOLD'],
  ['365 days', '131', '+72.6', '+26.2', '5.022', '0.000***', '61.8%', 'HOLD'],
];
const table4Note = 'Note: Returns measured from first free-trading day after the limit-up streak ends. ***, **, * denote significance at 1%, 5%, and 10% levels, respectively.';

// Table 5: Event Study
const table5Headers = ['Metric', 'Value'];
const table5Rows = [
  ['Stocks analysed', '45'],
  ['Mean CAR [\u221230, +30]', '\u22123.8%'],
  ['Significantly negative CARs (5% level)', '37 / 45 (82.2%)'],
  ['Negative CARs', '29 / 45 (64.4%)'],
  ['CAAR at Day 0', '\u22122.2%'],
  ['CAAR at Day +30', '\u22123.8%'],
  ['Pre-event CAAR [\u221230, \u22121]', '\u22122.9%'],
  ['Post-event CAAR [+1, +30]', '\u22121.6%'],
];
const table5Note = 'Note: CAR = Cumulative Abnormal Return. CAAR = Cumulative Average Abnormal Return. Event window: [\u221230, +30] trading days around SPK penalty announcement date.';

// Table 6: CSAD Herding
const table6Headers = ['Parameter', 'Estimate', 'Newey-West SE', 't-statistic', 'p-value'];
const table6Rows = [
  ['\u03B1 (intercept)', '0.0139', '0.0005', '27.45', '<0.001'],
  ['\u03B31 (|R_m|)', '0.1064', '0.0658', '1.617', '0.106'],
  ['\u03B32 (R_m\u00B2)', '0.6450', '1.4931', '0.432', '0.666'],
  ['R\u00B2', '0.096', '', '', ''],
  ['N (trading days)', '1,496', '', '', ''],
];
const table6Note = 'Note: Dependent variable is CSAD. Standard errors computed using Newey-West HAC estimator. Herding requires \u03B32 < 0 (significantly negative).';

// Table 7: OLS Regression
const table7Headers = ['Variable', 'Model 1 (Basic)', 'Model 2 (Fund.)', 'Model 4 (Log-level)'];
const table7Rows = [
  ['Constant', '1.251*** (0.260)', '4.069** (1.957)', '2.007** (0.900)'],
  ['ln(Offer Price)', '\u22120.113 (0.082)', '\u22120.086 (0.067)', '\u22120.055 (0.037)'],
  ['ln(Market Cap)', '\u2014', '\u22120.148 (0.089)', '\u22120.067 (0.040)'],
  ['P/E Ratio', '\u2014', '0.001 (0.003)', '0.000 (0.001)'],
  ['P/B Ratio', '\u2014', '0.035 (0.023)', '0.017** (0.008)'],
  ['Hot Market', '\u22120.110 (0.155)', '0.164 (0.174)', '0.063 (0.084)'],
  ['Year Dummies', 'Included', 'Included', 'Included'],
  ['N', '204', '112', '112'],
  ['R-squared', '0.097', '0.193', '0.214'],
  ['Adj. R-squared', '0.065', '0.113', '0.137'],
  ['F-statistic (p)', '4.66 (<0.001)', '1.54 (0.137)', '2.68 (0.006)'],
];
const table7Note = 'Note: HC3 robust standard errors in parentheses. ***, **, * denote significance at 1%, 5%, and 10% levels. Dependent variable: underpricing (Models 1\u20132) or ln(1 + underpricing) (Model 4).';

// ---- Extract text sections (skip table data lines) ----
// We'll parse sections by # headings and build paragraphs

function isTableDataLine(line, idx) {
  // Table data lines are between "Table N." caption and "Note:" lines
  // We handle tables separately, so skip them here
  return false; // We'll handle via section parsing
}

// ---- Build document content ----
const children = [];

// Helper for body paragraphs
function bodyPara(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 200, line: 480 }, // double space = 480 twips
    indent: opts.noIndent ? undefined : { firstLine: 720 }, // 0.5 inch indent
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: parseInline(text, 24),
  });
}

function headingPara(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 360, after: 240 },
    children: [makeRun(text, { bold: true, size: level === HeadingLevel.HEADING_1 ? 28 : 26 })],
  });
}

function tableCaptionPara(text) {
  return new Paragraph({
    spacing: { before: 240, after: 120 },
    children: [makeRun(text, { bold: true, italics: false, size: 22 })],
  });
}

// ====== TITLE PAGE ======
children.push(new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.CENTER,
  children: [makeRun('DOKUZ EYLUL UNIVERSITY', { bold: true, size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
  children: [makeRun('FACULTY OF ECONOMICS AND ADMINISTRATIVE SCIENCES', { size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [makeRun('Department of Economics', { size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
  children: [makeRun('IPO Fever and the Cost of the Crowd:', { bold: true, size: 32 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
  children: [makeRun('Underpricing, Herding, and Market Manipulation', { bold: true, size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [makeRun('in Borsa Istanbul IPOs, 2020\u20132025', { bold: true, size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [makeRun('Graduation Thesis', { italics: true, size: 26 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
  children: [makeRun('Supervisors:', { size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
  children: [makeRun('Prof. Dr. M. Banu Durukan Sali', { size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [makeRun('Assoc. Prof. Dr. Efe Caglar Cagli', { size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [makeRun('2026', { size: 24 })] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== SECTION 1: Introduction ======
children.push(headingPara('1. Introduction', HeadingLevel.HEADING_1));

// Extract body text for section 1 (lines 14-24)
const introTexts = [];
for (let i = 13; i < 24; i++) {
  const line = lines[i].trim();
  if (!line || line.startsWith('#')) continue;
  if (line.startsWith('RQ:') || line.startsWith('SQ1') || line.startsWith('SQ2') || line.startsWith('SQ3')) {
    introTexts.push({ text: line, special: true });
  } else {
    introTexts.push({ text: line, special: false });
  }
}

for (const item of introTexts) {
  if (item.special) {
    // Research questions - indented, italic label
    children.push(new Paragraph({
      spacing: { after: 120, line: 480 },
      indent: { left: 720 },
      children: parseInline(item.text, 24),
    }));
  } else {
    children.push(bodyPara(item.text));
  }
}

children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== SECTION 2: Literature Review ======
children.push(headingPara('2. Prior Literature on IPO Underpricing and Investor Behaviour', HeadingLevel.HEADING_1));

// 2.1
children.push(headingPara('2.1 IPO Underpricing: Theory and Turkish Evidence', HeadingLevel.HEADING_2));
// Lines 29-31
for (let i = 28; i <= 30; i++) {
  const line = lines[i].trim();
  if (!line || line.startsWith('#') || line.startsWith('Table')) continue;
  children.push(bodyPara(line));
}

// 2.2
children.push(headingPara('2.2 Herding Behaviour in Emerging Markets', HeadingLevel.HEADING_2));
for (let i = 33; i <= 36; i++) {
  const line = lines[i].trim();
  if (!line || line.startsWith('#') || line.startsWith('Table')) continue;
  children.push(bodyPara(line));
}

// 2.3
children.push(headingPara('2.3 IPO Aftermarket Performance and the Overreaction Hypothesis', HeadingLevel.HEADING_2));
for (let i = 38; i <= 40; i++) {
  const line = lines[i].trim();
  if (!line || line.startsWith('#')) continue;
  children.push(bodyPara(line));
}

// 2.4
children.push(headingPara('2.4 Market Manipulation and Regulatory Enforcement', HeadingLevel.HEADING_2));
// Line 43
children.push(bodyPara(lines[42].trim()));

// Table 1
children.push(tableCaptionPara('Table 1. Summary of Key Prior Studies'));
children.push(...buildTable(table1Headers, table1Rows, table1Note));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== SECTION 3: Data and Methodology ======
children.push(headingPara('3. Data and Methodology', HeadingLevel.HEADING_1));

// 3.1
children.push(headingPara('3.1 Data Sources and Sample', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[101].trim()));
children.push(bodyPara(lines[102].trim()));

// Table 2
children.push(tableCaptionPara('Table 2. IPO Distribution by Year'));
children.push(...buildTable(table2Headers, table2Rows, table2Note));

children.push(bodyPara(lines[145].trim()));

// 3.2
children.push(headingPara('3.2 Consecutive Limit-Up Methodology', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[148].trim()));

// Equations
children.push(new Paragraph({
  spacing: { before: 200, after: 100, line: 360 },
  alignment: AlignmentType.CENTER,
  children: [makeRun('Underpricing', { italics: true }), makeRun('_i', { size: 20 }), makeRun(' = (P', { italics: true }), makeRun('free,i', { size: 20 }), makeRun(' / P', { italics: true }), makeRun('close,i', { size: 20 }), makeRun(') \u2212 1')],
}));

children.push(bodyPara(lines[151].trim()));

children.push(new Paragraph({
  spacing: { before: 200, after: 100, line: 360 },
  alignment: AlignmentType.CENTER,
  children: [makeRun('Return', { italics: true }), makeRun('_d,i', { size: 20 }), makeRun(' = (P', { italics: true }), makeRun('d,i', { size: 20 }), makeRun(' / P', { italics: true }), makeRun('close,i', { size: 20 }), makeRun(') \u2212 1')],
}));

children.push(bodyPara(lines[154].trim()));

// 3.3
children.push(headingPara('3.3 CSAD Herding Methodology', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[157].trim()));

// CSAD equations
children.push(new Paragraph({
  spacing: { before: 200, after: 100, line: 360 },
  alignment: AlignmentType.CENTER,
  children: [makeRun('CSAD'), makeRun('t', { size: 20 }), makeRun(' = (1/N) \u03A3|R'), makeRun('i,t', { size: 20 }), makeRun(' \u2212 R'), makeRun('m,t', { size: 20 }), makeRun('|')],
}));

children.push(new Paragraph({
  spacing: { before: 100, after: 100, line: 360 },
  alignment: AlignmentType.CENTER,
  children: [makeRun('CSAD'), makeRun('t', { size: 20 }), makeRun(' = \u03B1 + \u03B3'), makeRun('1', { size: 20 }), makeRun('|R'), makeRun('m,t', { size: 20 }), makeRun('| + \u03B3'), makeRun('2', { size: 20 }), makeRun(' R'), makeRun('m,t', { size: 20 }), makeRun('\u00B2 + \u03B5'), makeRun('t', { size: 20 })],
}));

children.push(bodyPara(lines[161].trim()));

// 3.4
children.push(headingPara('3.4 Event Study Methodology', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[164].trim()));

// Event study equations
children.push(new Paragraph({
  spacing: { before: 200, after: 100, line: 360 },
  alignment: AlignmentType.CENTER,
  children: [makeRun('R'), makeRun('i,t', { size: 20 }), makeRun(' = \u03B1'), makeRun('i', { size: 20 }), makeRun(' + \u03B2'), makeRun('i', { size: 20 }), makeRun(' R'), makeRun('m,t', { size: 20 }), makeRun(' + \u03B5'), makeRun('i,t', { size: 20 })],
}));

children.push(bodyPara(lines[166].trim()));

children.push(new Paragraph({
  spacing: { before: 200, after: 100, line: 360 },
  alignment: AlignmentType.CENTER,
  children: [makeRun('AR'), makeRun('i,t', { size: 20 }), makeRun(' = R'), makeRun('i,t', { size: 20 }), makeRun(' \u2212 (\u03B1\u0302'), makeRun('i', { size: 20 }), makeRun(' + \u03B2\u0302'), makeRun('i', { size: 20 }), makeRun(' R'), makeRun('m,t', { size: 20 }), makeRun(')')],
}));

children.push(bodyPara(lines[168].trim()));

// 3.5
children.push(headingPara('3.5 Summary Statistics', HeadingLevel.HEADING_2));
children.push(tableCaptionPara('Table 3. Descriptive Statistics of Key Variables'));
children.push(...buildTable(table3Headers, table3Rows, table3Note));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== SECTION 4: Results ======
children.push(headingPara('4. Results and Discussion', HeadingLevel.HEADING_1));

// 4.1
children.push(headingPara('4.1 IPO Underpricing', HeadingLevel.HEADING_2));
for (let i = 240; i <= 243; i++) {
  const line = lines[i - 1]?.trim();
  if (!line || line.startsWith('#') || line.startsWith('Table')) continue;
  children.push(bodyPara(line));
}

// 4.2
children.push(headingPara('4.2 Contrarian Strategy: Post-Limit-Up Returns', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[245].trim()));

children.push(tableCaptionPara('Table 4. Post-Limit-Up Contrarian Strategy Returns'));
children.push(...buildTable(table4Headers, table4Rows, table4Note));

children.push(bodyPara(lines[312].trim()));
children.push(bodyPara(lines[313].trim()));

// 4.3
children.push(headingPara('4.3 SPK Manipulation Penalties: Event Study Results', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[316].trim()));

children.push(tableCaptionPara('Table 5. Event Study Results: SPK Manipulation Penalties'));
children.push(...buildTable(table5Headers, table5Rows, table5Note));

children.push(bodyPara(lines[337].trim()));
children.push(bodyPara(lines[338].trim()));

// 4.4
children.push(headingPara('4.4 CSAD Herding Analysis', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[341].trim()));

children.push(tableCaptionPara('Table 6. CSAD Herding Regression Results (CCK 2000)'));
children.push(...buildTable(table6Headers, table6Rows, table6Note));

children.push(bodyPara(lines[368].trim()));
children.push(bodyPara(lines[369].trim()));
children.push(bodyPara(lines[370].trim()));

// 4.5
children.push(headingPara('4.5 Cross-Sectional Determinants of Underpricing', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[373].trim()));
children.push(bodyPara(lines[374].trim()));

children.push(tableCaptionPara('Table 7. Cross-Sectional OLS Regression: Determinants of IPO Underpricing'));
children.push(...buildTable(table7Headers, table7Rows, table7Note));

children.push(bodyPara(lines[425].trim()));

// 4.6
children.push(headingPara('4.6 Inflation and Real Returns', HeadingLevel.HEADING_2));
children.push(bodyPara(lines[428].trim()));
children.push(bodyPara(lines[429].trim()));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== SECTION 5: Conclusions ======
children.push(headingPara('5. Conclusions and Policy Implications', HeadingLevel.HEADING_1));
children.push(bodyPara(lines[432].trim()));
children.push(bodyPara(lines[433].trim()));
children.push(bodyPara(lines[434].trim()));
children.push(bodyPara(lines[435].trim()));
children.push(bodyPara(lines[436].trim()));
children.push(bodyPara(lines[437].trim()));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== REFERENCES ======
children.push(headingPara('References', HeadingLevel.HEADING_1));

// Parse references (lines 440 onwards)
for (let i = 440; i < lines.length; i++) {
  const line = lines[i].trim();
  if (!line) continue;
  children.push(new Paragraph({
    spacing: { after: 120, line: 360 },
    indent: { left: 720, hanging: 720 }, // hanging indent for references
    children: parseInline(line, 22),
  }));
}

// ====== BUILD DOCUMENT ======
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: 'Times New Roman', size: 24 }, // 12pt
      },
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Times New Roman' },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Times New Roman' },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }, // 1 inch
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [makeRun('IPO Fever and the Cost of the Crowd', { italics: true, size: 18 })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: 'Times New Roman', size: 20 })],
        })],
      }),
    },
    children,
  }],
});

// Write to file
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('C:/Users/cugut/OneDrive/Desktop/tez/thesis_v4_final.docx', buffer);
  console.log('thesis_v4_final.docx generated successfully!');
}).catch(err => {
  console.error('Error generating docx:', err);
});
