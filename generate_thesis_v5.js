const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, TabStopType, TabStopPosition
} = require('docx');

// Read the thesis text
const text = fs.readFileSync('C:\\Users\\cugut\\OneDrive\\Desktop\\tez\\thesis_text_content.txt', 'utf-8');
const lines = text.split('\n');

// =============================================
// PARSE THE THESIS TEXT INTO STRUCTURED DATA
// =============================================

// Table definitions - which tables have which columns
const TABLE_DEFS = {
  'Table 1. Summary of Key Prior Studies': {
    cols: ['Study', 'Sample', 'Method', 'Key Finding'],
    widths: [2200, 2500, 2200, 2460],
  },
  'Table 2. IPO Distribution by Year': {
    cols: ['Year', 'N IPOs', 'Mean Underpricing (%)', 'Median Underpricing (%)', 'Avg Limit-Up Days'],
    widths: [1500, 1500, 2200, 2200, 2000],
  },
  'Table 3. Descriptive Statistics of Key Variables': {
    cols: ['Variable', 'N', 'Mean', 'Median', 'Std. Dev.', 'Min', 'Max'],
    widths: [2800, 900, 1000, 1000, 1100, 1000, 1000],
  },
  'Table 4. Post-Limit-Up Contrarian Strategy Returns': {
    cols: ['Horizon', 'N', 'Mean Return (%)', 'Median Return (%)', 't-statistic', 'p-value', '% Positive', 'Signal'],
    widths: [1100, 700, 1300, 1300, 1100, 1100, 1100, 900],
  },
  'Table 5. Event Study Results: SPK Manipulation Penalties': {
    cols: ['Metric', 'Value'],
    widths: [5500, 3860],
  },
  'Table 6. CSAD Herding Regression Results (CCK 2000)': {
    cols: ['Parameter', 'Estimate', 'Newey-West SE', 't-statistic', 'p-value'],
    widths: [2000, 1800, 1800, 1800, 1960],
  },
  'Table 7. Cross-Sectional OLS Regression: Determinants of IPO Underpricing': {
    cols: ['Variable', 'Model 1 (Basic)', 'Model 2 (Fund.)', 'Model 4 (Log-level)'],
    widths: [2400, 2320, 2320, 2320],
  },
};

// Parse the content
const elements = [];
let i = 0;

// Title page (lines 0-9)
elements.push({ type: 'title-page', lines: lines.slice(0, 10) });
i = 11; // skip blank line after title

while (i < lines.length) {
  const line = lines[i].trim();

  if (!line) { i++; continue; }

  // Main section heading: # 1. Introduction
  if (line.startsWith('# ')) {
    elements.push({ type: 'h1', text: line.slice(2) });
    i++;
    continue;
  }

  // Subsection heading: ## 2.1 ...
  if (line.startsWith('## ')) {
    elements.push({ type: 'h2', text: line.slice(3) });
    i++;
    continue;
  }

  // Table detection
  let foundTable = null;
  for (const tableName of Object.keys(TABLE_DEFS)) {
    if (line === tableName) {
      foundTable = tableName;
      break;
    }
  }

  if (foundTable) {
    const def = TABLE_DEFS[foundTable];
    const numCols = def.cols.length;

    // Skip the header row (column names) - they come one per line after table name
    let j = i + 1;
    // Read column headers
    const headers = [];
    for (let c = 0; c < numCols && j < lines.length; c++) {
      headers.push(lines[j].trim());
      j++;
    }

    // Read data rows until we hit "Note:" or a blank line followed by a non-data line
    const rows = [];
    while (j < lines.length) {
      const nextLine = lines[j].trim();
      if (nextLine.startsWith('Note:') || nextLine === '') break;
      if (nextLine.startsWith('#')) break;

      // Read one row (numCols lines)
      const row = [];
      for (let c = 0; c < numCols && j < lines.length; c++) {
        const val = lines[j].trim();
        if (val.startsWith('Note:') || val === '' || val.startsWith('#')) break;
        row.push(val);
        j++;
      }
      if (row.length === numCols) {
        rows.push(row);
      } else {
        // Put back unread lines
        j -= row.length;
        break;
      }
    }

    // Read note if present
    let note = '';
    if (j < lines.length && lines[j].trim().startsWith('Note:')) {
      note = lines[j].trim();
      j++;
    }

    elements.push({ type: 'table', name: foundTable, headers: def.cols, rows, note, widths: def.widths });
    i = j;
    continue;
  }

  // Regular paragraph
  elements.push({ type: 'paragraph', text: line });
  i++;
}

// =============================================
// BUILD THE DOCX
// =============================================

const border = { style: BorderStyle.SINGLE, size: 1, color: '999999' };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 80, right: 80 };

function makeTableCell(text, width, isHeader = false) {
  const runs = [new TextRun({
    text: text,
    bold: isHeader,
    font: 'Times New Roman',
    size: 18, // 9pt for tables
  })];

  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: cellMargins,
    shading: isHeader ? { fill: 'E8E8E8', type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      children: runs,
      spacing: { before: 0, after: 0, line: 240 },
      alignment: isHeader ? AlignmentType.CENTER : AlignmentType.LEFT,
    })],
  });
}

function buildTable(el) {
  const totalWidth = el.widths.reduce((a, b) => a + b, 0);

  // Header row
  const headerRow = new TableRow({
    children: el.headers.map((h, idx) => makeTableCell(h, el.widths[idx], true)),
    tableHeader: true,
  });

  // Data rows
  const dataRows = el.rows.map(row =>
    new TableRow({
      children: row.map((cell, idx) => makeTableCell(cell, el.widths[idx])),
    })
  );

  const result = [];

  // Table caption
  result.push(new Paragraph({
    children: [new TextRun({ text: el.name, bold: true, font: 'Times New Roman', size: 20, italics: true })],
    spacing: { before: 240, after: 120 },
    alignment: AlignmentType.LEFT,
  }));

  result.push(new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: el.widths,
    rows: [headerRow, ...dataRows],
  }));

  // Note
  if (el.note) {
    result.push(new Paragraph({
      children: [new TextRun({ text: el.note, font: 'Times New Roman', size: 18, italics: true })],
      spacing: { before: 60, after: 200 },
    }));
  }

  return result;
}

function makeTextRun(text, opts = {}) {
  return new TextRun({
    text: text,
    font: 'Times New Roman',
    size: 24, // 12pt
    ...opts,
  });
}

// Build children array
const children = [];

// Process each element
for (const el of elements) {
  if (el.type === 'title-page') {
    // Title page
    children.push(new Paragraph({ spacing: { before: 2400 }, children: [] }));
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[0], { size: 28, bold: true })], // DOKUZ EYLUL UNIVERSITY
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
    }));
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[1], { size: 24 })], // FACULTY OF ECONOMICS AND ADMINISTRATIVE SCIENCES
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
    }));
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[2], { size: 22 })], // Department
      alignment: AlignmentType.CENTER,
      spacing: { after: 600 },
    }));
    // Title
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[3], { size: 36, bold: true })], // IPO Fever...
      alignment: AlignmentType.CENTER,
      spacing: { after: 40 },
    }));
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[4], { size: 30, bold: true })], // Underpricing...
      alignment: AlignmentType.CENTER,
      spacing: { after: 40 },
    }));
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[5], { size: 30, bold: true })], // in Borsa Istanbul...
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
    }));
    // Graduation Thesis
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[6], { size: 26, italics: true })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
    }));
    // Student name
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[7], { size: 24, bold: true })], // Salih Çağatay Sönmez
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
    }));
    // Student number
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[8], { size: 22 })], // 2021431045
      alignment: AlignmentType.CENTER,
      spacing: { after: 600 },
    }));
    children.push(new Paragraph({
      children: [makeTextRun(el.lines[9], { size: 22 })], // 2026
      alignment: AlignmentType.CENTER,
      spacing: { after: 0 },
    }));
    // Page break after title
    children.push(new Paragraph({ children: [new PageBreak()] }));
    continue;
  }

  if (el.type === 'h1') {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: el.text, bold: true, font: 'Times New Roman', size: 28 })],
      spacing: { before: 360, after: 200 },
      pageBreakBefore: el.text.includes('1.') ? false : true, // Page break before sections 2+
    }));
    continue;
  }

  if (el.type === 'h2') {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: el.text, bold: true, font: 'Times New Roman', size: 24 })],
      spacing: { before: 280, after: 160 },
    }));
    continue;
  }

  if (el.type === 'table') {
    children.push(...buildTable(el));
    continue;
  }

  if (el.type === 'paragraph') {
    const text = el.text;

    // Special formatting for RQ and SQ lines
    if (text.startsWith('RQ:') || text.startsWith('SQ1') || text.startsWith('SQ2') || text.startsWith('SQ3')) {
      children.push(new Paragraph({
        children: [makeTextRun(text, { italics: true })],
        spacing: { before: 120, after: 120 },
        indent: { left: 360 },
      }));
      continue;
    }

    // References section - smaller font, hanging indent
    if (text.match(/^[A-Z][a-z]+,.*\(\d{4}\)/)) {
      children.push(new Paragraph({
        children: [makeTextRun(text, { size: 22 })],
        spacing: { before: 60, after: 60 },
        indent: { left: 720, hanging: 720 },
      }));
      continue;
    }

    // Formula lines
    if (text.startsWith('Underpricing_i') || text.startsWith('Return_d,i') ||
        text.startsWith('CSAD_t') || text.startsWith('R_i,t') ||
        text.startsWith('AR_i,t')) {
      children.push(new Paragraph({
        children: [makeTextRun(text, { italics: true })],
        spacing: { before: 120, after: 120 },
        alignment: AlignmentType.CENTER,
      }));
      continue;
    }

    // "where" explanation lines
    if (text.startsWith('where ')) {
      children.push(new Paragraph({
        children: [makeTextRun(text)],
        spacing: { before: 60, after: 120 },
        indent: { left: 360 },
      }));
      continue;
    }

    // Regular paragraph
    children.push(new Paragraph({
      children: [makeTextRun(text)],
      spacing: { before: 60, after: 120, line: 360 }, // 1.5 line spacing
      alignment: AlignmentType.JUSTIFIED,
    }));
  }
}

// Create the document
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: 'Times New Roman', size: 24 },
      },
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Times New Roman' },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Times New Roman' },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }, // slightly wider left margin
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [new TextRun({ text: 'IPO Fever and the Cost of the Crowd', font: 'Times New Roman', size: 16, italics: true, color: '888888' })],
          alignment: AlignmentType.RIGHT,
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          children: [new TextRun({ children: [PageNumber.CURRENT], font: 'Times New Roman', size: 18 })],
          alignment: AlignmentType.CENTER,
        })],
      }),
    },
    children: children,
  }],
});

// Generate the file
const outputPath = 'C:\\Users\\cugut\\OneDrive\\Desktop\\tez\\thesis_v5_spd.docx';
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log(`Thesis generated: ${outputPath}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(1)} KB`);
  console.log(`Elements: ${elements.length}, Children: ${children.length}`);
}).catch(err => {
  console.error('Error:', err);
});
