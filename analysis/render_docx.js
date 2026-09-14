// Render deliverables/dossier.json to a Word document.
//   node analysis/render_docx.js
// Tables are paper-ready: paste straight into the manuscript.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, ImageRun, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const data = JSON.parse(fs.readFileSync(path.join(ROOT, "deliverables", "dossier.json"), "utf8"));

const PAGE_W = 12240, MARGIN = 1080;         // US Letter, 0.75in margins
const CONTENT_W = PAGE_W - 2 * MARGIN;
const TEAL = "0C666D", GOLD = "8A6508", GREY = "5C6B70", RULE = "D3DCDE";

function text(t, opts = {}) { return new TextRun({ text: t, ...opts }); }

function para(runs, opts = {}) {
  return new Paragraph({ children: Array.isArray(runs) ? runs : [runs], ...opts });
}

function label(t, colour) {
  return para([text(t.toUpperCase(), { bold: true, size: 15, color: colour, characterSpacing: 20 })],
    { spacing: { before: 200, after: 40 } });
}

function body(t, opts = {}) {
  return para([text(t, { size: 21, ...opts })], { spacing: { after: 100, line: 276 } });
}

function cell(t, { header = false, width, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, fill: "EDF3F4" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [para([text(String(t ?? ""), { bold: header, size: 17 })], { alignment: align })],
  });
}

function buildTable(tbl) {
  const cols = tbl.columns.length;
  const widths = Array(cols).fill(Math.floor(CONTENT_W / cols));
  widths[0] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const fmt = (v) =>
    typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : v;
  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: tbl.columns.map((c, i) => cell(c, { header: true, width: widths[i] })),
      }),
      ...tbl.rows.map((r) => new TableRow({
        children: r.map((v, i) => cell(fmt(v), {
          width: widths[i],
          align: typeof v === "number" ? AlignmentType.RIGHT : AlignmentType.LEFT,
        })),
      })),
    ],
  });
}

const children = [];

// ---- title page
children.push(
  para([text(data.title, { bold: true, size: 40, color: "121A1D" })],
    { spacing: { before: 1200, after: 160 } }),
  para([text(data.subtitle, { size: 26, color: GREY })], { spacing: { after: 400 } }),
  para([text(`Generated ${data.generated_utc}  ·  commit ${data.git_sha}  ·  config ${data.config_hash}`,
    { size: 17, color: GREY })], { spacing: { after: 120 } }),
  body("Every number below carries the pipeline stage and results table that produced it. "
    + "Statements are written to be quoted directly; interpretation is kept separate from "
    + "measurement so the argument in the paper remains the author's."),
  para([text("Numbers keyed for citation are in results/quotable.csv. Each table here is "
    + "also written individually to deliverables/tables/.", { size: 19, italics: true, color: GREY })],
    { spacing: { after: 200 } }),
  para([new PageBreak()]),
);

// ---- sections
data.sections.forEach((section, si) => {
  if (si > 0) children.push(para([new PageBreak()]));
  children.push(
    para([text(section.title, { bold: true, size: 30, color: TEAL })],
      { heading: HeadingLevel.HEADING_1, spacing: { after: 100 } }),
    para([text(section.lead, { size: 21, color: GREY, italics: true })],
      { spacing: { after: 260 } }),
  );

  section.blocks.forEach((block) => {
    children.push(
      para([text(block.heading, { bold: true, size: 24 })],
        { heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 100 } }),
      label("Result", TEAL),
      body(block.statement),
      label("What it means", GOLD),
      body(block.interpretation),
    );

    if (block.caveat) {
      children.push(label("Caveat", GREY), body(block.caveat, { italics: true }));
    }

    if (block.table) {
      if (block.table_caption) {
        children.push(para([text(block.table_caption, { size: 18, bold: true })],
          { spacing: { before: 180, after: 80 } }));
      }
      children.push(buildTable(block.table));
      if (block.table_file) {
        children.push(para([text(`deliverables/tables/${block.table_file}`,
          { size: 15, color: GREY, italics: true })], { spacing: { before: 60, after: 140 } }));
      }
    }

    if (block.figure) {
      const fig = path.join(ROOT, "figures", block.figure);
      if (fs.existsSync(fig)) {
        children.push(para([new ImageRun({
          type: "png", data: fs.readFileSync(fig),
          transformation: { width: 620, height: 300 },
        })], { spacing: { before: 200, after: 60 }, alignment: AlignmentType.CENTER }));
        if (block.figure_caption) {
          children.push(para([text(block.figure_caption, { size: 17, color: GREY })],
            { spacing: { after: 140 } }));
        }
      }
    }

    if (block.source) {
      children.push(para([text(`Source: ${block.source}`, { size: 15, color: GREY })],
        { spacing: { after: 200 },
          border: { top: { style: BorderStyle.SINGLE, size: 2, color: RULE, space: 6 } } }));
    }
  });
});

const doc = new Document({
  creator: "MultiContouring pipeline",
  title: data.title,
  styles: { default: { document: { run: { font: "Calibri" } } } },
  sections: [{
    properties: {
      page: { size: { width: PAGE_W, height: 15840 },
              margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
    },
    children,
  }],
});

const out = path.join(ROOT, "deliverables", "results_dossier.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out}`);
});
