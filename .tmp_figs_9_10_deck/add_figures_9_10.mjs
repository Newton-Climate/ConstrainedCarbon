import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const root = "/Users/newtonnguyen/Documents/ecosystem-complexity";
const source = `${root}/.tmp_figs_9_10_deck/base_presentation.pptx`;
const output = `${root}/notebooks/ecosystem_complexity_soil_carbon_canonical_presentation.pptx`;

async function bytes(path) {
  const b = await fs.readFile(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}
function textbox(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", position, fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { fontFace: "Calibri", color: "#2A2A2A", ...style };
  return shape;
}
function rect(slide, position, fill, line = fill) {
  return slide.shapes.add({ geometry: "rect", position, fill, line: { style: "solid", fill: line, width: 1 } });
}
function header(slide, eyebrow, title, page) {
  textbox(slide, eyebrow, { left: 58, top: 30, width: 1080, height: 28 },
    { fontSize: 13, bold: true, color: "#4A7C59", charSpacing: 3 });
  textbox(slide, title, { left: 58, top: 62, width: 1120, height: 48 },
    { fontFace: "Georgia", fontSize: 27, bold: true, color: "#1F3A2E" });
  textbox(slide, `${page} / 27`, { left: 1180, top: 682, width: 60, height: 20 },
    { fontSize: 9, color: "#6B6B6B", alignment: "right" });
}
function notes(slide, body) {
  if (slide.notes?.setText) slide.notes.setText(body);
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(source));
const fig9 = await bytes(`${root}/notebooks/paper_figs/outputs/current_results/figures/figure_09.png`);
const fig10Top = await bytes(`${root}/.tmp_figs_9_10_deck/figure_10_top.png`);
const fig10Bottom = await bytes(`${root}/.tmp_figs_9_10_deck/figure_10_bottom.png`);

{
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  header(slide, "CROSS-ECOSYSTEM RESULTS  ·  01", "Ecosystem type shapes vulnerability; observations shape constrainability", 25);
  slide.images.add({ blob: fig9, contentType: "image/png", alt: "Figure 9: cross-ecosystem turnover, vulnerability, and observation-family contributions", fit: "contain", position: { left: 45, top: 138, width: 845, height: 525 } });
  textbox(slide, "WHAT FIGURE 9 SHOWS", { left: 915, top: 160, width: 290, height: 24 }, { fontSize: 11, bold: true, color: "#4A7C59", charSpacing: 2 });
  textbox(slide, "All ecosystems occupy the same multi-pool turnover space, but warming loss is not explained by total DFS alone.", { left: 915, top: 194, width: 295, height: 86 }, { fontFace: "Georgia", fontSize: 15, bold: true, color: "#1F3A2E" });
  textbox(slide, "• Boreal ecosystems have the largest mean fractional C loss.\n• Arctic/permafrost systems have the oldest excess respiration.\n• C stocks, bulk ¹⁴C, and respired ¹⁴C contribute differently across biomes.", { left: 915, top: 320, width: 295, height: 145 }, { fontSize: 12, color: "#2A2A2A" });
  rect(slide, { left: 915, top: 510, width: 295, height: 86 }, "#F7F4ED", "#D4A574");
  textbox(slide, "Constrainability is a property of the observation set—not a proxy for vulnerability.", { left: 932, top: 530, width: 260, height: 46 }, { fontFace: "Georgia", fontSize: 12.5, italic: true, color: "#1F3A2E" });
  notes(slide, "[Sources]\nFigure 9 generated from the canonical cross-ecosystem results pipeline: notebooks/paper_figs/outputs/current_results/figures/figure_09.png.\n[/Sources]");
}
{
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  header(slide, "CROSS-ECOSYSTEM RESULTS  ·  02", "Pool separation identifies older warming-enhanced respiration", 26);
  slide.images.add({ blob: fig10Top, contentType: "image/png", alt: "Figure 10 upper panels: turnover separation, old respiration, and carbon loss", fit: "contain", position: { left: 40, top: 126, width: 1200, height: 456 } });
  rect(slide, { left: 135, top: 600, width: 1010, height: 62 }, "#F7F4ED", "#D4A574");
  textbox(slide, "Across the 24 sites with direct warming output, greater passive-to-active turnover separation predicts a higher old-C share of excess respiration (ρₛ = 0.74), but a lower—not higher—fractional carbon loss (ρₛ = −0.76).", { left: 160, top: 617, width: 960, height: 30 }, { fontFace: "Georgia", fontSize: 13, italic: true, color: "#1F3A2E", alignment: "center" });
  notes(slide, "[Sources]\nFigure 10 generated from the canonical cross-ecosystem results pipeline: notebooks/paper_figs/outputs/current_results/figures/figure_10.png.\n[/Sources]");
}
{
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  header(slide, "CROSS-ECOSYSTEM RESULTS  ·  03", "Pool separation does not predict total DFS", 27);
  slide.images.add({ blob: fig10Bottom, contentType: "image/png", alt: "Figure 10 lower panels: constrainability, raw radiocarbon gap, and high-vs-low separation", fit: "contain", position: { left: 40, top: 126, width: 1200, height: 456 } });
  rect(slide, { left: 135, top: 600, width: 1010, height: 62 }, "#1F3A2E", "#1F3A2E");
  textbox(slide, "Total DFS is essentially unrelated to turnover separation (ρₛ = 0.12); the raw radiocarbon gap is weaker than the model-based separation metric. Complexity shifts vulnerability toward older carbon—it does not set its magnitude.", { left: 160, top: 616, width: 960, height: 34 }, { fontFace: "Georgia", fontSize: 12.5, italic: true, color: "#FFFFFF", alignment: "center" });
  notes(slide, "[Sources]\nFigure 10 generated from the canonical cross-ecosystem results pipeline: notebooks/paper_figs/outputs/current_results/figures/figure_10.png.\n[/Sources]");
}

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(output);
for (const [i, slide] of presentation.slides.items.entries()) {
  if (i < 24) continue;
  await writeBlob(`${root}/.tmp_figs_9_10_deck/slide-${i + 1}.png`, await presentation.export({ slide, format: "png", scale: 1 }));
}
await writeBlob(`${root}/.tmp_figs_9_10_deck/montage.webp`, await presentation.export({ format: "webp", montage: true, scale: 1 }));
console.log(output);
