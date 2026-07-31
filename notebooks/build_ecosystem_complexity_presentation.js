// build_ecosystem_complexity_presentation.js — Ecosystem-complexity / soil-C presentation.
// Run: node notebooks/build_ecosystem_complexity_presentation.js

const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG_DIR = "/Users/newtonnguyen/Documents/ecosystem-complexity/notebooks";
const fig = (name) => path.join(FIG_DIR, name);

// ─── Palette: Forest + warm radiocarbon accent ─────────────────────────────
const C = {
  primary:   "1F3A2E",  // deep forest
  secondary: "4A7C59",  // moss
  accent:    "D4A574",  // warm sand / radiocarbon
  body:      "2A2A2A",
  muted:     "6B6B6B",
  bg:        "FFFFFF",
  bgDark:    "1F3A2E",
  bgLight:   "F7F4ED",
};

const FONT_HEAD = "Georgia";
const FONT_BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.3" × 7.5"
pres.author = "Newton H. Nguyen";
pres.title  = "Ecosystem complexity and the vulnerability of soil carbon";

const W = 13.3, H = 7.5;

// ─── Reusable helpers ───────────────────────────────────────────────────────
function addSlideHeader(slide, eyebrow, title) {
  slide.addText(eyebrow, {
    x: 0.6, y: 0.35, w: 12.1, h: 0.32,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  slide.addText(title, {
    x: 0.6, y: 0.7, w: 12.1, h: 0.85,
    fontSize: 28, fontFace: FONT_HEAD, bold: true,
    color: C.primary, margin: 0, valign: "top",
  });
}

function addPageNumber(slide, n, total) {
  slide.addText(`${n} / ${total}`, {
    x: 12.4, y: 7.1, w: 0.7, h: 0.3,
    fontSize: 9, fontFace: FONT_BODY, color: C.muted,
    align: "right", margin: 0,
  });
}

function placeholderBox(slide, x, y, w, h, label) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.bgLight },
    line: { color: C.accent, width: 1.25, dashType: "dash" },
  });
  slide.addText([
    { text: "FIGURE PLACEHOLDER\n",
      options: { bold: true, fontSize: 11, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: label, options: { fontSize: 13, color: C.muted, italic: true } },
  ], {
    x, y, w, h, align: "center", valign: "middle",
    fontFace: FONT_BODY, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bgDark };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.25, h: H, fill: { color: C.accent }, line: { color: C.accent },
  });
  s.addText("ECOSYSTEM COMPLEXITY & SOIL CARBON", {
    x: 0.9, y: 1.8, w: 11.5, h: 0.5,
    fontSize: 14, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 6, margin: 0,
  });
  s.addText("Quantifying the information content\nof radiocarbon constraints on\nsoil-carbon vulnerability", {
    x: 0.9, y: 2.4, w: 11.5, h: 2.5,
    fontSize: 40, fontFace: FONT_HEAD, bold: true,
    color: "FFFFFF", margin: 0, valign: "top",
    paraSpaceAfter: 6,
  });
  s.addText([
    { text: "Newton H. Nguyen",
      options: { bold: true, color: "FFFFFF", fontSize: 16, breakLine: true } },
    { text: "Stanford University · nnewton@stanford.edu",
      options: { color: C.accent, fontSize: 12 } },
  ], { x: 0.9, y: 5.5, w: 11.5, h: 0.9, fontFace: FONT_BODY, margin: 0, valign: "top" });
  s.addText("Two canonical ecosystems · Optimal estimation inversion · Information theory", {
    x: 0.9, y: 6.6, w: 11.5, h: 0.4,
    fontSize: 11, fontFace: FONT_BODY, italic: true,
    color: C.accent, margin: 0,
  });
}

const TOTAL = 26;
let n = 1;

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Motivation: Carbon may be coming from deeper stocks
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "MOTIVATION  ·  01", "Carbon may be coming from deeper stocks");

  // Left column: thesis + bullets
  s.addText("Deep soils hold a vast reservoir of millennia-old carbon — and warming experiments show it's mobilizable.", {
    x: 0.6, y: 1.75, w: 5.0, h: 1.6,
    fontSize: 16, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });
  s.addText([
    { text: "What this means",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 5 } },
    { text: "Deep-soil C (30–100 cm) spans ages from ~10² to >10⁴ yr.",
      options: { bullet: true, fontSize: 12.5, color: C.body, breakLine: true } },
    { text: "Whole-soil warming releases this previously inert carbon as CO₂.",
      options: { bullet: true, fontSize: 12.5, color: C.body, breakLine: true } },
    { text: "Surface-only assumptions in ESMs systematically underestimate vulnerability.",
      options: { bullet: true, fontSize: 12.5, color: C.body, breakLine: true } },
    { text: "Radiocarbon (¹⁴C) provides a direct age fingerprint of what's being respired.",
      options: { bullet: true, fontSize: 12.5, color: C.body } },
  ], { x: 0.6, y: 3.55, w: 5.0, h: 3.5,
       fontFace: FONT_BODY, margin: 0, valign: "top", paraSpaceAfter: 5 });

  // Middle column: Shi 2020 Fig. 2 (AR 1.42 — landscape-ish)
  // Sized to match native AR: 3.45 × 2.43
  s.addImage({
    path: fig("shi2020_fig2_global_carbon_distribution.png"),
    x: 5.85, y: 1.85, w: 3.45, h: 2.43,
    sizing: { type: "contain", w: 3.45, h: 2.43 },
  });
  s.addText("Global soil-C age distribution by depth\n(Shi et al. 2020, Fig. 2)", {
    x: 5.75, y: 4.35, w: 3.65, h: 0.6,
    fontSize: 9.5, fontFace: FONT_BODY, italic: true,
    color: C.muted, align: "center", margin: 0,
  });

  // Right column: Nottingham 2020 Fig. 2 (AR 0.89 — portrait)
  // Sized to match native AR: 3.20 × 3.60
  s.addImage({
    path: fig("nottingham2020_fig2_tropical_soil_warming.png"),
    x: 9.55, y: 1.85, w: 3.20, h: 3.60,
    sizing: { type: "contain", w: 3.20, h: 3.60 },
  });
  s.addText("Tropical-forest whole-soil warming\n(Nottingham et al. 2020, Fig. 2)", {
    x: 9.45, y: 5.50, w: 3.40, h: 0.6,
    fontSize: 9.5, fontFace: FONT_BODY, italic: true,
    color: C.muted, align: "center", margin: 0,
  });

  // Bottom band: synthesis
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.85, y: 6.2, w: 6.95, h: 0.85,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 0.8 },
  });
  s.addText([
    { text: "One-two punch:  ",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 2 } },
    { text: "deep soils are old (Shi);  warming mobilizes that old C as respired CO₂ (Nottingham, Hicks Pries).",
      options: { fontSize: 12, fontFace: FONT_HEAD, italic: true, color: C.primary } },
  ], { x: 6.0, y: 6.2, w: 6.7, h: 0.85, fontFace: FONT_BODY, valign: "middle", margin: 0 });

  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Models disagree on vulnerability
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "MOTIVATION  ·  02", "Earth System Models disagree on soil-carbon vulnerability");
  // shi2020_fig3 native AR 1.21 — size container to match (5.93 × 4.90)
  s.addImage({
    path: fig("shi2020_fig3_modelled_14C_map.png"),
    x: 0.6, y: 1.75, w: 5.93, h: 4.9,
    sizing: { type: "contain", w: 5.93, h: 4.9 },
  });
  s.addText("Observed (a, b) vs modelled (c–f) soil-C Δ¹⁴C distributions, by biome  ·  Shi et al. 2020, Fig. 3", {
    x: 0.6, y: 6.7, w: 5.93, h: 0.3,
    fontSize: 9, fontFace: FONT_BODY, italic: true,
    color: C.muted, align: "center", margin: 0,
  });
  s.addText([
    { text: "The disagreement",
      options: { bold: true, fontSize: 16, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Observations (a, b): heavy tails of depleted Δ¹⁴C — old C is real.",
      options: { fontSize: 13.5, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 5 } },
    { text: "Models (c–f): distributions cluster near atmospheric Δ¹⁴C — old C is missing.",
      options: { fontSize: 13.5, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Why it matters",
      options: { bold: true, fontSize: 16, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Turnover time × stock = climate-feedback potential.",
      options: { fontSize: 13.5, color: C.body, italic: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 5 } },
    { text: "Without ¹⁴C constraints we cannot tell which ESM is right.",
      options: { fontSize: 13.5, color: C.body, italic: true } },
  ], { x: 6.8, y: 1.85, w: 5.9, h: 5.0, fontFace: FONT_BODY, margin: 0, valign: "top", paraSpaceAfter: 6 });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Radio carbon encodes time
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "WHY ¹⁴C  ·  01", "Radiocarbon encodes the age of soil carbon");
  s.addText("Atmospheric Δ¹⁴C has varied dramatically — pre-bomb depletion, bomb spike (1963), modern decline.", {
    x: 0.6, y: 1.75, w: 5.6, h: 1.3,
    fontSize: 17, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });
  s.addText([
    { text: "¹⁴C as a tracer",
      options: { bold: true, fontSize: 14, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 5 } },
    { text: "Carbon entering soil each year carries the atmospheric Δ¹⁴C of that year.",
      options: { bullet: true, fontSize: 13, color: C.body, breakLine: true } },
    { text: "Bulk soil Δ¹⁴C reflects a mixture of past inputs weighted by pool age structure.",
      options: { bullet: true, fontSize: 13, color: C.body, breakLine: true } },
    { text: "Density-fractionated samples separate fast-cycling vs. mineral-protected C.",
      options: { bullet: true, fontSize: 13, color: C.body, breakLine: true } },
    { text: "Respired CO₂ Δ¹⁴C reveals the age of carbon actively being released.",
      options: { bullet: true, fontSize: 13, color: C.body } },
  ], { x: 0.6, y: 3.3, w: 5.6, h: 3.6, fontFace: FONT_BODY, margin: 0, valign: "top", paraSpaceAfter: 5 });
  // atm_14c_timeseries: native AR 2.27 (wide).  Container 6.4 wide → 2.82 tall.
  s.addImage({
    path: fig("atmospheric_delta14c_sampling_windows.png"),
    x: 6.4, y: 1.85, w: 6.4, h: 2.82,
    sizing: { type: "contain", w: 6.4, h: 2.82 },
  });
  s.addText("Atmospheric Δ¹⁴C record (Hua 2021 NH/SH + Graven 2017)  ·  IntCal20 pre-1950 + Hua post-bomb", {
    x: 6.4, y: 4.75, w: 6.4, h: 0.3,
    fontSize: 9.5, fontFace: FONT_BODY, italic: true,
    color: C.muted, align: "center", margin: 0,
  });

  // Bottom-right: site / observation summary card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.4, y: 5.25, w: 6.4, h: 1.85,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 0.8 },
  });
  s.addText("OUR ¹⁴C OBSERVATIONS", {
    x: 6.55, y: 5.35, w: 6.1, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText([
    { text: "Harvard Forest:  ", options: { bold: true, fontSize: 11, color: C.primary } },
    { text: "8 density-fractionated pool obs (hf212-03) · 41 NWN respired CO₂ obs (1996–2010) · 6 ISRaD H1–H5 fraction obs",
      options: { fontSize: 11, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "Barrow:  ", options: { bold: true, fontSize: 11, color: C.primary } },
    { text: "3 ISRaD bulk soil layers (Vaughn 2018 + Nave 2021, 73 layer measurements pooled) · 14 Vaughn surface emissions (2012–2014)",
      options: { fontSize: 11, color: C.body } },
  ], { x: 6.55, y: 5.65, w: 6.1, h: 1.4, fontFace: FONT_BODY, valign: "top", margin: 0, paraSpaceAfter: 2 });

  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Different ecosystems show different respired ¹⁴C
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "WHY ¹⁴C  ·  02",
    "Respired ¹⁴C varies systematically across biomes — and our sites fit the pattern");

  s.addImage({
    path: fig("soil_delta14c_biome_distribution_with_sites.png"),
    x: 0.6, y: 1.65, w: 12.3, h: 4.6,
    sizing: { type: "contain", w: 12.3, h: 4.6 },
  });

  // Bottom band with the key finding (resp − bulk gap)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.3, w: 12.3, h: 0.85,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "Respired Δ¹⁴C is always more enriched than bulk soil — but the gap differs by biome.  ",
      options: { bold: true, fontSize: 12.5, color: C.secondary } },
    { text: " ", options: { fontSize: 5, breakLine: true } },
    { text: "Tropical +49 ‰   ·   Boreal +70 ‰   ·   Tundra +95 ‰   ·   Temperate +111 ‰  ",
      options: { fontSize: 12, color: C.body } },
    { text: "— respiration draws from carbon younger than the bulk soil mean, and the gap grows where deep stable C dominates.",
      options: { fontSize: 12, fontFace: FONT_HEAD, italic: true, color: C.primary } },
  ], { x: 0.75, y: 6.3, w: 12.0, h: 0.85, fontFace: FONT_BODY, valign: "middle", margin: 0 });

  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Gap / research questions
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bgLight };
  n++;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 1.4, fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("THE GAP", {
    x: 0.6, y: 0.35, w: 8, h: 0.35,
    fontSize: 12, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 6, margin: 0,
  });
  s.addText("What do we still not know?", {
    x: 0.6, y: 0.65, w: 12.1, h: 0.55,
    fontSize: 26, fontFace: FONT_HEAD, italic: true,
    color: "FFFFFF", margin: 0,
  });

  const qY = 2.0;
  const cardH = 1.55;
  const questions = [
    { label: "Q1", text: "What is the role of ecosystem complexity in the vulnerability of carbon stocks?" },
    { label: "Q2", text: "How much can different soil-carbon observations constrain that vulnerability?" },
    { label: "Q3", text: "Are climate models over- or under-predicting the vulnerability of soil C?" },
  ];
  questions.forEach((q, i) => {
    const y = qY + i * (cardH + 0.3);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y, w: 12.1, h: cardH,
      fill: { color: "FFFFFF" }, line: { color: C.secondary, width: 0.75 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y, w: 1.4, h: cardH,
      fill: { color: C.secondary }, line: { color: C.secondary },
    });
    s.addText(q.label, {
      x: 0.6, y, w: 1.4, h: cardH,
      fontSize: 38, fontFace: FONT_HEAD, bold: true,
      color: "FFFFFF", align: "center", valign: "middle", margin: 0,
    });
    s.addText(q.text, {
      x: 2.2, y, w: 10.3, h: cardH,
      fontSize: 18, fontFace: FONT_HEAD, color: C.body,
      valign: "middle", margin: 0,
    });
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Approach
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "APPROACH", "Constrained-Carbon model: data + structure + information theory");

  s.addText("MODEL STRUCTURE", {
    x: 0.6, y: 1.75, w: 4.2, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  ["Flexible n-pool topology", "JAX-differentiable forward model", "¹²C + ¹⁴C tracer fluxes in parallel"].forEach((t, i) => {
    s.addText("●  " + t, {
      x: 0.6, y: 2.15 + i*0.4, w: 4.2, h: 0.35,
      fontSize: 13, fontFace: FONT_BODY, color: C.body, margin: 0,
    });
  });

  s.addText("OPTIMIZATION", {
    x: 5.0, y: 1.75, w: 3.5, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 5.0, y: 2.2, w: 3.5, h: 2.5,
    fill: { color: C.primary }, line: { color: C.primary }, rectRadius: 0.15,
  });
  s.addText([
    { text: "OPTIMAL\nESTIMATION",
      options: { fontSize: 18, bold: true, color: "FFFFFF", breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Levenberg-Marquardt\nposterior + averaging kernel",
      options: { fontSize: 11, color: C.accent, italic: true } },
  ], { x: 5.0, y: 2.2, w: 3.5, h: 2.5,
       fontFace: FONT_HEAD, align: "center", valign: "middle", margin: 0 });

  s.addText("CONSTRAINTS", {
    x: 8.8, y: 1.75, w: 4.0, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  const constraints = [
    { label: "Density-fractionated soil ¹⁴C", color: C.accent },
    { label: "Bulk soil ¹⁴C (ISRaD layer)", color: C.accent },
    { label: "Respired CO₂ ¹⁴C", color: C.accent },
    { label: "Soil C stocks (gC m⁻²)", color: C.secondary },
    { label: "Eddy-covariance GPP / NEE", color: C.secondary },
  ];
  constraints.forEach((c, i) => {
    const y = 2.15 + i*0.45;
    s.addShape(pres.shapes.OVAL, {
      x: 8.85, y: y+0.08, w: 0.16, h: 0.16, fill: { color: c.color }, line: { color: c.color },
    });
    s.addText(c.label, {
      x: 9.15, y, w: 3.6, h: 0.35,
      fontSize: 12, fontFace: FONT_BODY, color: C.body, margin: 0,
    });
  });
  s.addShape(pres.shapes.LINE, {
    x: 4.8, y: 3.45, w: 0.2, h: 0,
    line: { color: C.secondary, width: 2.5, endArrowType: "triangle" },
  });
  s.addShape(pres.shapes.LINE, {
    x: 8.6, y: 3.45, w: 0.2, h: 0,
    line: { color: C.secondary, width: 2.5, beginArrowType: "triangle" },
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 5.2, w: 12.1, h: 1.6,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText("INFORMATION-THEORETIC ANALYSIS", {
    x: 0.6, y: 5.3, w: 12.1, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, align: "center", margin: 0,
  });
  s.addText("Fisher Information  ·  Degrees of Freedom for Signal (DFS)  ·  Posterior covariance  ·  Averaging kernel", {
    x: 0.6, y: 5.75, w: 12.1, h: 0.55,
    fontSize: 14, fontFace: FONT_HEAD, italic: true,
    color: C.primary, align: "center", margin: 0,
  });
  s.addText("Quantifies how much each observation type tells us about which parameters.", {
    x: 0.6, y: 6.3, w: 12.1, h: 0.4,
    fontSize: 12, fontFace: FONT_BODY, color: C.muted,
    align: "center", italic: true, margin: 0,
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Two ecosystems table
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "TWO ECOSYSTEMS",
    "Canonical sites: Barrow Alaska (US-A10) vs. Harvard Forest (US-Ha1)");

  const headerH = 0.55;
  const colY = 1.85;
  const col1 = { x: 0.6, w: 4.4 };
  const col2 = { x: 5.1, w: 4.0 };
  const col3 = { x: 9.2, w: 3.5 };

  s.addShape(pres.shapes.RECTANGLE, { x: col1.x, y: colY, w: col1.w, h: headerH,
    fill: { color: C.bgLight }, line: { color: C.muted, width: 0.5 } });
  s.addShape(pres.shapes.RECTANGLE, { x: col2.x, y: colY, w: col2.w, h: headerH,
    fill: { color: C.secondary }, line: { color: C.secondary } });
  s.addShape(pres.shapes.RECTANGLE, { x: col3.x, y: colY, w: col3.w, h: headerH,
    fill: { color: C.primary }, line: { color: C.primary } });
  s.addText("CHARACTERISTIC", { x: col1.x, y: colY, w: col1.w, h: headerH,
    fontSize: 10, fontFace: FONT_BODY, bold: true, color: C.secondary, charSpacing: 3,
    valign: "middle", align: "center", margin: 0 });
  s.addText("HARVARD FOREST", { x: col2.x, y: colY, w: col2.w, h: headerH,
    fontSize: 11, fontFace: FONT_BODY, bold: true, color: "FFFFFF", charSpacing: 3,
    valign: "middle", align: "center", margin: 0 });
  s.addText("BARROW (UTQIAĠVIK)", { x: col3.x, y: colY, w: col3.w, h: headerH,
    fontSize: 11, fontFace: FONT_BODY, bold: true, color: "FFFFFF", charSpacing: 3,
    valign: "middle", align: "center", margin: 0 });

  const rows = [
    ["Biome", "Temperate deciduous", "Polygonal tundra"],
    ["Latitude / climate", "42.5°N  /  MAT 8.5 °C", "71.3°N  /  MAT −11.5 °C"],
    ["Permafrost", "No", "Yes  (continuous)"],
    ["¹⁴C pool data", "Density fractionated\n(hf212-03 + ISRaD H1–H5)", "Bulk soil layer\n(ISRaD Vaughn 2018, Nave 2021)"],
    ["¹⁴C respired", "41 NWN obs, 1996–2010", "14 surface obs, 2012–2014"],
    ["Soil C stocks", "ISRaD H1–H5 layer integrated\n(O / A / B horizons)", "Ping 2008 (active)\n+ Hugelius 2014 (permafrost)"],
    ["Modeled vulnerability", "Modest — mostly modern C", "High — old, frozen C exposed"],
  ];

  let rowY = colY + headerH;
  rows.forEach((r, i) => {
    const rh = (i === 3 || i === 5) ? 0.65 : 0.5;
    const altFill = i % 2 === 0 ? "FFFFFF" : "F7F4ED";
    s.addShape(pres.shapes.RECTANGLE, {
      x: col1.x, y: rowY, w: col1.w + col2.w + col3.w + 0.2, h: rh,
      fill: { color: altFill }, line: { color: "EAEAEA", width: 0.5 },
    });
    s.addText(r[0], { x: col1.x + 0.15, y: rowY, w: col1.w - 0.3, h: rh,
      fontSize: 11, fontFace: FONT_BODY, bold: true, color: C.secondary,
      valign: "middle", margin: 0 });
    s.addText(r[1], { x: col2.x + 0.1, y: rowY, w: col2.w - 0.2, h: rh,
      fontSize: 11, fontFace: FONT_BODY, color: C.body, valign: "middle", margin: 0 });
    s.addText(r[2], { x: col3.x + 0.1, y: rowY, w: col3.w - 0.2, h: rh,
      fontSize: 11, fontFace: FONT_BODY, color: C.body, valign: "middle", margin: 0 });
    rowY += rh;
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: rowY + 0.25, w: 12.1, h: 0.7,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "Why these two sites:  ",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 2 } },
    { text: "they bracket soil-C vulnerability while sharing a 3-pool topology — enabling an apples-to-apples information comparison.",
      options: { fontSize: 12, color: C.body, italic: true } },
  ], { x: 0.8, y: rowY + 0.25, w: 11.7, h: 0.7, fontFace: FONT_BODY, valign: "middle", margin: 0 });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Sierra (merged prior + posterior)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "VALIDATION", "We reproduce — then improve on — Sierra et al. (2012) at Harvard Forest");

  s.addImage({
    path: fig("harvard_forest_radiocarbon_inversion_summary.png"),
    x: 0.6, y: 1.65, w: 8.7, h: 5.5,
    sizing: { type: "contain", w: 8.7, h: 5.5 },
  });

  // Right column: two-card layout
  // Card 1: PRIOR
  s.addShape(pres.shapes.RECTANGLE, { x: 9.5, y: 1.75, w: 3.4, h: 2.25,
    fill: { color: C.bgLight }, line: { color: C.muted, width: 0.5 } });
  s.addText("PRIOR  (Gaudinski / Sierra 2012)", {
    x: 9.6, y: 1.85, w: 3.2, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "7-pool topology, Q10 = 1, constant climate",
      options: { fontSize: 11, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "Reproduces Sierra Fig. 3 to within ~20–50 ‰",
      options: { fontSize: 11, italic: true, color: C.primary, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "Resp Δ¹⁴C RMSE 13.5 ‰",
      options: { fontSize: 11, color: C.body } },
  ], { x: 9.6, y: 2.15, w: 3.2, h: 1.85, fontFace: FONT_BODY, valign: "top", margin: 0 });

  // Card 2: POSTERIOR
  s.addShape(pres.shapes.RECTANGLE, { x: 9.5, y: 4.15, w: 3.4, h: 3.0,
    fill: { color: C.primary }, line: { color: C.primary } });
  s.addText("POSTERIOR  (OE inversion)", {
    x: 9.6, y: 4.25, w: 3.2, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "τ_dead_roots", options: { fontSize: 11, color: "FFFFFF", bold: true } },
    { text: ":  6 → 5.6 yr  (pinned)", options: { fontSize: 11, color: "FFFFFF", breakLine: true } },
    { text: "τ_ALF_lt80", options: { fontSize: 11, color: "FFFFFF", bold: true } },
    { text: ":  75 → 13 yr  ↓ 5.9×", options: { fontSize: 11, color: C.accent, breakLine: true } },
    { text: "τ_MinAss", options: { fontSize: 11, color: "FFFFFF", bold: true } },
    { text: ":  110 → 182 yr  ↑ 1.7×", options: { fontSize: 11, color: C.accent, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Bomb-spike Δ¹⁴C requires faster light fractions and older mineral-associated C than Gaudinski's prior assumed.",
      options: { fontSize: 11, italic: true, color: "FFFFFF" } },
  ], { x: 9.6, y: 4.55, w: 3.2, h: 2.5, fontFace: FONT_BODY, valign: "top", margin: 0, paraSpaceAfter: 4 });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Canonical inversions overview (both 3-pool figures)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "CANONICAL INVERSIONS",
    "3-pool model · same algorithm · same field set · both sites");

  // 8-panel OE figures are portrait (AR 0.76).  3-column layout:
  // HF (left)  ·  comparison highlights (centre)  ·  Barrow (right)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.75, w: 3.6, h: 0.45,
    fill: { color: C.secondary }, line: { color: C.secondary },
  });
  s.addText("HARVARD FOREST  ·  3-POOL OE", {
    x: 0.6, y: 1.75, w: 3.6, h: 0.45,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: "FFFFFF", charSpacing: 3, valign: "middle", align: "center", margin: 0,
  });
  s.addImage({
    path: fig("harvard_forest_three_pool_oe_summary.png"),
    x: 0.6, y: 2.25, w: 3.6, h: 4.74,
    sizing: { type: "contain", w: 3.6, h: 4.74 },
  });

  // Centre column: shared method + comparison highlights
  s.addText("SHARED ALGORITHM", {
    x: 4.45, y: 1.75, w: 4.4, h: 0.45,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, valign: "middle", align: "center", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.45, y: 2.25, w: 4.4, h: 4.74,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "Algorithm",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "optimize_oe  (Levenberg-Marquardt)\nfields = (log τ, log f_transfer)\nanalytical SS state at MAP",
      options: { fontSize: 11.5, color: C.body, fontFace: "Consolas", breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Observations",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "HF:  ",         options: { bold: true, fontSize: 11.5, color: C.primary } },
    { text: "50 obs  (41 resp + 3 C-stock + 6 ISRaD fraction Δ¹⁴C)",
      options: { fontSize: 11.5, color: C.body, breakLine: true } },
    { text: "Barrow:  ",     options: { bold: true, fontSize: 11.5, color: C.primary } },
    { text: "19 obs  (3 ISRaD layer + 14 resp + 2 C-stock)",
      options: { fontSize: 11.5, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Convergence",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "HF:  ",         options: { bold: true, fontSize: 11.5, color: C.primary } },
    { text: "26 iter  ·  J  20 098 → 72.0",
      options: { fontSize: 11.5, color: C.body, breakLine: true } },
    { text: "Barrow:  ",     options: { bold: true, fontSize: 11.5, color: C.primary } },
    { text: "7 iter  ·  J  44 340 → 26.9",
      options: { fontSize: 11.5, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Reduced χ²",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "HF:  ",         options: { bold: true, fontSize: 11.5, color: C.primary } },
    { text: "2.06",
      options: { fontSize: 11.5, color: C.body, breakLine: true } },
    { text: "Barrow:  ",     options: { bold: true, fontSize: 11.5, color: C.primary } },
    { text: "6.73",
      options: { fontSize: 11.5, color: C.body } },
  ], { x: 4.65, y: 2.4, w: 4.0, h: 4.5,
        fontFace: FONT_BODY, margin: 0, valign: "top", paraSpaceAfter: 2 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.1, y: 1.75, w: 3.6, h: 0.45,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("BARROW  ·  CANONICAL 3-POOL OE", {
    x: 9.1, y: 1.75, w: 3.6, h: 0.45,
    fontSize: 10.5, fontFace: FONT_BODY, bold: true,
    color: "FFFFFF", charSpacing: 3, valign: "middle", align: "center", margin: 0,
  });
  s.addImage({
    path: fig("barrow_canonical_information_content_analysis.png"),
    x: 9.1, y: 2.25, w: 3.6, h: 4.74,
    sizing: { type: "contain", w: 3.6, h: 4.74 },
  });

  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 11 — Model fits the data (resp residuals)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "RESULTS  ·  FIT QUALITY",
    "Posterior model fits respired Δ¹⁴C within analytical precision at both sites");

  s.addImage({
    path: fig("canonical_respired_delta14c_model_observation_fits.png"),
    x: 0.6, y: 1.8, w: 9.5, h: 4.5,
    sizing: { type: "contain", w: 9.5, h: 4.5 },
  });

  // Right: takeaway box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.3, y: 1.95, w: 2.6, h: 4.2,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText("TAKEAWAY", {
    x: 10.45, y: 2.05, w: 2.4, h: 0.3,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText([
    { text: "Both fits converged.",
      options: { fontSize: 13, bold: true, color: C.primary, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "HF:  12.6 ‰ RMSE,  −5.4 ‰ bias  (41 obs)",
      options: { fontSize: 11, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 3 } },
    { text: "Barrow:  20.2 ‰ RMSE,  −0.2 ‰ bias  (14 obs)",
      options: { fontSize: 11, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Both within typical AMS Δ¹⁴C precision (~10–20 ‰).",
      options: { fontSize: 11, italic: true, color: C.primary, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Barrow residuals are scatter — Vaughn surface flux measurements are spatially variable at decimeter scales.",
      options: { fontSize: 10, italic: true, color: C.muted } },
  ], { x: 10.45, y: 2.4, w: 2.4, h: 3.6, fontFace: FONT_BODY, valign: "top", margin: 0, paraSpaceAfter: 3 });

  // Bottom: "headline" strip
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.45, w: 12.3, h: 0.7,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("Our 3-pool model reproduces the bomb-spike decline at Harvard Forest and the muted, near-zero-trend signal at Barrow.", {
    x: 0.6, y: 6.45, w: 12.3, h: 0.7,
    fontSize: 13, fontFace: FONT_HEAD, italic: true,
    color: "FFFFFF", align: "center", valign: "middle", margin: 0,
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 12 — HEADLINE: posterior τ at both sites
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "RESULTS  ·  HEADLINE",
    "Data want older deep carbon and younger transitional carbon than the literature prior");

  s.addImage({
    path: fig("canonical_posterior_turnover_time_comparison.png"),
    x: 0.6, y: 1.75, w: 12.1, h: 4.7,
    sizing: { type: "contain", w: 12.1, h: 4.7 },
  });

  // Bottom synthesis cards (left and right)
  const synthY = 6.55;
  const synthH = 0.6;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: synthY, w: 6.0, h: synthH,
    fill: { color: C.bgLight }, line: { color: C.secondary, width: 0.75 },
  });
  s.addText([
    { text: "At HF: ", options: { bold: true, fontSize: 12, color: C.secondary } },
    { text: "active stays near prior (2.0 → 2.2 yr); slow lengthens 1.6× (20 → 31 yr); passive lengthens 3.1× (100 → 307 yr).",
      options: { fontSize: 12, color: C.body, italic: true } },
  ], { x: 0.75, y: synthY, w: 5.7, h: synthH, fontFace: FONT_BODY, valign: "middle", margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.9, y: synthY, w: 6.0, h: synthH,
    fill: { color: C.bgLight }, line: { color: C.primary, width: 0.75 },
  });
  s.addText([
    { text: "At Barrow: ", options: { bold: true, fontSize: 12, color: C.primary } },
    { text: "active lengthens 2.4× (2.0 → 4.9 yr), slow shortens to 0.57× (50 → 28.5 yr), and passive lengthens 1.6× (1000 → 1577 yr).",
      options: { fontSize: 12, color: C.body, italic: true } },
  ], { x: 7.05, y: synthY, w: 5.7, h: synthH, fontFace: FONT_BODY, valign: "middle", margin: 0 });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 13 — Averaging kernel: what each constraint does (HF deep dive)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "INFORMATION CONTENT  ·  01",
    "What does each constraint actually constrain?");

  s.addImage({
    path: fig("harvard_forest_canonical_information_content_analysis.png"),
    x: 0.6, y: 1.65, w: 8.6, h: 5.5,
    sizing: { type: "contain", w: 8.6, h: 5.5 },
  });

  s.addText("Averaging kernel (Rodgers 2000)", {
    x: 9.4, y: 1.85, w: 3.5, h: 0.45,
    fontSize: 13, fontFace: FONT_HEAD, bold: true,
    color: C.primary, margin: 0,
  });
  s.addText([
    { text: "A = (KᵀSe⁻¹K + Sa⁻¹)⁻¹ KᵀSe⁻¹K",
      options: { fontSize: 11, fontFace: "Consolas", color: C.secondary, italic: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "trace(A) = Degrees of Freedom for Signal — total information from data.",
      options: { fontSize: 11, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "diag(A) per parameter shows ",
      options: { fontSize: 11, color: C.body } },
    { text: "which τ is data-informed and which stays at the prior.",
      options: { fontSize: 11, bold: true, color: C.primary } },
  ], { x: 9.4, y: 2.35, w: 3.5, h: 2.0, fontFace: FONT_BODY, margin: 0, valign: "top", paraSpaceAfter: 4 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.4, y: 4.55, w: 3.5, h: 2.5,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText("WHICH OBS CONSTRAIN WHICH PARAMS?", {
    x: 9.55, y: 4.65, w: 3.2, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "HF:  ", options: { bold: true, fontSize: 11, color: C.secondary } },
    { text: "respired + fraction Δ¹⁴C constrain all three log τ; log f_transfer stays near the prior.",
      options: { fontSize: 10.5, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "Barrow:  ", options: { bold: true, fontSize: 11, color: C.primary } },
    { text: "pool Δ¹⁴C carries the slow/passive-age signal; respired Δ¹⁴C mainly informs active-to-slow turnover.",
      options: { fontSize: 10.5, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "C stocks:  ", options: { bold: true, fontSize: 11, color: C.body } },
    { text: "anchor active and passive pool sizes at Barrow, partly compensating for the respiration-blind frozen passive pool.",
      options: { fontSize: 10.5, color: C.body, italic: true } },
  ], { x: 9.55, y: 4.95, w: 3.2, h: 2.0, fontFace: FONT_BODY, margin: 0, valign: "top", paraSpaceAfter: 2 });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 14 — Averaging kernels of both sites
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "INFORMATION CONTENT  ·  02",
    "Averaging kernels: which turnover times the data actually resolve");

  // Wide τ-block averaging-kernel heatmaps (native AR ≈ 2.34)
  s.addImage({
    path: fig("averaging_kernel_tau_comparison.png"),
    x: 0.55, y: 1.9, w: 9.4, h: 4.02,
    sizing: { type: "contain", w: 9.4, h: 4.02 },
  });

  // Right column — how to read it
  s.addText("READING THE KERNEL", {
    x: 10.15, y: 1.95, w: 2.75, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "A = (KᵀSe⁻¹K + Sa⁻¹)⁻¹ KᵀSe⁻¹K",
      options: { fontSize: 10, fontFace: "Consolas", color: C.secondary, italic: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "Each row shows how a retrieved τ is a weighted average of the true τ's.",
      options: { fontSize: 11, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "Diagonal = DFS",
      options: { fontSize: 11, bold: true, color: C.primary } },
    { text: "  (0 = prior-dominated, 1 = fully data-resolved).",
      options: { fontSize: 11, color: C.body } },
  ], { x: 10.15, y: 2.35, w: 2.75, h: 2.0, fontFace: FONT_BODY, margin: 0, valign: "top", paraSpaceAfter: 3 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.15, y: 4.4, w: 2.75, h: 1.5,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "HF:  ", options: { bold: true, fontSize: 11, color: C.secondary } },
    { text: "τ_active 1.00, τ_slow 0.65, τ_passive 0.52.",
      options: { fontSize: 10.5, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 3 } },
    { text: "Barrow:  ", options: { bold: true, fontSize: 11, color: C.primary } },
    { text: "τ_active 0.99, τ_slow 0.50, τ_passive 0.42 — deep pools less resolved.",
      options: { fontSize: 10.5, color: C.body } },
  ], { x: 10.3, y: 4.5, w: 2.45, h: 1.35, fontFace: FONT_BODY, valign: "top", margin: 0, paraSpaceAfter: 1 });

  // Bottom insight band
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.15, w: 12.3, h: 0.95,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "The slow / passive trade-off:  ",
      options: { bold: true, fontSize: 12, color: C.secondary } },
    { text: "τ_slow and τ_passive share a negative off-diagonal (−0.12 to −0.18) — the data pin their combination better than either alone. The mixing is stronger at Barrow, where the frozen passive pool is nearly invisible to surface respiration and total DFS falls from 3.14 to 2.74.",
      options: { fontSize: 12, fontFace: FONT_HEAD, italic: true, color: C.primary } },
  ], { x: 0.75, y: 6.15, w: 12.0, h: 0.95, fontFace: FONT_BODY, valign: "middle", margin: 0 });

  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 15 — Cross-site DFS comparison
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "INFORMATION CONTENT  ·  03",
    "The utility of each radiocarbon constraint is ecosystem-specific");

  s.addImage({
    path: fig("cross_site_observation_type_dfs_comparison.png"),
    x: 0.6, y: 1.65, w: 8.7, h: 4.2,
    sizing: { type: "contain", w: 8.7, h: 4.2 },
  });

  s.addText("THE HEADLINE", {
    x: 9.5, y: 1.85, w: 3.4, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 3, margin: 0,
  });
  s.addText("HF reaches 3.14 DFS while Barrow reaches 2.74 — and the obs-type contributions differ.", {
    x: 9.5, y: 2.2, w: 3.4, h: 1.0,
    fontSize: 14, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.5, y: 3.4, w: 3.4, h: 2.4,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  const dfsRows = [
    ["",            "HF",     "Barrow"],
    ["C stocks",    "2.00",   "1.43"],
    ["Pool Δ¹⁴C",   "2.99",   "2.60"],
    ["Resp Δ¹⁴C",   "2.06",   "1.06"],
  ];
  let dy = 3.5;
  dfsRows.forEach((r, i) => {
    const isHead = i === 0;
    const txtColor = isHead ? C.accent : "FFFFFF";
    s.addText(r[0], { x: 9.65, y: dy, w: 1.5, h: 0.4,
      fontSize: 12, fontFace: FONT_BODY, bold: true, color: txtColor, valign: "middle", margin: 0 });
    s.addText(r[1], { x: 11.0, y: dy, w: 0.85, h: 0.4,
      fontSize: isHead ? 11 : 13, fontFace: FONT_BODY, bold: true,
      color: txtColor, align: "right", valign: "middle", margin: 0 });
    s.addText(r[2], { x: 11.85, y: dy, w: 0.95, h: 0.4,
      fontSize: isHead ? 11 : 13, fontFace: FONT_BODY, bold: true,
      color: txtColor, align: "right", valign: "middle", margin: 0 });
    dy += 0.45;
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.05, w: 12.3, h: 1.1,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText("WHY?", {
    x: 0.75, y: 6.1, w: 1.0, h: 0.3,
    fontSize: 12, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText("At HF, respired plus fraction Δ¹⁴C jointly constrain all three turnover times. At Barrow, pool Δ¹⁴C constrains the deep-age structure, while C stocks anchor active and passive pools because the frozen passive pool is mostly invisible to surface respiration.", {
    x: 0.75, y: 6.4, w: 12.0, h: 0.7,
    fontSize: 12.5, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 15b — Bulk + respired ¹⁴C on par with fraction ¹⁴C
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "INFORMATION CONTENT  ·  04",
    "Bulk + respired ¹⁴C is on par with density-fraction ¹⁴C");

  s.addImage({
    path: fig("bulk_respired_on_par_with_fraction_14c.png"),
    x: 0.5, y: 1.7, w: 12.3, h: 3.5,
    sizing: { type: "contain", w: 12.3, h: 3.5 },
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 5.55, w: 12.3, h: 1.55,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText("WHY IT MATTERS", {
    x: 0.75, y: 5.62, w: 2.4, h: 0.3,
    fontSize: 12, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText([
    { text: "Bulk ¹⁴C alone cannot resolve the pool structure — a 3-pool model caps it. However you weight the layers (one collapsed mixture 1.0, honest per-layer mass-weighting 1.65, or a depth→pool proxy 2.0), bulk-alone stays below fractions (2.95). ",
      options: { fontSize: 12.5 } },
    { text: "But bulk is stock-weighted (slow + passive) and respiration flux-weighted (active), so ",
      options: { fontSize: 12.5 } },
    { text: "bulk + respired ≈ fractions (≈3.0). ",
      options: { fontSize: 12.5, bold: true, color: C.secondary } },
    { text: "That matters because density fractionation is rare — and absent for permafrost (Barrow), where bulk + respired is the only option and still nearly saturates (2.86).",
      options: { fontSize: 12.5 } },
  ], {
    x: 0.75, y: 5.95, w: 12.0, h: 1.1,
    fontFace: FONT_HEAD, italic: true, color: C.primary, margin: 0, valign: "top",
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 16 — Complexity ladder: diminishing returns
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "INFORMATION CONTENT  ·  05",
    "Information saturates at ~3 DFS — matched to the 3-pool model dimensionality");

  // complexity_ladder native AR 2.62 — size container to match (8.5 × 3.24)
  s.addImage({
    path: fig("model_complexity_information_saturation.png"),
    x: 0.6, y: 2.0, w: 8.5, h: 3.24,
    sizing: { type: "contain", w: 8.5, h: 3.24 },
  });

  // Right column: implications
  s.addText("WHAT THIS MEANS", {
    x: 9.3, y: 1.95, w: 3.55, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText([
    { text: "C stocks alone get to 1.9 DFS",
      options: { bullet: true, fontSize: 12, color: C.body, breakLine: true } },
    { text: "Adding pool Δ¹⁴C pushes both sites to ≈ 3 DFS",
      options: { bullet: true, fontSize: 12, color: C.body, breakLine: true } },
    { text: "Adding respired Δ¹⁴C adds only ~0.1 DFS",
      options: { bullet: true, fontSize: 12, color: C.body, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "Saturation at 3 DFS = full constraint of a 3-pool τ vector.",
      options: { fontSize: 12, italic: true, color: C.primary } },
  ], { x: 9.3, y: 2.35, w: 3.55, h: 2.6, fontFace: FONT_BODY, valign: "top", margin: 0, paraSpaceAfter: 5 });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.3, y: 5.3, w: 3.55, h: 1.5,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("THE OPPORTUNITY", {
    x: 9.45, y: 5.4, w: 3.25, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 3, margin: 0,
  });
  s.addText("Adding model complexity (4-pool, depth-resolved) only helps if it adds new resolved dimensions.", {
    x: 9.45, y: 5.7, w: 3.25, h: 1.0,
    fontSize: 11, fontFace: FONT_HEAD, italic: true,
    color: "FFFFFF", margin: 0, valign: "top",
  });

  // Bottom synthesis
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.5, w: 12.3, h: 0.7,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "Implication for model design  ",
      options: { bold: true, fontSize: 12, color: C.secondary, charSpacing: 2 } },
    { text: " — a 3-pool model is the right complexity for these data.  Higher-resolution models will be prior-dominated unless paired with new observation types.",
      options: { fontSize: 12, color: C.body, italic: true } },
  ], { x: 0.75, y: 6.5, w: 12.0, h: 0.7, fontFace: FONT_BODY, valign: "middle", margin: 0 });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 16 — CESM2 / CLM5 vs OE posterior  (NEW RESULT)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "RESULTS  ·  CLIMATE MODEL COMPARISON",
    "CLM5 in our framework: the fast pool is too fast, while slow and deep pools are closer but still biased low");

  s.addImage({
    path: fig("clm/clm_emulator_posterior.png"),
    x: 0.6, y: 1.65, w: 9.5, h: 5.0,
    sizing: { type: "contain", w: 9.5, h: 5.0 },
  });

  // Right-side method
  s.addText("METHOD", {
    x: 10.3, y: 1.85, w: 2.6, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText("Treat CLM5 stocks + total Rh as observations. Fit our 3-pool model with optimize_oe. Result is an emulated CLM5 τ vector directly comparable to the ¹⁴C-constrained posterior.", {
    x: 10.3, y: 2.2, w: 2.6, h: 1.85,
    fontSize: 11, fontFace: FONT_BODY, italic: true,
    color: C.body, margin: 0, valign: "top",
  });

  // Headline numbers — per-pool ratios
  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.3, y: 4.1, w: 2.6, h: 1.8,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "CLM5 / ¹⁴C  τ RATIOS", options: { bold: true, fontSize: 10, color: C.secondary,
      charSpacing: 3, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "active:  ",  options: { bold: true, fontSize: 11, color: C.body } },
    { text: "0.40× HF · 0.29× Barrow", options: { fontSize: 11, color: "B85042", breakLine: true } },
    { text: "slow:    ", options: { bold: true, fontSize: 11, color: C.body } },
    { text: "0.64× HF · 0.95× Barrow", options: { fontSize: 11, color: C.primary, breakLine: true } },
    { text: "passive: ", options: { bold: true, fontSize: 11, color: C.body } },
    { text: "0.78× HF · 0.62× Barrow", options: { fontSize: 11, color: C.primary } },
  ], { x: 10.45, y: 4.2, w: 2.4, h: 1.6, fontFace: FONT_BODY,
        valign: "top", margin: 0, paraSpaceAfter: 1 });

  // Implication
  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.3, y: 5.95, w: 2.6, h: 1.0,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("THE FINDING", {
    x: 10.45, y: 6.0, w: 2.4, h: 0.25,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 3, margin: 0,
  });
  s.addText("CLM5 is systematically too fast, with the strongest bias in the active pool; slow and passive τ are closer but still low at both sites.", {
    x: 10.45, y: 6.25, w: 2.4, h: 0.7,
    fontSize: 10.5, fontFace: FONT_HEAD, italic: true,
    color: "FFFFFF", margin: 0, valign: "top",
  });

  // Provenance footer
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 7.05, w: 12.3, h: 0.3,
    fill: { color: C.bgLight }, line: { color: C.muted, width: 0.5 },
  });
  s.addText("CESM2 historical r1i1p1f1, 2005–2014 means.  HF cell (42.88°N, −72.50°E)  ·  Barrow inland tundra cell (68.32°N, −150.00°E).  3-pool model forced with FluxNet climate (CLM5-GPP-forced run planned).", {
    x: 0.6, y: 7.05, w: 12.3, h: 0.3,
    fontSize: 9, fontFace: FONT_BODY, italic: true,
    color: C.muted, align: "center", valign: "middle", margin: 0,
  });

  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 17 — CLM5 prediction vs ¹⁴C observations  (the punchline)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "RESULTS  ·  CLM5 vs. ¹⁴C OBSERVATIONS",
    "Does CLM5 actually reproduce observed respired Δ¹⁴C?");

  s.addImage({
    path: fig("clm/clm_emulator_14c_mismatch.png"),
    x: 0.6, y: 1.65, w: 9.5, h: 4.9,
    sizing: { type: "contain", w: 9.5, h: 4.9 },
  });

  // Right-side findings
  s.addText("TEST  (held-out validation)", {
    x: 10.3, y: 1.85, w: 2.6, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText("Fit τ to CLM5 stocks + Rh only (no ¹⁴C).  Forward-run with that τ.  Predict respired Δ¹⁴C and compare to held-out observations.", {
    x: 10.3, y: 2.2, w: 2.6, h: 1.7,
    fontSize: 11, fontFace: FONT_BODY, italic: true,
    color: C.body, margin: 0, valign: "top",
  });

  // Numbers card — HF
  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.3, y: 4.0, w: 2.6, h: 1.25,
    fill: { color: C.bgLight }, line: { color: C.secondary, width: 1 },
  });
  s.addText("HARVARD FOREST", {
    x: 10.45, y: 4.08, w: 2.4, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "OE (¹⁴C-fit):  ", options: { bold: true, fontSize: 11, color: C.body } },
    { text: "RMSE 12.6 ‰", options: { fontSize: 11, color: C.primary, breakLine: true } },
    { text: "CLM-emulator:  ", options: { bold: true, fontSize: 11, color: C.body } },
    { text: "RMSE 21.0 ‰", options: { fontSize: 11, color: C.primary, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "CLM5 is too modern even at HF.", options: { fontSize: 10.5, italic: true, color: C.primary } },
  ], { x: 10.45, y: 4.4, w: 2.4, h: 0.9, fontFace: FONT_BODY, margin: 0,
        valign: "top", paraSpaceAfter: 1 });

  // Numbers card — Barrow
  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.3, y: 5.4, w: 2.6, h: 1.45,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("BARROW TUNDRA", {
    x: 10.45, y: 5.48, w: 2.4, h: 0.3,
    fontSize: 10, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 3, margin: 0,
  });
  s.addText([
    { text: "OE (¹⁴C-fit):  ", options: { bold: true, fontSize: 11, color: "FFFFFF" } },
    { text: "RMSE 20.2 ‰  bias −0.2 ‰", options: { fontSize: 10.5, color: "FFFFFF", breakLine: true } },
    { text: "CLM-emulator:  ", options: { bold: true, fontSize: 11, color: "FFFFFF" } },
    { text: "RMSE 56.9 ‰  bias ", options: { fontSize: 10.5, color: "FFFFFF" } },
    { text: "+53.3 ‰", options: { bold: true, fontSize: 11, color: C.accent, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 4 } },
    { text: "CLM5 fails badly — respired CO₂ is 53 ‰ too modern.",
      options: { fontSize: 10.5, italic: true, color: C.accent } },
  ], { x: 10.45, y: 5.8, w: 2.4, h: 1.0, fontFace: FONT_BODY, margin: 0,
        valign: "top", paraSpaceAfter: 1 });

  // Bottom insight band
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.75, w: 12.3, h: 0.6,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText([
    { text: "Why the asymmetry?  ", options: { bold: true, fontSize: 12, color: C.secondary } },
    { text: "At HF, stocks + Rh + ¹⁴C are mutually consistent — any subset finds the same τ. At Barrow, stocks + Rh leave τ ambiguous in directions ¹⁴C cares about; CLM5's tuning lands in the wrong corner of τ-space.",
      options: { fontSize: 12, fontFace: FONT_HEAD, italic: true, color: C.primary } },
  ], { x: 0.75, y: 6.75, w: 12.0, h: 0.6, fontFace: FONT_BODY, valign: "middle", margin: 0 });

  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 18 — Future experiments
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "FUTURE WORK", "Four experiments that complete the story");

  const cards = [
    {
      n: "01",
      title: "Spatial extension to 10–20 ISRaD sites",
      desc: "Apply canonical pipeline along climate / biome gradient. Map posterior τ vs MAT, MAP, % permafrost. Tests whether the HF / Barrow contrast generalizes.",
      tag: "generalizes Q1",
    },
    {
      n: "02",
      title: "Warming-scenario forecast",
      desc: "Project posterior model forward to 2100 with +2 °C, +4 °C scenarios. Compare to CMIP6 land outputs at the same grid cells.",
      tag: "directly motivates",
    },
    {
      n: "03",
      title: "Process attribution of Rh source",
      desc: "Decompose simulated Rh by pool over time at both sites. Quantifies how much of the modern flux comes from pre-bomb carbon — the deep-stock question of Q1.",
      tag: "diagnoses mechanism",
    },
    {
      n: "04",
      title: "Extend CESM comparison to SSP scenarios",
      desc: "Pull CESM2 ssp585 (2015–2100) and ssp245 stocks/fluxes. Compare projected τ trajectories to OE posterior + spatial extension. Quantify CLM5 bias under warming.",
      tag: "completes Q3",
    },
  ];

  const cardW = 6.0, cardH = 2.4, gapX = 0.3, gapY = 0.25;
  cards.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const cx = 0.6 + col * (cardW + gapX);
    const cy = 1.75 + row * (cardH + gapY);

    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cy, w: cardW, h: cardH,
      fill: { color: "FFFFFF" }, line: { color: C.secondary, width: 0.75 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cy, w: 0.85, h: cardH,
      fill: { color: C.primary }, line: { color: C.primary },
    });
    s.addText(c.n, {
      x: cx, y: cy, w: 0.85, h: cardH,
      fontSize: 36, fontFace: FONT_HEAD, bold: true,
      color: C.accent, align: "center", valign: "middle", margin: 0,
    });
    s.addText(c.title, {
      x: cx + 1.0, y: cy + 0.15, w: cardW - 1.2, h: 0.5,
      fontSize: 15, fontFace: FONT_HEAD, bold: true,
      color: C.primary, margin: 0, valign: "top",
    });
    s.addText(c.desc, {
      x: cx + 1.0, y: cy + 0.65, w: cardW - 1.2, h: 1.4,
      fontSize: 11, fontFace: FONT_BODY, color: C.body,
      margin: 0, valign: "top", italic: true,
    });
    s.addText(c.tag, {
      x: cx + 1.0, y: cy + cardH - 0.4, w: cardW - 1.2, h: 0.3,
      fontSize: 9, fontFace: FONT_BODY, bold: true,
      color: C.accent, charSpacing: 4, margin: 0,
    });
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 19 — Conclusions
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bgDark };
  n++;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.25, h: H, fill: { color: C.accent }, line: { color: C.accent },
  });
  s.addText("CONCLUSIONS", {
    x: 0.9, y: 0.6, w: 12.0, h: 0.4,
    fontSize: 13, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 6, margin: 0,
  });
  s.addText("Four takeaways", {
    x: 0.9, y: 1.0, w: 12.0, h: 0.7,
    fontSize: 32, fontFace: FONT_HEAD, italic: true,
    color: "FFFFFF", margin: 0,
  });

  const concl = [
    {
      n: "01",
      bold: "Information content of ¹⁴C is ecosystem-specific.",
      sub: "Respired Δ¹⁴C carries 2.06 DFS at HF but only 1.06 at Barrow — the frozen passive pool is invisible to surface respiration.",
    },
    {
      n: "02",
      bold: "Data wants older deep C and younger transitional C.",
      sub: "Posterior τ_passive increases 3.1× at HF and 1.6× at Barrow; τ_slow shortens to 0.57× at Barrow. Literature priors are site-biased.",
    },
    {
      n: "03",
      bold: "Total information saturates at ≈ 3 DFS.",
      sub: "Adding observation types beyond C stocks + pool Δ¹⁴C yields diminishing returns. A 3-pool model is the right complexity for available data.",
    },
    {
      n: "04",
      bold: "CLM5 is too modern at both sites, and fails badly at Barrow.",
      sub: "Fit our 3-pool model to CLM5 stocks + Rh (no ¹⁴C); hold out observed respired Δ¹⁴C as validation. HF RMSE 21.0 ‰, Barrow RMSE 56.9 ‰ with +53.3 ‰ bias. The Barrow mismatch is especially severe where ¹⁴C resolves otherwise ambiguous τ directions.",
    },
  ];

  const cardW = 5.95, cardH = 2.5, gapX = 0.4, gapY = 0.3;
  concl.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const cx = 0.9 + col * (cardW + gapX);
    const cy = 2.0 + row * (cardH + gapY);

    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cy, w: cardW, h: cardH,
      fill: { color: "FFFFFF" }, line: { color: C.accent, width: 0.5 },
    });
    s.addText(c.n, {
      x: cx + 0.2, y: cy + 0.2, w: 1.2, h: 1.2,
      fontSize: 44, fontFace: FONT_HEAD, bold: true,
      color: C.accent, margin: 0,
    });
    s.addText(c.bold, {
      x: cx + 1.5, y: cy + 0.25, w: cardW - 1.7, h: 1.0,
      fontSize: 15, fontFace: FONT_HEAD, bold: true,
      color: C.primary, margin: 0, valign: "top",
    });
    s.addText(c.sub, {
      x: cx + 1.5, y: cy + 1.3, w: cardW - 1.7, h: 1.0,
      fontSize: 11, fontFace: FONT_BODY, color: C.body,
      margin: 0, valign: "top", italic: true,
    });
  });

  s.addText("Newton H. Nguyen  ·  nnewton@stanford.edu", {
    x: 0.9, y: 7.0, w: 12.0, h: 0.3,
    fontSize: 11, fontFace: FONT_BODY, color: C.accent, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 22 — Multi-site extension: observation utility across 14 ecosystems
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "MULTI-SITE  ·  OBSERVATION UTILITY",
    "One recipe, 14 ecosystems — turnover is never fully identified");

  s.addImage({
    path: fig("fig08_panelC_dfs_by_site.png"),
    x: 0.55, y: 1.65, w: 8.8, h: 4.25,
    sizing: { type: "contain", w: 8.8, h: 4.25 },
  });

  s.addText("THE HEADLINE", {
    x: 9.55, y: 1.85, w: 3.4, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 3, margin: 0,
  });
  s.addText("Stacking observation types lifts the fast pool — but never the passive pool.", {
    x: 9.55, y: 2.2, w: 3.4, h: 1.0,
    fontSize: 14, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.55, y: 3.45, w: 3.4, h: 2.35,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("τ uncertainty reduction (MO Ozark)", {
    x: 9.7, y: 3.53, w: 3.15, h: 0.3,
    fontSize: 9.5, fontFace: FONT_BODY, bold: true, color: C.accent, margin: 0,
  });
  const urRows = [
    ["",          "Frac", "+Bulk", "+Stk"],
    ["τ active",  "0.68", "0.74",  "0.82"],
    ["τ slow",    "0.46", "0.46",  "0.45"],
    ["τ passive", "0.27", "0.27",  "0.27"],
  ];
  let uy = 3.95;
  urRows.forEach((r, i) => {
    const isHead = i === 0;
    const isPass = r[0] === "τ passive";
    const tc = isHead ? C.accent : (isPass ? C.accent : "FFFFFF");
    s.addText(r[0], { x: 9.68, y: uy, w: 1.1, h: 0.4,
      fontSize: 12, fontFace: FONT_BODY, bold: true, color: tc, valign: "middle", margin: 0 });
    [1, 2, 3].forEach((k) => {
      s.addText(r[k], { x: 10.78 + (k - 1) * 0.68, y: uy, w: 0.66, h: 0.4,
        fontSize: isHead ? 9.5 : 12.5, fontFace: FONT_BODY, bold: true,
        color: tc, align: "right", valign: "middle", margin: 0 });
    });
    uy += 0.44;
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.05, w: 12.35, h: 1.1,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText("WHY?", {
    x: 0.75, y: 6.1, w: 1.0, h: 0.3,
    fontSize: 12, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText("New sites (UMBS, Ozark, Bartlett) run a combined pathway — density-fraction + bulk-layer depth Δ¹⁴C + measured SOC stocks. Across all 14 ecosystems turnover DFS saturates near 2.1–2.5, never reaching 3: the century-scale passive pool is structurally unidentifiable from single-epoch ¹⁴C, because τ_passive trades off with the slow→passive transfer flux. Even a −272 ‰ deep layer plus a tight measured stock leaves passive uncertainty reduction at ≈ 0.27.", {
    x: 0.75, y: 6.4, w: 12.05, h: 0.72,
    fontSize: 11.5, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 23 — GPP time structure carries almost no turnover information
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "FORCING  ·  GPP TIME STRUCTURE",
    "GPP's seasonal structure carries almost no turnover information");

  s.addImage({
    path: fig("harvard_forest_gpp_ladder.png"),
    x: 0.55, y: 1.65, w: 8.8, h: 4.25,
    sizing: { type: "contain", w: 8.8, h: 4.25 },
  });

  s.addText("THE HEADLINE", {
    x: 9.55, y: 1.85, w: 3.4, h: 0.35,
    fontSize: 11, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 3, margin: 0,
  });
  s.addText("Same mean GPP, with vs without seasonal structure: identical information.", {
    x: 9.55, y: 2.2, w: 3.4, h: 1.0,
    fontSize: 14, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.55, y: 3.45, w: 3.4, h: 2.35,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("Harvard Forest — tower vs mean GPP", {
    x: 9.7, y: 3.53, w: 3.15, h: 0.3,
    fontSize: 9.5, fontFace: FONT_BODY, bold: true, color: C.accent, margin: 0,
  });
  const gppRows = [
    ["",           "Tower", "Mean"],
    ["Total DFS",  "3.14",  "3.14"],
    ["UR τ-act",   "98%",   "98%"],
    ["UR τ-slow",  "41%",   "41%"],
    ["UR τ-pass",  "31%",   "31%"],
  ];
  let gy = 3.9;
  gppRows.forEach((r, i) => {
    const isHead = i === 0;
    const tc = isHead ? C.accent : "FFFFFF";
    s.addText(r[0], { x: 9.68, y: gy, w: 1.6, h: 0.36,
      fontSize: 12, fontFace: FONT_BODY, bold: true, color: tc, valign: "middle", margin: 0 });
    s.addText(r[1], { x: 11.3, y: gy, w: 0.8, h: 0.36,
      fontSize: isHead ? 10.5 : 12.5, fontFace: FONT_BODY, bold: true,
      color: tc, align: "right", valign: "middle", margin: 0 });
    s.addText(r[2], { x: 12.1, y: gy, w: 0.75, h: 0.36,
      fontSize: isHead ? 10.5 : 12.5, fontFace: FONT_BODY, bold: true,
      color: tc, align: "right", valign: "middle", margin: 0 });
    gy += 0.38;
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 6.05, w: 12.35, h: 1.1,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 1 },
  });
  s.addText("WHY?", {
    x: 0.75, y: 6.1, w: 1.0, h: 0.3,
    fontSize: 12, fontFace: FONT_BODY, bold: true,
    color: C.secondary, charSpacing: 4, margin: 0,
  });
  s.addText("Both treatments share the same record-mean GPP, so the steady-state operating point is identical; only the transient trajectory differs. The near-zero ΔDFS (+0.003 over all observations) shows turnover information comes from the mean carbon-input magnitude and the ¹⁴C + stock data — not the seasonal or interannual structure of GPP. Total information saturates near 3 DFS, so a 3-pool model is the right complexity for single-epoch data.", {
    x: 0.75, y: 6.4, w: 12.05, h: 0.72,
    fontSize: 11.5, fontFace: FONT_HEAD, italic: true,
    color: C.primary, margin: 0, valign: "top",
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 24 — Extended conclusions (multi-site + combined-obs synthesis)
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bgDark };
  n++;
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.25, h: H, fill: { color: C.accent }, line: { color: C.accent },
  });
  s.addText("EXTENDED CONCLUSIONS", {
    x: 0.9, y: 0.6, w: 12.0, h: 0.4,
    fontSize: 13, fontFace: FONT_BODY, bold: true,
    color: C.accent, charSpacing: 6, margin: 0,
  });
  s.addText("What the multi-site + combined-observation analysis adds", {
    x: 0.9, y: 1.0, w: 12.0, h: 0.7,
    fontSize: 28, fontFace: FONT_HEAD, italic: true,
    color: "FFFFFF", margin: 0,
  });

  const ext = [
    {
      n: "05",
      bold: "Passive turnover is prior-dominated in every biome.",
      sub: "Across 14 ecosystems the passive-pool uncertainty reduction sits at ≈ 0.25–0.30. Adding deep depleted bulk-layer Δ¹⁴C (−272 ‰) and tight measured SOC stocks moves it < 0.01 — τ_passive trades off with the slow→passive transfer flux, an identifiability limit, not a data-volume one.",
    },
    {
      n: "06",
      bold: "Observation value is complementarity-driven — and capped.",
      sub: "Combining soil and respired Δ¹⁴C helps only where they constrain different modes (Harvard +0.20) and is redundant elsewhere (Howland −0.04). Combined value tops out near 0.8–0.9 and never reaches 1, because the deep pool that governs century-scale loss stays unconstrained.",
    },
    {
      n: "07",
      bold: "GPP time-structure is information-free; ≈ 3 DFS is the ceiling.",
      sub: "Time-varying vs record-mean GPP give identical information (ΔDFS +0.003). Turnover information lives in the mean operating point and the ¹⁴C/stock data. Breaking the passive degeneracy needs multi-decadal or pre-bomb Δ¹⁴C — not more observation types at one epoch.",
    },
  ];

  const cardW = 12.35, cardH = 1.5, gapY = 0.24;
  ext.forEach((c, i) => {
    const cy = 2.0 + i * (cardH + gapY);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.9, y: cy, w: cardW, h: cardH,
      fill: { color: "FFFFFF" }, line: { color: C.accent, width: 0.5 },
    });
    s.addText(c.n, {
      x: 1.1, y: cy + 0.2, w: 1.3, h: 1.1,
      fontSize: 40, fontFace: FONT_HEAD, bold: true,
      color: C.accent, margin: 0, valign: "top",
    });
    s.addText(c.bold, {
      x: 2.5, y: cy + 0.18, w: cardW - 1.8, h: 0.4,
      fontSize: 15, fontFace: FONT_HEAD, bold: true,
      color: C.primary, margin: 0, valign: "top",
    });
    s.addText(c.sub, {
      x: 2.5, y: cy + 0.62, w: cardW - 1.8, h: 0.82,
      fontSize: 11, fontFace: FONT_BODY, color: C.body,
      margin: 0, valign: "top", italic: true,
    });
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 25 — Cross-ecosystem turnover and vulnerability
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "CROSS-ECOSYSTEM RESULTS  ·  01", "Ecosystem type shapes vulnerability; observations shape constrainability");
  s.addImage({
    path: fig("paper_figs/outputs/current_results/figures/figure_09.png"),
    x: 0.55, y: 1.45, w: 8.75, h: 5.45,
    sizing: { type: "contain", w: 8.75, h: 5.45 },
  });
  s.addText("WHAT FIGURE 9 SHOWS", {
    x: 9.55, y: 1.65, w: 3.0, h: 0.32,
    fontSize: 11, fontFace: FONT_BODY, bold: true, color: C.secondary, charSpacing: 2, margin: 0,
  });
  s.addText("All ecosystems occupy the same multi-pool turnover space, but warming loss is not explained by total DFS alone.", {
    x: 9.55, y: 2.05, w: 3.0, h: 1.0,
    fontSize: 15, fontFace: FONT_HEAD, bold: true, color: C.primary, margin: 0, valign: "top",
  });
  s.addText([
    { text: "• ", options: { bold: true, color: C.accent } },
    { text: "Boreal ecosystems have the largest mean fractional C loss.\n", options: { breakLine: true } },
    { text: "• ", options: { bold: true, color: C.accent } },
    { text: "Arctic/permafrost systems have the oldest excess respiration.\n", options: { breakLine: true } },
    { text: "• ", options: { bold: true, color: C.accent } },
    { text: "C stocks, bulk ¹⁴C, and respired ¹⁴C contribute differently across biomes." },
  ], {
    x: 9.55, y: 3.35, w: 3.0, h: 1.7,
    fontSize: 12.5, fontFace: FONT_BODY, color: C.body, margin: 0, valign: "top", paraSpaceAfter: 7,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.55, y: 5.45, w: 3.0, h: 1.05,
    fill: { color: C.bgLight }, line: { color: C.accent, width: 0.8 },
  });
  s.addText("Constrainability is a property of the observation set—not a proxy for vulnerability.", {
    x: 9.73, y: 5.66, w: 2.65, h: 0.6,
    fontSize: 12.5, fontFace: FONT_HEAD, italic: true, color: C.primary, margin: 0, valign: "middle",
  });
  addPageNumber(s, n, TOTAL);
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 26 — Turnover separation and carbon vulnerability
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  n++;
  addSlideHeader(s, "CROSS-ECOSYSTEM RESULTS  ·  02", "Separated turnover pools mobilize older—not necessarily more—carbon");
  s.addImage({
    path: fig("paper_figs/outputs/current_results/figures/figure_10.png"),
    x: 0.45, y: 1.45, w: 9.45, h: 5.45,
    sizing: { type: "contain", w: 9.45, h: 5.45 },
  });
  s.addText("WHAT FIGURE 10 SHOWS", {
    x: 10.0, y: 1.65, w: 2.8, h: 0.32,
    fontSize: 11, fontFace: FONT_BODY, bold: true, color: C.secondary, charSpacing: 2, margin: 0,
  });
  s.addText("The passive-to-active turnover ratio is a useful complexity axis: it predicts the age of warming-enhanced respiration, but not the largest fractional loss.", {
    x: 10.0, y: 2.05, w: 2.8, h: 1.35,
    fontSize: 14.5, fontFace: FONT_HEAD, bold: true, color: C.primary, margin: 0, valign: "top",
  });
  s.addText([
    { text: "• ", options: { bold: true, color: C.accent } },
    { text: "Pool separation and old-RH share correlate strongly (ρₛ = 0.74).\n", options: { breakLine: true } },
    { text: "• ", options: { bold: true, color: C.accent } },
    { text: "Pool separation and fractional C loss anticorrelate (ρₛ = −0.76).\n", options: { breakLine: true } },
    { text: "• ", options: { bold: true, color: C.accent } },
    { text: "Total DFS is nearly independent of turnover separation." },
  ], {
    x: 10.0, y: 3.75, w: 2.8, h: 1.55,
    fontSize: 12, fontFace: FONT_BODY, color: C.body, margin: 0, valign: "top", paraSpaceAfter: 7,
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.0, y: 5.65, w: 2.8, h: 0.85,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  s.addText("Complexity shifts the source of vulnerability toward older C; it does not by itself determine its magnitude.", {
    x: 10.16, y: 5.82, w: 2.45, h: 0.52,
    fontSize: 11.5, fontFace: FONT_HEAD, italic: true, color: "FFFFFF", margin: 0, valign: "middle",
  });
  addPageNumber(s, n, TOTAL);
}

pres.writeFile({ fileName: fig("ecosystem_complexity_soil_carbon_canonical_presentation.pptx") })
  .then((name) => console.log("Wrote:", name));
