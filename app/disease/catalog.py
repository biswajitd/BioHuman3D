"""
Disease catalogue: conditions, their accepted staging systems, and the
anatomical change each stage produces.

Accuracy rules this file follows:

* every condition is staged with the system clinicians actually use (KDIGO,
  GOLD, Kellgren–Lawrence, METAVIR, WHO T-score, ACC/AHA …) and each stage
  states its defining criteria with the published thresholds;
* the visual change of each stage is the gross-pathology change described for
  that stage (size, colour, surface, focal lesion, luminal narrowing) — scaled
  to what is visible at whole-organ level, never exaggerated into a different
  disease;
* narration is educational, sourced from the guideline named in ``sources``,
  and every tour closes with the reminder that diagnosis and treatment belong
  to a clinician.

Effects are declarative so the same stage applies to the reference body and to
an imported atlas: structures are matched by name keywords, and focal lesions
are positioned as fractions of the target structure's bounding box
(x: patient's right → left, y: anterior → posterior, z: inferior → superior) and
then snapped onto the structure's surface, so a plaque sits on the artery wall
and an osteophyte on the bone margin whatever the mesh.

Effect kinds:
    tint     {match, color, amount}                 colour shift (0–1)
    scale    {match, factor}                        size change about the centroid
    nodular  {match, amplitude, frequency}          irregular / nodular surface (metres)
    pinch    {match, at, radius, amount}            local luminal narrowing (0–1)
    lesion   {match, at, radii, color, opacity, inset}  focal lesion (plaque, infarct, osteophyte …);
                                                    inset pulls it into the organ by inset × radius
    opacity  {match, value}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Effect = Dict[str, object]


@dataclass(frozen=True)
class Stage:
    label: str                       # "G3b"
    title: str                       # "Moderately to severely decreased GFR"
    criteria: str                    # "eGFR 30–44 mL/min/1.73 m²"
    narration: str
    effects: Tuple[Effect, ...] = ()


@dataclass(frozen=True)
class Condition:
    id: str
    name: str
    system: str
    organ_layers: Tuple[str, ...]    # layers to show solid during the simulation
    focus: Tuple[str, ...]           # structure keywords the camera frames
    view: str
    staging: str                     # staging system name
    summary: str
    stages: Tuple[Stage, ...]
    sources: Tuple[str, ...]
    context_layers: Tuple[str, ...] = ("skeleton",)
    organ_opacity: float = 1.0       # < 1 when lesions lie inside the organ
    isolate: bool = False            # ghost every structure not involved


# Colours used across conditions
FAT_YELLOW = (0.86, 0.72, 0.36)
INFLAMED = (0.80, 0.30, 0.25)
FIBROTIC = (0.62, 0.50, 0.38)
CIRRHOTIC = (0.66, 0.52, 0.30)
PALE = (0.86, 0.78, 0.72)
PLAQUE = (0.95, 0.86, 0.45)
THROMBUS = (0.45, 0.04, 0.06)
INFARCT_ACUTE = (0.88, 0.80, 0.70)
SCAR = (0.92, 0.90, 0.86)
ISCHAEMIC = (0.55, 0.48, 0.60)
GREY_LUNG = (0.62, 0.58, 0.60)
OSTEOPHYTE = (0.95, 0.93, 0.82)
SCLEROTIC = (0.98, 0.98, 0.92)

LAD = ("left anterior descending", "anterior interventricular")
LV = ("left ventricle",)
CORONARY = ("coronary", "anterior descending", "circumflex", "anterior interventricular")
LIVER = ("lobe of liver", "liver")
KIDNEY = ("kidney",)
LUNG = ("lung",)

DISCLAIMER = ("This is an educational simulation of typical anatomy, not a diagnosis. "
              "Decisions about tests and treatment belong to a qualified clinician.")


def _e(kind: str, match: Sequence[str], **params) -> Effect:
    return dict(kind=kind, match=tuple(match), **params)


CONDITIONS: Tuple[Condition, ...] = (
    # =====================================================================
    Condition(
        id="cad_mi", name="Coronary artery disease → myocardial infarction", system="Cardiovascular",
        organ_layers=("heart",), focus=("ventricle", "atrium", "coronary", "descending"), view="anterior",
        staging="AHA (Stary) lesion types; Fourth Universal Definition of Myocardial Infarction",
        summary=("Atherosclerosis narrows the coronary arteries over decades. When a plaque ruptures, "
                 "a thrombus can block the artery and the heart muscle it supplies dies."),
        stages=(
            Stage("0", "Healthy coronary arteries", "No atherosclerotic lesion",
                  "Here is a healthy heart. The left anterior descending artery runs down the front of the "
                  "heart in the groove between the ventricles, supplying the anterior wall of the left "
                  "ventricle and most of the septum."),
            Stage("I–II", "Fatty streak", "AHA type I–II: lipid-laden foam cells in the intima",
                  "Atherosclerosis begins early, often in the teens or twenties. Lipid-laden macrophages "
                  "collect in the arterial wall as a fatty streak. It does not narrow the lumen and causes "
                  "no symptoms.",
                  (_e("lesion", LAD, id="plaque", at=(0.5, 0.25, 0.80), radii=(0.0025, 0.0025, 0.004),
                      color=PLAQUE, opacity=0.9),)),
            Stage("IV–V", "Fibroatheroma, about 50% stenosis",
                  "AHA type IV–V: lipid core with a fibrous cap",
                  "Over years a lipid core forms under a fibrous cap. The plaque now narrows the artery by "
                  "about half. Blood flow at rest is usually still adequate, so many people remain "
                  "symptom-free.",
                  (_e("lesion", LAD, id="plaque", at=(0.5, 0.25, 0.80), radii=(0.0040, 0.0040, 0.0065),
                      color=PLAQUE, opacity=0.95),
                   _e("pinch", LAD, id="stenosis", at=(0.5, 0.25, 0.80), radius=0.008, amount=0.5))),
            Stage("CCS", "Severe stenosis — stable angina", "≥70% luminal narrowing",
                  "When narrowing exceeds about seventy percent, flow cannot rise to meet demand. Exertion "
                  "brings on chest pain or tightness that settles with rest: stable angina. This is when "
                  "tests such as a stress study or coronary angiography are usually considered.",
                  (_e("lesion", LAD, id="plaque", at=(0.5, 0.25, 0.80), radii=(0.0050, 0.0050, 0.0085),
                      color=PLAQUE, opacity=1.0),
                   _e("pinch", LAD, id="stenosis", at=(0.5, 0.25, 0.80), radius=0.010, amount=0.75),
                   _e("tint", LV, id="ischaemia", color=ISCHAEMIC, amount=0.15))),
            Stage("Type 1 MI", "Plaque rupture and acute anterior infarction",
                  "Rise and fall of cardiac troponin with symptoms, ECG changes or imaging evidence",
                  "If the fibrous cap ruptures, a thrombus forms within minutes and can block the artery "
                  "completely. The anterior wall of the left ventricle loses its supply and begins to die. "
                  "This is a medical emergency: every minute of delay to reopening the artery means more "
                  "lost heart muscle.",
                  (_e("lesion", LAD, id="plaque", at=(0.5, 0.25, 0.80), radii=(0.0050, 0.0050, 0.0085),
                      color=PLAQUE, opacity=1.0),
                   _e("lesion", LAD, id="thrombus", at=(0.5, 0.22, 0.74), radii=(0.0035, 0.0035, 0.007),
                      color=THROMBUS, opacity=1.0),
                   _e("pinch", LAD, id="stenosis", at=(0.5, 0.25, 0.80), radius=0.012, amount=0.95),
                   _e("lesion", LV, id="infarct", inset=0.35, at=(0.55, 0.08, 0.45), radii=(0.024, 0.008, 0.022),
                      color=INFARCT_ACUTE, opacity=0.95))),
            Stage("Healed", "Healed infarct — scar and thinning", "Weeks to months after infarction",
                  "Dead muscle is replaced by collagen scar over the following weeks. The scarred wall is "
                  "thin and no longer contracts, which can lead to heart failure. Secondary prevention — "
                  "medicines, risk-factor control and cardiac rehabilitation — aims to prevent a second event.",
                  (_e("lesion", LAD, id="plaque", at=(0.5, 0.25, 0.80), radii=(0.0050, 0.0050, 0.0085),
                      color=PLAQUE, opacity=1.0),
                   _e("pinch", LAD, id="stenosis", at=(0.5, 0.25, 0.80), radius=0.012, amount=0.6),
                   _e("lesion", LV, id="infarct", inset=0.35, at=(0.55, 0.08, 0.45), radii=(0.026, 0.006, 0.024),
                      color=SCAR, opacity=1.0),
                   _e("scale", LV, id="remodel", factor=1.06))),
        ),
        sources=("Stary HC et al. AHA Committee on Vascular Lesions, Circulation 1995",
                 "Thygesen K et al. Fourth Universal Definition of Myocardial Infarction, 2018",
                 "ESC 2023 Guidelines for the management of acute coronary syndromes"),
    ),
    # =====================================================================
    Condition(
        id="masld_cirrhosis", name="Fatty liver disease → cirrhosis (MASLD)", system="Digestive",
        organ_layers=("liver", "lymphatics"), focus=("lobe of liver", "gallbladder", "spleen"), view="anterior",
        staging="Steatosis → steatohepatitis → fibrosis stage F0–F4 (METAVIR / NASH-CRN)",
        summary=("Metabolic dysfunction-associated steatotic liver disease is fat accumulation in the liver "
                 "linked to obesity, type 2 diabetes and insulin resistance. In some people it progresses "
                 "through inflammation and fibrosis to cirrhosis."),
        stages=(
            Stage("Healthy", "Healthy liver", "Liver fat below 5% of hepatocytes",
                  "A healthy liver is smooth, red-brown and sits under the right dome of the diaphragm. "
                  "Fat makes up less than five percent of its cells."),
            Stage("MASLD", "Steatosis", "Fat in ≥5% of hepatocytes with a cardiometabolic risk factor",
                  "In steatosis, fat droplets fill more than five percent of liver cells. The liver enlarges "
                  "and turns paler and yellower. It is usually silent and is often found on an ultrasound "
                  "done for another reason. At this stage it is reversible with weight loss and exercise.",
                  (_e("tint", LIVER, id="fat", color=FAT_YELLOW, amount=0.45),
                   _e("scale", LIVER, id="size", factor=1.08))),
            Stage("MASH", "Steatohepatitis", "Steatosis with lobular inflammation and hepatocyte ballooning",
                  "When fat triggers inflammation and injury to liver cells, it becomes steatohepatitis. "
                  "The liver remains enlarged. This is the stage that drives scarring, and liver enzymes "
                  "may be raised on blood tests.",
                  (_e("tint", LIVER, id="fat", color=(0.84, 0.58, 0.34), amount=0.5),
                   _e("scale", LIVER, id="size", factor=1.10))),
            Stage("F2–F3", "Significant to advanced fibrosis", "Portal and bridging fibrosis",
                  "Repeated injury lays down collagen. Fibrous bands first surround the portal tracts and "
                  "then bridge between them. The liver becomes firmer, and its surface starts to lose its "
                  "smoothness. Non-invasive tests such as the FIB-4 score and elastography are used to "
                  "detect this stage.",
                  (_e("tint", LIVER, id="fat", color=FIBROTIC, amount=0.45),
                   _e("scale", LIVER, id="size", factor=1.02),
                   _e("nodular", LIVER, id="surface", amplitude=0.0015, frequency=70))),
            Stage("F4", "Cirrhosis", "Regenerative nodules surrounded by fibrous septa",
                  "In cirrhosis the architecture is replaced by regenerating nodules wrapped in scar. The "
                  "liver shrinks, its surface becomes coarsely nodular, and resistance to portal blood "
                  "flow causes portal hypertension — the spleen enlarges. Cirrhosis carries risks of "
                  "variceal bleeding, ascites and liver cancer, so people need regular specialist "
                  "surveillance.",
                  (_e("tint", LIVER, id="fat", color=CIRRHOTIC, amount=0.6),
                   _e("scale", LIVER, id="size", factor=0.84),
                   _e("nodular", LIVER, id="surface", amplitude=0.0035, frequency=85),
                   _e("scale", ("spleen",), id="spleen", factor=1.35))),
        ),
        sources=("EASL–EASD–EASO Clinical Practice Guidelines on MASLD, J Hepatol 2024",
                 "Rinella ME et al. Multisociety Delphi consensus on steatotic liver disease nomenclature, 2023",
                 "Bedossa P, Poynard T. METAVIR scoring system, Hepatology 1996"),
    ),
    # =====================================================================
    Condition(
        id="ckd", name="Chronic kidney disease (KDIGO G1–G5)", system="Urinary",
        organ_layers=("kidneys",), focus=("kidney",), view="posterior",
        staging="KDIGO GFR categories G1–G5",
        summary=("Chronic kidney disease is abnormal kidney structure or function lasting more than three "
                 "months. It is staged by glomerular filtration rate and albuminuria; diabetes and high "
                 "blood pressure are the leading causes."),
        stages=(
            Stage("G1", "Normal or high GFR", "eGFR ≥90 mL/min/1.73 m² with kidney damage markers",
                  "In category G1 filtration is normal, but there is evidence of kidney damage, such as "
                  "albumin in the urine. The kidneys look normal in size — about eleven centimetres long."),
            Stage("G2", "Mildly decreased", "eGFR 60–89 mL/min/1.73 m²",
                  "In G2, filtration is mildly reduced. There are usually no symptoms. Controlling blood "
                  "pressure and blood sugar now does the most to slow progression.",
                  (_e("tint", KIDNEY, id="pallor", color=PALE, amount=0.1),)),
            Stage("G3a–G3b", "Moderately decreased", "eGFR 30–59 mL/min/1.73 m²",
                  "In G3 the filtration rate is between thirty and fifty-nine. The cortex begins to thin "
                  "and the kidneys become slightly smaller and paler, with a finely granular surface in "
                  "hypertensive and diabetic disease. Anaemia and bone-mineral changes may begin.",
                  (_e("tint", KIDNEY, id="pallor", color=PALE, amount=0.3),
                   _e("scale", KIDNEY, id="size", factor=0.93),
                   _e("nodular", KIDNEY, id="granular", amplitude=0.0008, frequency=140))),
            Stage("G4", "Severely decreased", "eGFR 15–29 mL/min/1.73 m²",
                  "In G4 the kidneys are visibly shrunken. Fluid, potassium and acid build up more easily, "
                  "and planning for kidney replacement therapy begins.",
                  (_e("tint", KIDNEY, id="pallor", color=PALE, amount=0.5),
                   _e("scale", KIDNEY, id="size", factor=0.85),
                   _e("nodular", KIDNEY, id="granular", amplitude=0.0013, frequency=140))),
            Stage("G5", "Kidney failure", "eGFR <15 mL/min/1.73 m²",
                  "In G5, kidney failure, the kidneys are small, pale and scarred. Without dialysis or a "
                  "transplant, toxins and fluid accumulate. With treatment, many people live well for "
                  "years.",
                  (_e("tint", KIDNEY, id="pallor", color=PALE, amount=0.65),
                   _e("scale", KIDNEY, id="size", factor=0.74),
                   _e("nodular", KIDNEY, id="granular", amplitude=0.0018, frequency=140))),
        ),
        sources=("KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of CKD",),
    ),
    # =====================================================================
    Condition(
        id="copd", name="COPD and emphysema (GOLD 1–4)", system="Respiratory",
        organ_layers=("lungs",), focus=("lung", "diaphragm"), view="anterior",
        staging="GOLD spirometric grades 1–4 (post-bronchodilator FEV₁/FVC <0.70)",
        summary=("Chronic obstructive pulmonary disease is persistent airflow limitation, most often from "
                 "tobacco smoke or biomass fuel exposure. Emphysema — destruction of the alveolar walls — "
                 "makes the lungs over-inflate and trap air."),
        stages=(
            Stage("Healthy", "Healthy lungs", "FEV₁/FVC ≥0.70",
                  "Healthy lungs are spongy and pink, and the diaphragm forms two high domes. On breathing "
                  "out, the elastic lungs recoil and empty."),
            Stage("GOLD 1", "Mild", "FEV₁ ≥80% predicted",
                  "In GOLD 1 airflow is only mildly limited. Many people notice just a morning cough. "
                  "Stopping smoking at this stage changes the course of the disease more than any medicine.",
                  (_e("tint", LUNG, id="grey", color=GREY_LUNG, amount=0.15),
                   _e("scale", LUNG, id="size", factor=1.02))),
            Stage("GOLD 2", "Moderate", "FEV₁ 50–79% predicted",
                  "In GOLD 2 breathlessness on exertion appears. Destroyed alveolar walls reduce elastic "
                  "recoil, so air is trapped and the lungs begin to over-inflate.",
                  (_e("tint", LUNG, id="grey", color=GREY_LUNG, amount=0.3),
                   _e("scale", LUNG, id="size", factor=1.05),
                   _e("scale", ("diaphragm",), id="flatten", factor=(1.02, 1.02, 0.85)))),
            Stage("GOLD 3", "Severe", "FEV₁ 30–49% predicted",
                  "In GOLD 3 the lungs are hyperinflated and the diaphragm is flattened, so each breath "
                  "takes more effort. Large air spaces called bullae form, often at the apices. Flare-ups "
                  "become more frequent.",
                  (_e("tint", LUNG, id="grey", color=GREY_LUNG, amount=0.45),
                   _e("scale", LUNG, id="size", factor=1.09),
                   _e("scale", ("diaphragm",), id="flatten", factor=(1.04, 1.04, 0.65)),
                   _e("lesion", ("superior lobe of right lung",), id="bulla_r", inset=0.9, at=(0.5, 0.45, 0.85),
                      radii=(0.014, 0.014, 0.012), color=(0.92, 0.86, 0.86), opacity=0.55))),
            Stage("GOLD 4", "Very severe", "FEV₁ <30% predicted",
                  "In GOLD 4 breathlessness limits everyday activity, and oxygen levels may fall. The chest "
                  "is barrel-shaped from chronic over-inflation. Treatment focuses on inhalers, pulmonary "
                  "rehabilitation, oxygen where indicated and preventing exacerbations.",
                  (_e("tint", LUNG, id="grey", color=GREY_LUNG, amount=0.6),
                   _e("scale", LUNG, id="size", factor=1.13),
                   _e("scale", ("diaphragm",), id="flatten", factor=(1.05, 1.05, 0.5)),
                   _e("lesion", ("superior lobe of right lung",), id="bulla_r", inset=0.9, at=(0.5, 0.45, 0.85),
                      radii=(0.020, 0.020, 0.017), color=(0.92, 0.86, 0.86), opacity=0.55),
                   _e("lesion", ("superior lobe of left lung",), id="bulla_l", inset=0.9, at=(0.5, 0.45, 0.82),
                      radii=(0.016, 0.016, 0.014), color=(0.92, 0.86, 0.86), opacity=0.55))),
        ),
        sources=("Global Initiative for Chronic Obstructive Lung Disease (GOLD) 2024 Report",),
        organ_opacity=0.7,
    ),
    # =====================================================================
    Condition(
        id="stroke", name="Ischaemic stroke (left middle cerebral artery)", system="Nervous",
        organ_layers=("brain", "arteries"), focus=("cerebral hemisphere", "cerebellum", "carotid"),
        view="superior",
        staging="Time course: hyperacute, acute, subacute, chronic",
        summary=("An ischaemic stroke occurs when an artery to the brain is blocked. A left middle cerebral "
                 "artery stroke typically affects movement of the right side and, in most people, language."),
        stages=(
            Stage("Before", "Healthy brain", "Normal perfusion",
                  "The middle cerebral artery supplies the outer surface of each hemisphere, including the "
                  "motor and sensory areas for the face and arm, and, on the left, the language areas."),
            Stage("Hyperacute", "First hours: core and penumbra", "Within about 6 hours of onset",
                  "When the artery is blocked, a core of tissue dies quickly, surrounded by a penumbra that "
                  "is starved of blood but still salvageable. Sudden face drooping, arm weakness and speech "
                  "difficulty mean: call emergency services immediately. Clot-dissolving treatment or clot "
                  "removal can save the penumbra, but only within hours.",
                  (_e("lesion", ("left cerebral hemisphere",), id="penumbra", inset=1.0, at=(0.85, 0.45, 0.55),
                      radii=(0.030, 0.026, 0.022), color=ISCHAEMIC, opacity=0.45),
                   _e("lesion", ("left cerebral hemisphere",), id="core", inset=1.1, at=(0.88, 0.45, 0.55),
                      radii=(0.014, 0.012, 0.011), color=(0.42, 0.22, 0.48), opacity=0.9))),
            Stage("Acute", "Days 1–7: infarct and swelling", "Cytotoxic and vasogenic oedema",
                  "Without reperfusion, the penumbra joins the infarct. The damaged tissue swells over the "
                  "next few days; large infarcts can raise pressure inside the skull.",
                  (_e("lesion", ("left cerebral hemisphere",), id="core", inset=1.1, at=(0.88, 0.45, 0.55),
                      radii=(0.028, 0.024, 0.021), color=(0.55, 0.30, 0.50), opacity=0.9),
                   _e("scale", ("left cerebral hemisphere",), id="oedema", factor=1.02))),
            Stage("Chronic", "Months later: encephalomalacia", "Fluid-filled cavity with gliosis",
                  "Dead tissue is cleared and replaced by a fluid-filled cavity with surrounding scarring, "
                  "and the hemisphere loses some volume. Rehabilitation helps the surviving brain take over "
                  "lost functions.",
                  (_e("lesion", ("left cerebral hemisphere",), id="core", inset=1.1, at=(0.88, 0.45, 0.55),
                      radii=(0.026, 0.022, 0.019), color=(0.30, 0.45, 0.70), opacity=0.85),
                   _e("scale", ("left cerebral hemisphere",), id="oedema", factor=0.98))),
        ),
        sources=("Powers WJ et al. AHA/ASA Guidelines for the Early Management of Acute Ischemic Stroke, 2019",
                 "ESO guidelines on intravenous thrombolysis (2021) and mechanical thrombectomy (2019)"),
        context_layers=("skeleton",), organ_opacity=0.42,
    ),
    # =====================================================================
    Condition(
        id="knee_oa", name="Knee osteoarthritis (Kellgren–Lawrence 0–4)", system="Skeletal",
        organ_layers=("cartilage", "skeleton"),
        focus=("right patella", "right medial meniscus", "right lateral meniscus", "right articular cartilage of knee"),
        view="anterior",
        staging="Kellgren–Lawrence radiographic grade 0–4",
        summary=("Osteoarthritis is the gradual loss of joint cartilage with reactive changes in the bone "
                 "beneath. The knee is the commonest large joint affected."),
        stages=(
            Stage("KL 0", "Normal joint", "No radiographic features of osteoarthritis",
                  "In a healthy knee, smooth hyaline cartilage covers the femur and tibia, and the menisci "
                  "cushion the joint. On an X-ray, the joint space looks wide and even."),
            Stage("KL 1", "Doubtful", "Doubtful joint-space narrowing, possible osteophytic lipping",
                  "Grade 1 shows doubtful narrowing and perhaps a tiny bony spur. The cartilage surface "
                  "begins to soften and fray.",
                  (_e("scale", ("right articular cartilage of knee", "right medial meniscus"), id="cartilage",
                      factor=(1.0, 1.0, 0.92)),
                   _e("lesion", ("right femur",), id="osteo_fm", at=(0.92, 0.45, 0.04),
                      radii=(0.003, 0.003, 0.003), color=OSTEOPHYTE, opacity=1.0))),
            Stage("KL 2", "Minimal", "Definite osteophytes, possible joint-space narrowing",
                  "Grade 2 has definite osteophytes at the joint margins. Pain often comes with activity "
                  "and eases with rest. Exercise and weight management are the core treatment.",
                  (_e("scale", ("right articular cartilage of knee", "right medial meniscus"), id="cartilage",
                      factor=(1.0, 1.0, 0.8)),
                   _e("lesion", ("right femur",), id="osteo_fm", at=(0.92, 0.45, 0.04),
                      radii=(0.005, 0.004, 0.005), color=OSTEOPHYTE, opacity=1.0),
                   _e("lesion", ("right tibia",), id="osteo_tm", at=(0.95, 0.45, 0.97),
                      radii=(0.004, 0.004, 0.004), color=OSTEOPHYTE, opacity=1.0))),
            Stage("KL 3", "Moderate", "Multiple osteophytes, definite narrowing, some sclerosis",
                  "Grade 3 shows definite narrowing of the joint space, usually on the medial side first, "
                  "with several osteophytes and hardening of the bone under the cartilage.",
                  (_e("scale", ("right articular cartilage of knee", "right medial meniscus"), id="cartilage",
                      factor=(1.0, 1.0, 0.55)),
                   _e("lesion", ("right femur",), id="osteo_fm", at=(0.92, 0.45, 0.04),
                      radii=(0.007, 0.005, 0.007), color=OSTEOPHYTE, opacity=1.0),
                   _e("lesion", ("right tibia",), id="osteo_tm", at=(0.95, 0.45, 0.97),
                      radii=(0.006, 0.005, 0.005), color=OSTEOPHYTE, opacity=1.0),
                   _e("lesion", ("right femur",), id="osteo_fl", at=(0.08, 0.45, 0.04),
                      radii=(0.004, 0.004, 0.004), color=OSTEOPHYTE, opacity=1.0),
                   _e("tint", ("right tibia",), id="sclerosis", color=SCLEROTIC, amount=0.3))),
            Stage("KL 4", "Severe", "Large osteophytes, marked narrowing, severe sclerosis, deformity",
                  "Grade 4 has marked joint-space narrowing, large osteophytes, severe sclerosis and often "
                  "a bow-legged deformity. When pain and disability persist despite other treatment, "
                  "knee replacement is considered.",
                  (_e("scale", ("right articular cartilage of knee", "right medial meniscus"), id="cartilage",
                      factor=(1.0, 1.0, 0.3)),
                   _e("lesion", ("right femur",), id="osteo_fm", at=(0.92, 0.45, 0.04),
                      radii=(0.010, 0.006, 0.009), color=OSTEOPHYTE, opacity=1.0),
                   _e("lesion", ("right tibia",), id="osteo_tm", at=(0.95, 0.45, 0.97),
                      radii=(0.009, 0.006, 0.007), color=OSTEOPHYTE, opacity=1.0),
                   _e("lesion", ("right femur",), id="osteo_fl", at=(0.08, 0.45, 0.04),
                      radii=(0.007, 0.005, 0.006), color=OSTEOPHYTE, opacity=1.0),
                   _e("tint", ("right tibia", "right femur"), id="sclerosis", color=SCLEROTIC, amount=0.5))),
        ),
        sources=("Kellgren JH, Lawrence JS. Radiological assessment of osteo-arthrosis, Ann Rheum Dis 1957",
                 "NICE guideline NG226: Osteoarthritis in over 16s, 2022"),
        context_layers=("skeleton", "cartilage", "tendons"), isolate=True,
    ),
    # =====================================================================
    Condition(
        id="osteoporosis", name="Osteoporosis and vertebral fracture (WHO T-score)", system="Skeletal",
        organ_layers=("skeleton",), focus=("T12 vertebra", "L1 vertebra", "L2 vertebra", "twelfth thoracic", "first lumbar",
                                           "second lumbar"),
        view="left",
        staging="WHO bone mineral density categories (DXA T-score)",
        summary=("Osteoporosis is low bone mass with deterioration of bone micro-architecture, making bones "
                 "fragile. The first sign is often a fracture of a vertebra, wrist or hip."),
        stages=(
            Stage("Normal", "Normal bone density", "T-score −1.0 or above",
                  "Healthy vertebral bodies are box-shaped, with dense trabecular bone inside a firm cortex."),
            Stage("Osteopenia", "Low bone mass", "T-score between −1.0 and −2.5",
                  "In osteopenia, bone density is below that of a young adult but above the osteoporosis "
                  "threshold. Weight-bearing exercise, calcium, vitamin D and fall prevention all matter.",
                  (_e("tint", ("vertebra",), id="porous", color=(0.97, 0.95, 0.90), amount=0.25),)),
            Stage("Osteoporosis", "Osteoporosis", "T-score −2.5 or below",
                  "In osteoporosis, the trabeculae thin and disconnect, so the vertebrae can no longer "
                  "carry normal loads safely. Medicines that reduce fracture risk are usually recommended.",
                  (_e("tint", ("vertebra",), id="porous", color=(0.98, 0.96, 0.92), amount=0.45),)),
            Stage("Severe", "Established osteoporosis: vertebral compression fracture",
                  "T-score −2.5 or below with one or more fragility fractures",
                  "Severe osteoporosis means a fragility fracture has occurred. Here the first lumbar vertebra "
                  "has collapsed into a wedge after a minor strain. Repeated fractures cause loss of height "
                  "and a stooped posture.",
                  (_e("tint", ("vertebra",), id="porous", color=(0.98, 0.96, 0.92), amount=0.55),
                   _e("scale", ("L1 vertebra", "first lumbar vertebra"), id="wedge", factor=(1.06, 1.04, 0.62)))),
        ),
        sources=("WHO Study Group, Assessment of fracture risk and its application to screening for "
                 "postmenopausal osteoporosis, 1994",
                 "NOGG 2022 UK clinical guideline for the prevention and treatment of osteoporosis"),
        isolate=True,
    ),
    # =====================================================================
    Condition(
        id="hypertension", name="Hypertension and left ventricular hypertrophy", system="Cardiovascular",
        organ_layers=("heart", "arteries", "kidneys"), focus=("ventricle", "atrium", "aort"), view="anterior",
        staging="ACC/AHA 2017 blood-pressure categories",
        summary=("Persistently raised blood pressure makes the left ventricle work against a higher load. "
                 "Over years the muscle thickens, the arteries stiffen and the kidneys and brain are damaged."),
        stages=(
            Stage("Normal", "Normal blood pressure", "Below 120/80 mmHg",
                  "At normal blood pressure, the left ventricular wall is about one centimetre thick and the "
                  "aorta remains elastic."),
            Stage("Elevated", "Elevated", "Systolic 120–129 and diastolic below 80 mmHg",
                  "Elevated blood pressure has no symptoms. Lifestyle measures — less salt, more activity, "
                  "weight control and limiting alcohol — can bring it back to normal."),
            Stage("Stage 1", "Stage 1 hypertension", "130–139 or 80–89 mmHg",
                  "In stage 1, the heart starts adapting to the higher pressure. Whether medicine is "
                  "started depends on overall cardiovascular risk.",
                  (_e("scale", LV, id="lvh", factor=1.05),)),
            Stage("Stage 2", "Stage 2 hypertension with LVH", "≥140 or ≥90 mmHg",
                  "With stage 2 hypertension sustained for years, the left ventricle develops concentric "
                  "hypertrophy: its wall thickens and becomes stiff, and the aorta loses elasticity. "
                  "Treating the blood pressure can partly reverse this hypertrophy.",
                  (_e("scale", LV, id="lvh", factor=1.14),
                   _e("tint", LV, id="lv_colour", color=(0.62, 0.12, 0.18), amount=0.3),
                   _e("tint", ("aorta",), id="stiff", color=(0.88, 0.80, 0.62), amount=0.35),
                   _e("tint", KIDNEY, id="kidney", color=PALE, amount=0.15))),
        ),
        sources=("Whelton PK et al. 2017 ACC/AHA Guideline for High Blood Pressure in Adults",
                 "2023 ESH Guidelines for the management of arterial hypertension"),
    ),
)

CONDITION_BY_ID: Dict[str, Condition] = {c.id: c for c in CONDITIONS}


def stage_text(condition: Condition, index: int) -> str:
    stage = condition.stages[index]
    return f"{stage.label} — {stage.title}\nCriteria: {stage.criteria}"
