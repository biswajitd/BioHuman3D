"""
Muscle knowledge base: attachments, innervation and actions.

Sources: Gray's Anatomy (42nd ed.), Moore's Clinically Oriented Anatomy (9th
ed.). Normal ranges of motion: American Academy of Orthopaedic Surgeons (AAOS)
values as tabulated by Norkin & White, *Measurement of Joint Motion* (5th ed.).

Each muscle lists its actions as joint *motions* (``"elbow.flexion"``). A motion
names a degree of freedom of the kinematic rig (``app/anatomy/kinematics.py``)
and the angles to animate, so "show me the biceps" moves the elbow through
flexion and the forearm through supination — the real actions, not a generic
wobble. Muscles without a limb joint action (masseter, diaphragm …) are shown
contracting in place.

Clinical notes are teaching statements, not diagnostic advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Motion:
    """One joint motion the rig can animate."""

    id: str                  # "elbow.flexion"
    label: str               # "Elbow flexion"
    limb: str                # "upper" | "lower" | ""
    dof: str                 # rig degree of freedom, e.g. "elbow_flex"
    start: float             # degrees: pose the joint is moved to passively
    end: float               # degrees: pose reached by the contraction
    normal_range: str        # "0–150°" (AAOS)
    plane: str = ""


#: Positive DOF directions: flexion / abduction / internal rotation / pronation /
#: dorsiflexion / inversion / radial deviation. Opposite motions use negative
#: angles or start displaced and return to neutral.
MOTIONS: Dict[str, Motion] = {m.id: m for m in (
    Motion("shoulder.flexion", "Shoulder flexion", "upper", "shoulder_flex", 0, 120, "0–180°", "sagittal"),
    Motion("shoulder.extension", "Shoulder extension", "upper", "shoulder_flex", 0, -50, "0–60°", "sagittal"),
    Motion("shoulder.abduction", "Shoulder abduction", "upper", "shoulder_abd", 0, 100, "0–180°", "coronal"),
    Motion("shoulder.adduction", "Shoulder adduction", "upper", "shoulder_abd", 80, 0, "0–30° beyond neutral", "coronal"),
    Motion("shoulder.internal_rotation", "Shoulder internal (medial) rotation", "upper", "shoulder_rot", 0, 60, "0–70°", "transverse"),
    Motion("shoulder.external_rotation", "Shoulder external (lateral) rotation", "upper", "shoulder_rot", 0, -70, "0–90°", "transverse"),
    Motion("elbow.flexion", "Elbow flexion", "upper", "elbow_flex", 0, 135, "0–150°", "sagittal"),
    Motion("elbow.extension", "Elbow extension", "upper", "elbow_flex", 120, 0, "150–0°", "sagittal"),
    Motion("forearm.pronation", "Forearm pronation", "upper", "pronation", 0, 75, "0–80°", "transverse"),
    Motion("forearm.supination", "Forearm supination", "upper", "pronation", 75, 0, "0–80°", "transverse"),
    Motion("wrist.flexion", "Wrist flexion", "upper", "wrist_flex", 0, 70, "0–80°", "sagittal"),
    Motion("wrist.extension", "Wrist extension", "upper", "wrist_flex", 0, -60, "0–70°", "sagittal"),
    Motion("wrist.radial_deviation", "Wrist radial deviation", "upper", "wrist_dev", 0, 20, "0–20°", "coronal"),
    Motion("wrist.ulnar_deviation", "Wrist ulnar deviation", "upper", "wrist_dev", 0, -30, "0–30°", "coronal"),
    Motion("hip.flexion", "Hip flexion", "lower", "hip_flex", 0, 100, "0–120°", "sagittal"),
    Motion("hip.extension", "Hip extension", "lower", "hip_flex", 0, -25, "0–30°", "sagittal"),
    Motion("hip.abduction", "Hip abduction", "lower", "hip_abd", 0, 40, "0–45°", "coronal"),
    Motion("hip.adduction", "Hip adduction", "lower", "hip_abd", 35, -10, "0–30°", "coronal"),
    Motion("hip.internal_rotation", "Hip internal (medial) rotation", "lower", "hip_rot", 0, 35, "0–45°", "transverse"),
    Motion("hip.external_rotation", "Hip external (lateral) rotation", "lower", "hip_rot", 0, -40, "0–45°", "transverse"),
    Motion("knee.flexion", "Knee flexion", "lower", "knee_flex", 0, 120, "0–135°", "sagittal"),
    Motion("knee.extension", "Knee extension", "lower", "knee_flex", 100, 0, "135–0°", "sagittal"),
    Motion("ankle.dorsiflexion", "Ankle dorsiflexion", "lower", "ankle_dorsi", 0, 20, "0–20°", "sagittal"),
    Motion("ankle.plantarflexion", "Ankle plantarflexion", "lower", "ankle_dorsi", 0, -45, "0–50°", "sagittal"),
    Motion("foot.inversion", "Foot inversion", "lower", "ankle_inv", 0, 30, "0–35°", "coronal"),
    Motion("foot.eversion", "Foot eversion", "lower", "ankle_inv", 0, -15, "0–15°", "coronal"),
)}

#: Motion → the motion that opposes it (used to find antagonists).
OPPOSITE: Dict[str, str] = {}
for _a, _b in (("shoulder.flexion", "shoulder.extension"), ("shoulder.abduction", "shoulder.adduction"),
               ("shoulder.internal_rotation", "shoulder.external_rotation"),
               ("elbow.flexion", "elbow.extension"), ("forearm.pronation", "forearm.supination"),
               ("wrist.flexion", "wrist.extension"), ("wrist.radial_deviation", "wrist.ulnar_deviation"),
               ("hip.flexion", "hip.extension"), ("hip.abduction", "hip.adduction"),
               ("hip.internal_rotation", "hip.external_rotation"), ("knee.flexion", "knee.extension"),
               ("ankle.dorsiflexion", "ankle.plantarflexion"), ("foot.inversion", "foot.eversion")):
    OPPOSITE[_a], OPPOSITE[_b] = _b, _a


@dataclass(frozen=True)
class Muscle:
    key: str                         # lower-case name used to match structures
    name: str
    origin: str
    insertion: str
    innervation: str
    actions: Tuple[str, ...]         # Motion ids, prime action first
    action_text: str
    blood_supply: str = ""
    clinical: str = ""
    region: str = ""
    aliases: Tuple[str, ...] = ()

    def matches(self, structure_name: str) -> bool:
        low = structure_name.lower()
        return any(k in low for k in (self.key,) + self.aliases)


MUSCLES: Tuple[Muscle, ...] = (
    # -- shoulder girdle & arm ---------------------------------------------
    Muscle("deltoid", "Deltoid",
           "Lateral third of the clavicle, acromion and spine of the scapula",
           "Deltoid tuberosity of the humerus",
           "Axillary nerve (C5, C6)",
           ("shoulder.abduction", "shoulder.flexion", "shoulder.extension"),
           "The middle fibres abduct the arm from about 15° to 90°; the anterior fibres flex and "
           "medially rotate it, and the posterior fibres extend and laterally rotate it.",
           "Posterior circumflex humeral artery",
           "The axillary nerve can be stretched in anterior shoulder dislocation or a surgical-neck "
           "fracture, weakening abduction and numbing the skin over the lower deltoid.", "Shoulder"),
    Muscle("supraspinatus", "Supraspinatus", "Supraspinous fossa of the scapula",
           "Superior facet of the greater tubercle of the humerus", "Suprascapular nerve (C5, C6)",
           ("shoulder.abduction",),
           "Initiates abduction over the first 15° and holds the humeral head in the glenoid "
           "as part of the rotator cuff.", "Suprascapular artery",
           "The most commonly torn rotator-cuff tendon; impingement under the acromion causes a "
           "painful arc between about 60° and 120° of abduction.", "Shoulder"),
    Muscle("infraspinatus", "Infraspinatus", "Infraspinous fossa of the scapula",
           "Middle facet of the greater tubercle of the humerus", "Suprascapular nerve (C5, C6)",
           ("shoulder.external_rotation",), "Laterally rotates the arm and stabilises the glenohumeral joint.",
           "Suprascapular and circumflex scapular arteries", "", "Shoulder"),
    Muscle("teres major", "Teres major", "Posterior surface of the inferior angle of the scapula",
           "Medial lip of the intertubercular sulcus of the humerus", "Lower subscapular nerve (C5–C7)",
           ("shoulder.internal_rotation", "shoulder.adduction", "shoulder.extension"),
           "Adducts, medially rotates and extends the arm.", "Circumflex scapular artery", "", "Shoulder"),
    Muscle("pectoralis major", "Pectoralis major",
           "Clavicular head: medial half of the clavicle. Sternocostal head: sternum, costal cartilages 1–6 "
           "and the external oblique aponeurosis",
           "Lateral lip of the intertubercular sulcus of the humerus",
           "Lateral and medial pectoral nerves (C5–T1)",
           ("shoulder.adduction", "shoulder.internal_rotation", "shoulder.flexion"),
           "Adducts and medially rotates the arm; the clavicular head flexes it and the sternocostal "
           "head extends it from a flexed position.", "Pectoral branch of the thoracoacromial artery",
           "Congenital absence of the sternocostal head is seen in Poland syndrome.", "Thorax"),
    Muscle("pectoralis minor", "Pectoralis minor", "Ribs 3–5 near their costal cartilages",
           "Coracoid process of the scapula", "Medial pectoral nerve (C8, T1)", (),
           "Stabilises the scapula by drawing it anteriorly and inferiorly against the thoracic wall.",
           "Thoracoacromial artery", "", "Thorax"),
    Muscle("latissimus dorsi", "Latissimus dorsi",
           "Spinous processes of T7–T12, thoracolumbar fascia, iliac crest and the inferior 3–4 ribs",
           "Floor of the intertubercular sulcus of the humerus", "Thoracodorsal nerve (C6–C8)",
           ("shoulder.extension", "shoulder.adduction", "shoulder.internal_rotation"),
           "Extends, adducts and medially rotates the arm — the climbing and swimming muscle.",
           "Thoracodorsal artery",
           "Its reliable pedicle makes it a workhorse flap in reconstructive surgery.", "Back"),
    Muscle("trapezius", "Trapezius",
           "Superior nuchal line, external occipital protuberance, ligamentum nuchae and spinous "
           "processes of C7–T12",
           "Lateral third of the clavicle, acromion and spine of the scapula",
           "Spinal accessory nerve (CN XI); C3, C4 for proprioception", (),
           "Descending fibres elevate the scapula, middle fibres retract it and ascending fibres depress "
           "it; together they rotate the glenoid upward during overhead reach.",
           "Transverse cervical artery",
           "Accessory-nerve injury in posterior-triangle surgery causes a drooping shoulder.", "Back"),
    Muscle("rhomboid major", "Rhomboid major", "Spinous processes of T2–T5",
           "Medial border of the scapula below the spine", "Dorsal scapular nerve (C4, C5)", (),
           "Retracts the scapula and rotates the glenoid downward.", "Dorsal scapular artery", "", "Back"),
    Muscle("serratus anterior", "Serratus anterior", "External surfaces of ribs 1–8",
           "Anterior surface of the medial border of the scapula", "Long thoracic nerve (C5–C7)", (),
           "Protracts the scapula and rotates it upward; holds it against the thoracic wall.",
           "Lateral thoracic artery",
           "Long thoracic nerve injury produces a 'winged' scapula on pushing against a wall.", "Thorax"),
    Muscle("biceps brachii", "Biceps brachii",
           "Short head: tip of the coracoid process. Long head: supraglenoid tubercle of the scapula",
           "Radial tuberosity and, via the bicipital aponeurosis, the deep fascia of the forearm",
           "Musculocutaneous nerve (C5, C6)",
           ("forearm.supination", "elbow.flexion", "shoulder.flexion"),
           "The most powerful supinator of the forearm, and a strong elbow flexor when the forearm "
           "is supinated.", "Muscular branches of the brachial artery",
           "Rupture of the long-head tendon produces the 'Popeye' bulge; the biceps reflex tests C5–C6.",
           "Arm"),
    Muscle("brachialis", "Brachialis", "Distal half of the anterior surface of the humerus",
           "Coronoid process and tuberosity of the ulna", "Musculocutaneous nerve (C5, C6)",
           ("elbow.flexion",), "The prime flexor of the elbow in every forearm position.",
           "Brachial and radial recurrent arteries",
           "Can calcify after elbow trauma (myositis ossificans).", "Arm"),
    Muscle("triceps brachii", "Triceps brachii",
           "Long head: infraglenoid tubercle. Lateral and medial heads: posterior humerus above and below "
           "the radial groove", "Olecranon of the ulna", "Radial nerve (C6–C8)",
           ("elbow.extension", "shoulder.extension"),
           "The main extensor of the elbow; the long head also helps extend and adduct the arm.",
           "Profunda brachii artery", "The triceps reflex tests C7.", "Arm"),
    Muscle("brachioradialis", "Brachioradialis", "Proximal two-thirds of the lateral supracondylar ridge of the humerus",
           "Lateral surface of the distal radius near the styloid process", "Radial nerve (C5, C6)",
           ("elbow.flexion",), "Flexes the elbow, most effectively with the forearm mid-way between "
                               "pronation and supination.", "Radial recurrent artery",
           "The brachioradialis reflex tests C6.", "Forearm"),
    Muscle("pronator teres", "Pronator teres",
           "Humeral head: medial epicondyle. Ulnar head: coronoid process",
           "Middle of the lateral surface of the radius", "Median nerve (C6, C7)",
           ("forearm.pronation", "elbow.flexion"), "Pronates the forearm and weakly flexes the elbow.",
           "Ulnar and anterior ulnar recurrent arteries",
           "The median nerve can be compressed between its two heads (pronator syndrome).", "Forearm"),
    Muscle("flexor carpi radialis", "Flexor carpi radialis", "Medial epicondyle (common flexor origin)",
           "Base of the second metacarpal", "Median nerve (C6, C7)",
           ("wrist.flexion", "wrist.radial_deviation"), "Flexes the wrist and deviates it radially.",
           "Radial artery", "", "Forearm"),
    Muscle("flexor carpi ulnaris", "Flexor carpi ulnaris",
           "Medial epicondyle, olecranon and posterior border of the ulna",
           "Pisiform, hook of hamate and base of the fifth metacarpal", "Ulnar nerve (C7, C8)",
           ("wrist.flexion", "wrist.ulnar_deviation"), "Flexes the wrist and deviates it toward the ulna.",
           "Ulnar artery", "The ulnar nerve enters the forearm between its two heads (cubital tunnel).",
           "Forearm"),
    Muscle("flexor digitorum superficialis", "Flexor digitorum superficialis",
           "Medial epicondyle, coronoid process and anterior radius", "Middle phalanges of digits 2–5",
           "Median nerve (C7–T1)", ("wrist.flexion",),
           "Flexes the proximal interphalangeal joints of the fingers and assists wrist flexion.",
           "Ulnar and radial arteries", "", "Forearm"),
    Muscle("extensor carpi radialis longus", "Extensor carpi radialis longus",
           "Lateral supracondylar ridge of the humerus", "Dorsal base of the second metacarpal",
           "Radial nerve (C6, C7)", ("wrist.extension", "wrist.radial_deviation"),
           "Extends the wrist and deviates it radially; essential for a strong grip.", "Radial artery",
           "", "Forearm"),
    Muscle("extensor digitorum", "Extensor digitorum", "Lateral epicondyle (common extensor origin)",
           "Extensor expansions of digits 2–5", "Posterior interosseous nerve (C7, C8)",
           ("wrist.extension",), "Extends the fingers and assists wrist extension.",
           "Posterior interosseous artery",
           "Overuse at the common extensor origin underlies lateral epicondylitis (tennis elbow).",
           "Forearm"),
    # -- head, neck & trunk --------------------------------------------------
    Muscle("sternocleidomastoid", "Sternocleidomastoid",
           "Sternal head: manubrium. Clavicular head: medial third of the clavicle",
           "Mastoid process and lateral superior nuchal line", "Spinal accessory nerve (CN XI); C2, C3",
           (), "Acting alone it tilts the head to the same side and turns the face to the opposite side; "
               "both together flex the neck.", "Occipital and superior thyroid arteries",
           "Fibrosis in infancy causes congenital torticollis.", "Neck"),
    Muscle("masseter", "Masseter", "Inferior border and medial surface of the zygomatic arch",
           "Lateral surface of the ramus and angle of the mandible",
           "Masseteric nerve (mandibular division of the trigeminal, V3)", (),
           "Elevates the mandible to close the jaw — the strongest muscle of mastication.",
           "Masseteric artery", "Hypertrophy is associated with bruxism.", "Head"),
    Muscle("temporalis", "Temporalis", "Floor of the temporal fossa and deep temporal fascia",
           "Coronoid process and anterior border of the ramus of the mandible",
           "Deep temporal nerves (V3)", (),
           "Elevates the mandible; its posterior fibres retract it.", "Deep temporal arteries", "", "Head"),
    Muscle("erector spinae", "Erector spinae",
           "Sacrum, iliac crest, lumbar and lower thoracic spinous processes",
           "Ribs, transverse and spinous processes of vertebrae, and the mastoid process",
           "Posterior rami of the spinal nerves", (),
           "Extends the vertebral column and bends it to the same side; controls flexion eccentrically.",
           "Posterior intercostal, subcostal and lumbar arteries",
           "Strain is a leading cause of acute low back pain.", "Back"),
    Muscle("rectus abdominis", "Rectus abdominis", "Pubic symphysis and pubic crest",
           "Xiphoid process and costal cartilages 5–7", "Thoracoabdominal nerves (T7–T11) and subcostal nerve (T12)",
           (), "Flexes the trunk and compresses the abdominal viscera.",
           "Superior and inferior epigastric arteries",
           "Separation of the two recti (diastasis recti) is common after pregnancy.", "Abdomen"),
    Muscle("external oblique", "External oblique", "External surfaces of ribs 5–12",
           "Linea alba, pubic tubercle and anterior half of the iliac crest",
           "Thoracoabdominal nerves (T7–T11) and subcostal nerve (T12)", (),
           "Flexes and rotates the trunk to the opposite side; compresses the abdomen.",
           "Lower posterior intercostal arteries",
           "Its aponeurosis forms the inguinal ligament and the anterior wall of the inguinal canal.",
           "Abdomen"),
    Muscle("diaphragm", "Diaphragm", "Xiphoid process, inner surfaces of the lower six costal cartilages, "
           "and L1–L3 vertebral bodies via the crura", "Central tendon", "Phrenic nerve (C3–C5)", (),
           "The principal muscle of inspiration: contraction flattens the dome and enlarges the thorax.",
           "Pericardiacophrenic, musculophrenic and inferior phrenic arteries",
           "'C3, 4, 5 keeps the diaphragm alive' — high cervical cord injury abolishes breathing.",
           "Thorax", aliases=("hemidiaphragm",)),
    # -- hip & thigh -----------------------------------------------------------
    Muscle("iliopsoas", "Iliopsoas",
           "Psoas major: T12–L5 bodies and transverse processes. Iliacus: iliac fossa",
           "Lesser trochanter of the femur",
           "Psoas: anterior rami of L1–L3. Iliacus: femoral nerve (L2, L3)",
           ("hip.flexion",), "The strongest flexor of the hip.", "Lumbar and iliolumbar arteries",
           "A psoas abscess can track down to present as a groin swelling.", "Hip",
           aliases=("psoas", "iliacus")),
    Muscle("gluteus maximus", "Gluteus maximus",
           "Posterior ilium, dorsal sacrum and coccyx, and the sacrotuberous ligament",
           "Iliotibial tract and gluteal tuberosity of the femur", "Inferior gluteal nerve (L5–S2)",
           ("hip.extension", "hip.external_rotation"),
           "Extends and laterally rotates the hip — used to rise from sitting or climb stairs.",
           "Superior and inferior gluteal arteries", "", "Hip"),
    Muscle("gluteus medius", "Gluteus medius", "External surface of the ilium between the anterior and posterior gluteal lines",
           "Lateral surface of the greater trochanter", "Superior gluteal nerve (L4–S1)",
           ("hip.abduction", "hip.internal_rotation"),
           "Abducts and medially rotates the hip; keeps the pelvis level when standing on one leg.",
           "Superior gluteal artery",
           "Weakness gives a positive Trendelenburg sign — the pelvis drops on the unsupported side.",
           "Hip"),
    Muscle("tensor fasciae latae", "Tensor fasciae latae", "Anterior superior iliac spine and anterior iliac crest",
           "Iliotibial tract", "Superior gluteal nerve (L4–S1)",
           ("hip.abduction", "hip.flexion", "hip.internal_rotation"),
           "Abducts, flexes and medially rotates the hip; tenses the iliotibial tract to stabilise the knee.",
           "Lateral circumflex femoral artery", "", "Hip"),
    Muscle("sartorius", "Sartorius", "Anterior superior iliac spine",
           "Superior medial surface of the tibia (pes anserinus)", "Femoral nerve (L2, L3)",
           ("hip.flexion", "hip.abduction", "hip.external_rotation", "knee.flexion"),
           "Flexes, abducts and laterally rotates the hip and flexes the knee — the 'tailor's' position.",
           "Femoral artery", "The longest muscle in the body.", "Thigh"),
    Muscle("rectus femoris", "Rectus femoris", "Anterior inferior iliac spine and ilium above the acetabulum",
           "Tibial tuberosity via the quadriceps tendon, patella and patellar ligament",
           "Femoral nerve (L2–L4)", ("knee.extension", "hip.flexion"),
           "Extends the knee and, uniquely among the quadriceps, flexes the hip.",
           "Lateral circumflex femoral artery", "The patellar (knee-jerk) reflex tests L3–L4.", "Thigh"),
    Muscle("vastus lateralis", "Vastus lateralis", "Greater trochanter and lateral lip of the linea aspera",
           "Tibial tuberosity via the patella and patellar ligament", "Femoral nerve (L2–L4)",
           ("knee.extension",), "Extends the knee; the largest part of the quadriceps.",
           "Lateral circumflex femoral artery", "Common site for intramuscular injection in infants.", "Thigh"),
    Muscle("vastus medialis", "Vastus medialis", "Intertrochanteric line and medial lip of the linea aspera",
           "Tibial tuberosity via the patella and patellar ligament", "Femoral nerve (L2–L4)",
           ("knee.extension",), "Extends the knee; its oblique fibres keep the patella tracking centrally.",
           "Femoral and profunda femoris arteries",
           "Wasting after knee injury is an early, visible sign.", "Thigh"),
    Muscle("adductor longus", "Adductor longus", "Body of the pubis below the pubic crest",
           "Middle third of the linea aspera", "Obturator nerve (L2–L4)", ("hip.adduction", "hip.flexion"),
           "Adducts and assists flexion of the hip.", "Profunda femoris artery",
           "The usual site of a 'groin strain' in athletes.", "Thigh"),
    Muscle("adductor magnus", "Adductor magnus",
           "Adductor part: inferior pubic and ischial rami. Hamstring part: ischial tuberosity",
           "Gluteal tuberosity, linea aspera and adductor tubercle of the femur",
           "Obturator nerve (adductor part); tibial division of the sciatic nerve (hamstring part)",
           ("hip.adduction", "hip.extension"),
           "Adducts the hip; the hamstring part also extends it.", "Profunda femoris and obturator arteries",
           "", "Thigh"),
    Muscle("biceps femoris", "Biceps femoris",
           "Long head: ischial tuberosity. Short head: linea aspera", "Head of the fibula",
           "Long head: tibial division of the sciatic nerve. Short head: common fibular division (L5–S2)",
           ("knee.flexion", "hip.extension"),
           "Flexes the knee and laterally rotates the flexed knee; the long head extends the hip.",
           "Perforating branches of the profunda femoris artery",
           "Hamstring strains are among the commonest sprinting injuries.", "Thigh"),
    Muscle("semitendinosus", "Semitendinosus", "Ischial tuberosity", "Superior medial surface of the tibia (pes anserinus)",
           "Tibial division of the sciatic nerve (L5–S2)", ("knee.flexion", "hip.extension"),
           "Flexes the knee, extends the hip and medially rotates the flexed knee.",
           "Perforating branches of the profunda femoris artery",
           "Its tendon is a common graft for anterior cruciate ligament reconstruction.", "Thigh"),
    Muscle("semimembranosus", "Semimembranosus", "Ischial tuberosity", "Posterior surface of the medial tibial condyle",
           "Tibial division of the sciatic nerve (L5–S2)", ("knee.flexion", "hip.extension"),
           "Flexes the knee, extends the hip and medially rotates the flexed knee.",
           "Perforating branches of the profunda femoris artery", "", "Thigh"),
    # -- leg ---------------------------------------------------------------
    Muscle("gastrocnemius", "Gastrocnemius",
           "Medial head: above the medial femoral condyle. Lateral head: lateral femoral condyle",
           "Posterior surface of the calcaneus via the calcaneal (Achilles) tendon", "Tibial nerve (S1, S2)",
           ("ankle.plantarflexion", "knee.flexion"),
           "Plantarflexes the ankle — propulsion in walking, running and jumping — and flexes the knee.",
           "Sural arteries", "The ankle-jerk reflex tests S1; Achilles rupture is tested with Thompson's squeeze test.",
           "Leg"),
    Muscle("soleus", "Soleus", "Posterior head and upper fibula, and the soleal line of the tibia",
           "Posterior surface of the calcaneus via the calcaneal tendon", "Tibial nerve (S1, S2)",
           ("ankle.plantarflexion",),
           "Plantarflexes the ankle; the postural 'peripheral heart' whose contraction pumps venous blood upward.",
           "Posterior tibial, fibular and sural arteries",
           "Its venous sinuses are a frequent site of deep vein thrombosis.", "Leg"),
    Muscle("tibialis anterior", "Tibialis anterior",
           "Lateral condyle and upper half of the lateral surface of the tibia, and interosseous membrane",
           "Medial cuneiform and base of the first metatarsal", "Deep fibular nerve (L4, L5)",
           ("ankle.dorsiflexion", "foot.inversion"),
           "Dorsiflexes the ankle and inverts the foot; lifts the toes clear of the ground in swing phase.",
           "Anterior tibial artery",
           "Paralysis (common fibular nerve injury at the fibular neck) causes foot drop and a high-stepping gait.",
           "Leg"),
    Muscle("fibularis longus", "Fibularis (peroneus) longus", "Head and upper two-thirds of the lateral fibula",
           "Plantar base of the first metatarsal and medial cuneiform", "Superficial fibular nerve (L5–S2)",
           ("foot.eversion", "ankle.plantarflexion"), "Everts the foot and weakly plantarflexes the ankle; "
                                                      "supports the transverse arch.",
           "Fibular artery", "", "Leg", aliases=("peroneus longus",)),
    Muscle("extensor digitorum longus", "Extensor digitorum longus",
           "Lateral tibial condyle and upper three-quarters of the anterior fibula",
           "Middle and distal phalanges of toes 2–5", "Deep fibular nerve (L5, S1)",
           ("ankle.dorsiflexion", "foot.eversion"), "Extends the lateral four toes and dorsiflexes the ankle.",
           "Anterior tibial artery", "", "Leg"),
)

MUSCLE_BY_KEY: Dict[str, Muscle] = {m.key: m for m in MUSCLES}


def muscle_for_structure(name: str) -> Optional[Muscle]:
    """Knowledge-base entry for a structure name, longest key first
    ("biceps femoris" must not resolve to "biceps brachii")."""
    low = (name or "").lower()
    best = None
    for muscle in MUSCLES:
        for key in (muscle.key,) + muscle.aliases:
            if key in low and (best is None or len(key) > best[0]):
                best = (len(key), muscle)
    return best[1] if best else None


def side_of(name: str) -> str:
    low = (name or "").lower()
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    return ""


def agonists_for(motion_id: str) -> List[Muscle]:
    return [m for m in MUSCLES if motion_id in m.actions]


def antagonists_for(motion_id: str) -> List[Muscle]:
    opposite = OPPOSITE.get(motion_id, "")
    return [m for m in MUSCLES if opposite and opposite in m.actions]


def describe(muscle: Muscle) -> str:
    """Plain-text card used by the UI and the AI context."""
    lines = [muscle.name,
             f"Origin: {muscle.origin}",
             f"Insertion: {muscle.insertion}",
             f"Innervation: {muscle.innervation}"]
    if muscle.blood_supply:
        lines.append(f"Blood supply: {muscle.blood_supply}")
    lines.append(f"Action: {muscle.action_text}")
    if muscle.actions:
        lines.append("Joint motions: " + "; ".join(
            f"{MOTIONS[a].label} (normal {MOTIONS[a].normal_range})" for a in muscle.actions if a in MOTIONS))
    if muscle.clinical:
        lines.append(f"Clinical: {muscle.clinical}")
    return "\n".join(lines)
