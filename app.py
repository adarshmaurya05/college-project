import hashlib
import tempfile
from pathlib import Path

import pyttsx3
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = Path(__file__).parent / "best.pt"

st.set_page_config(
    page_title="Indian Traffic Sign Detection",
    page_icon="🚦",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Hide only the Deploy button */
    [data-testid="stAppDeployButton"] {
        display: none;
    }

    /* Hide top decoration */
    [data-testid="stDecoration"] {
        display: none;
    }

    .block-container {
        padding-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =========================================================
# NATURAL CLASS NAMES
# =========================================================

SIGN_NAMES = {

    # -----------------------------------------------------
    # Prohibitory / regulatory
    # -----------------------------------------------------

    "ALL_MOTOR_VEHICLE_PROHIBITED":
        "no motor vehicles sign",

    "AXLE_LOAD_LIMIT":
        "axle load limit sign",

    "BULLOCK_AND_HANDCART_PROHIBITED":
        "bullock and handcart prohibited sign",

    "BULLOCK_PROHIBITED":
        "bullock prohibited sign",

    "CYCLE_PROHIBITED":
        "cycle prohibited sign",

    "HANDCART_PROHIBITED":
        "handcart prohibited sign",

    "HEIGHT_LIMIT":
        "height limit sign",

    "HORN_PROHIBITED":
        "horn prohibited sign",

    "LENGTH_LIMIT":
        "length limit sign",

    "LOAD_LIMIT":
        "load limit sign",

    "NO_ENTRY":
        "no entry sign",

    "NO_PARKING":
        "no parking sign",

    "NO_STOPPING_OR_STANDING":
        "no stopping or standing sign",

    "OVERTAKING_PROHIBITED":
        "overtaking prohibited sign",

    "PEDESTRIAN_PROHIBITED":
        "pedestrian prohibited sign",

    "STRAIGHT_PROHIBITED":
        "straight movement prohibited sign",

    "TONGA_PROHIBITED":
        "tonga prohibited sign",

    "TRUCK_PROHIBITED":
        "truck prohibited sign",

    "TURN_RIGHT":
        "turn right sign",

    "U_TURN_PROHIBITED":
        "U-turn prohibited sign",

    "U_TURN":
        "U-turn sign",

    "LEFT_TURN_PROHIBITED":
        "left turn prohibited sign",

    "RIGHT_TURN_PROHIBITED":
        "right turn prohibited sign",

    "PASS_EITHER_SIDE":
        "pass either side sign",

    # -----------------------------------------------------
    # Mandatory
    # -----------------------------------------------------

    "COMPULSARY_AHEAD":
        "compulsory ahead sign",

    "COMPULSARY_AHEAD_OR_TURN_LEFT":
        "compulsory ahead or turn left sign",

    "COMPULSARY_AHEAD_OR_TURN_RIGHT":
        "compulsory ahead or turn right sign",

    "COMPULSARY_CYCLE_TRACK":
        "compulsory cycle track sign",

    "COMPULSARY_KEEP_LEFT":
        "compulsory keep left sign",

    "COMPULSARY_KEEP_RIGHT":
        "compulsory keep right sign",

    "COMPULSARY_MINIMUM_SPEED":
        "compulsory minimum speed sign",

    "COMPULSARY_SOUND_HORN":
        "compulsory sound horn sign",

    "COMPULSARY_TURN_LEFT":
        "compulsory turn left sign",

    "COMPULSARY_TURN_LEFT_AHEAD":
        "compulsory turn left ahead sign",

    "COMPULSARY_TURN_RIGHT":
        "compulsory turn right sign",

    "COMPULSARY_TURN_RIGHT_AHEAD":
        "compulsory turn right ahead sign",

    # -----------------------------------------------------
    # Warning / road condition
    # -----------------------------------------------------

    "BARRIER_AHEAD":
        "barrier ahead warning",

    "CATTLE":
        "cattle warning",

    "CROSS_ROAD":
        "crossroad warning",

    "CYCLE_CROSSING":
        "cycle crossing warning",

    "DANGEROUS_DIP":
        "dangerous dip warning",

    "FALLING_ROCKS":
        "falling rocks warning",

    "FERRY":
        "ferry warning",

    "GAP_IN_MEDIAN":
        "gap in median warning",

    "GUARDED_LEVEL_CROSSING":
        "guarded level crossing warning",

    "HUMP_OR_ROUGH_ROAD":
        "hump or rough road warning",

    "LEFT_HAIR_PIN_BEND":
        "left hairpin bend warning",

    "LEFT_HAND_CURVE":
        "left hand curve warning",

    "LEFT_REVERSE_BEND":
        "left reverse bend warning",

    "LOOSE_GRAVEL":
        "loose gravel warning",

    "MEN_AT_WORK":
        "men at work warning",

    "NARROW_BRIDGE":
        "narrow bridge warning",

    "NARROW_ROAD_AHEAD":
        "narrow road ahead warning",

    "PEDESTRIAN_CROSSING":
        "pedestrian crossing warning",

    "PRIORITY_FOR_ONCOMING_VEHICLES":
        "priority for oncoming vehicles sign",

    "QUAY_SIDE_OR_RIVER_BANK":
        "quay side or river bank warning",

    "RIGHT_HAIR_PIN_BEND":
        "right hairpin bend warning",

    "RIGHT_HAND_CURVE":
        "right hand curve warning",

    "RIGHT_REVERSE_BEND":
        "right reverse bend warning",

    "ROAD_WIDENS_AHEAD":
        "road widens ahead warning",

    "ROUNDABOUT":
        "roundabout warning",

    "SCHOOL_AHEAD":
        "school ahead warning",

    "SIDE_ROAD_LEFT":
        "side road on the left warning",

    "SIDE_ROAD_RIGHT":
        "side road on the right warning",

    "SLIPPERY_ROAD":
        "slippery road warning",

    "STEEP_ASCENT":
        "steep ascent warning",

    "STEEP_DESCENT":
        "steep descent warning",

    "T_INTERSECTION":
        "T-intersection warning",

    "Y_INTERSECTION":
        "Y-intersection warning",

    # -----------------------------------------------------
    # Information / other
    # -----------------------------------------------------

    "DIRECTION":
        "direction sign",

    "GIVE_WAY":
        "give way sign",

    "HOSPITAL_AHEAD":
        "hospital ahead sign",

    "PETROL_PUMP_AHEAD":
        "petrol pump ahead sign",

    "PRIORITY_FOR_ONCOMING_VEHICLES":
        "priority for oncoming vehicles sign",

    "RESTRICTION_ENDS":
        "restriction ends sign",

    "TRAFFIC_SIGNAL":
        "traffic signal warning",

    # -----------------------------------------------------
    # Speed limits
    # -----------------------------------------------------

    "SPEED_LIMIT_5":
        "speed limit 5",

    "SPEED_LIMIT_15":
        "speed limit 15",

    "SPEED_LIMIT_20":
        "speed limit 20",

    "SPEED_LIMIT_30":
        "speed limit 30",

    "SPEED_LIMIT_40":
        "speed limit 40",

    "SPEED_LIMIT_50":
        "speed limit 50",

    "SPEED_LIMIT_60":
        "speed limit 60",

    "SPEED_LIMIT_70":
        "speed limit 70",

    "SPEED_LIMIT_80":
        "speed limit 80",

    "SPEED_LIMIT_100":
        "speed limit 100",

    # -----------------------------------------------------
    # Other
    # -----------------------------------------------------

    "STOP":
        "stop sign",

    "GAP_IN_MEDIAN":
        "gap in median warning",

    "LEFT_HAIR_PIN_BEND":
        "left hairpin bend warning",

    "RIGHT_HAIR_PIN_BEND":
        "right hairpin bend warning",
}


# =========================================================
# SPEECH FUNCTIONS
# =========================================================

def get_natural_name(class_name):
    """
    Convert YOLO class name into a natural phrase.
    """

    if class_name in SIGN_NAMES:
        return SIGN_NAMES[class_name]

    return class_name.replace("_", " ").lower()


def get_confidence_level(confidence):
    """
    Convert numerical confidence into natural language.
    """

    if confidence >= 0.85:
        return "very high"

    if confidence >= 0.70:
        return "high"

    if confidence >= 0.50:
        return "moderate"

    return "low"


def make_single_sentence(class_name, confidence):
    """
    Generate natural speech for one detection.
    """

    name = get_natural_name(class_name)

    # Strong detection
    if confidence >= 0.85:

        return (
            f"{name.capitalize()} detected. "
            f"Confidence is very high."
        )

    # Good detection
    elif confidence >= 0.70:

        return (
            f"{name.capitalize()} detected "
            f"with high confidence."
        )

    # Moderate detection
    elif confidence >= 0.50:

        return (
            f"A {name} may be present. "
            f"Confidence is moderate."
        )

    # Weak detection
    else:

        return (
            f"Possible {name}. "
            f"Detection confidence is low."
        )


def make_announcement(detections):
    """
    Generate a natural spoken message for all detections.
    """

    if not detections:
        return "No traffic signs detected."

    # -----------------------------------------------------
    # One detection
    # -----------------------------------------------------

    if len(detections) == 1:

        detection = detections[0]

        return make_single_sentence(
            detection["Traffic Sign"],
            detection["Confidence"]
        )

    # -----------------------------------------------------
    # Multiple detections
    # -----------------------------------------------------

    count = len(detections)

    if count == 2:
        message = "Two traffic signs detected. "
    else:
        message = f"{count} traffic signs detected. "

    sentences = []

    for detection in detections:

        name = get_natural_name(
            detection["Traffic Sign"]
        )

        confidence = detection["Confidence"]

        if confidence >= 0.85:

            sentence = (
                f"{name.capitalize()}, "
                f"confidence is very high."
            )

        elif confidence >= 0.70:

            sentence = (
                f"{name.capitalize()}, "
                f"confidence is high."
            )

        elif confidence >= 0.39:

            sentence = (
                f"Possible {name}, "
                f"confidence is moderate."
            )

        else:

            sentence = (
                f"Possible {name}, "
                f"confidence is low."
            )

        sentences.append(sentence)

    return message + " ".join(sentences)


def speak_text(text):
    """
    Speak announcement using Windows text-to-speech.
    """

    try:

        engine = pyttsx3.init()

        engine.setProperty("rate", 130)
        engine.setProperty("volume", 1.0)

        engine.say(text)
        engine.runAndWait()

        engine.stop()

    except Exception as e:

        st.warning(
            f"Voice announcement could not be played: {e}"
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Detection Settings")

    confidence_percent = st.slider(
        "Confidence threshold",
        min_value=10,
        max_value=90,
        value=25,
        step=5,
        format="%d%%",
    )

    confidence_threshold = confidence_percent / 100.0

    image_size = st.selectbox(
        "Inference resolution",
        options=[640, 960, 1280],
        index=2,
    )

    st.divider()

    st.subheader("Model Information")

    st.write("Model: YOLOv8s")
    st.write("Classes: 89")
    st.write("Device: CPU")
    st.write(
        f"Resolution: {image_size} × {image_size}"
    )


# =========================================================
# PAGE HEADER
# =========================================================

st.title("Indian Traffic Sign Detection")

st.write(
    "Upload a road image to detect Indian traffic signs "
    "using YOLOv8."
)

st.divider()


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is None:

    st.info(
        "Upload an image to begin traffic-sign detection."
    )

    st.stop()


# =========================================================
# LOAD IMAGE
# =========================================================

try:

    image = Image.open(uploaded_file).convert("RGB")

except Exception as e:

    st.error(f"Could not read the image: {e}")
    st.stop()


width, height = image.size

st.caption(
    f"{uploaded_file.name}  •  "
    f"{width} × {height} pixels"
)


# =========================================================
# TEMPORARY IMAGE
# =========================================================

suffix = Path(uploaded_file.name).suffix

with tempfile.NamedTemporaryFile(
    suffix=suffix,
    delete=False,
) as tmp:

    image.save(tmp.name)

    temp_image_path = tmp.name


# =========================================================
# RUN YOLO
# =========================================================

with st.spinner("Analyzing image..."):

    try:

        results = model.predict(
            source=temp_image_path,
            conf=confidence_threshold,
            imgsz=image_size,
            device="cpu",
            verbose=False,
        )

    except Exception as e:

        st.error(
            f"Detection failed: {e}"
        )

        st.stop()


result = results[0]


# =========================================================
# NO DETECTION
# =========================================================

if len(result.boxes) == 0:

    st.subheader("Detection Details")

    st.success(
        "No traffic signs detected."
    )

    st.divider()

    st.subheader("Image")

    st.image(
        image,
        use_container_width=True,
    )

    st.stop()


# =========================================================
# EXTRACT DETECTIONS
# =========================================================

detections = []

for box in result.boxes:

    class_id = int(box.cls[0])

    score = float(box.conf[0])

    class_name = result.names[class_id]

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    detections.append(
        {
            "Traffic Sign": class_name,
            "Confidence": score,
            "x1": round(x1),
            "y1": round(y1),
            "x2": round(x2),
            "y2": round(y2),
        }
    )


# Highest confidence first

detections.sort(
    key=lambda d: d["Confidence"],
    reverse=True,
)


# =========================================================
# DETECTION SUMMARY
# =========================================================

total_detections = len(detections)

unique_signs = len(
    set(
        d["Traffic Sign"]
        for d in detections
    )
)

highest_confidence = detections[0]["Confidence"]


st.subheader("Detection Details")


metric1, metric2, metric3 = st.columns(3)


with metric1:

    st.metric(
        "Total detections",
        total_detections,
    )


with metric2:

    st.metric(
        "Unique signs",
        unique_signs,
    )


with metric3:

    st.metric(
        "Highest confidence",
        f"{highest_confidence * 100:.1f}%",
    )


# =========================================================
# DETECTION TABLE
# =========================================================

table_data = []

for detection in detections:

    table_data.append(
        {
            "Traffic Sign":
                detection["Traffic Sign"],

            "Confidence":
                f"{detection['Confidence'] * 100:.1f}%",

            "Bounding Box":
                (
                    f"({detection['x1']}, "
                    f"{detection['y1']}) → "
                    f"({detection['x2']}, "
                    f"{detection['y2']})"
                ),
        }
    )


st.dataframe(
    table_data,
    use_container_width=True,
    hide_index=True,
)




# =========================================================
# IMAGES — BELOW DETECTION DETAILS
# =========================================================

st.divider()

left, right = st.columns(2)


with left:

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True,
    )


with right:

    st.subheader("Detection Result")

    annotated_image = result.plot()

    st.image(
        annotated_image,
        use_container_width=True,
    )

    # =========================================================
# VOICE ANNOUNCEMENT
# =========================================================

announcement = make_announcement(detections)

st.info(
    f"🔊 {announcement}"
)


# =========================================================
# SPEAK ONLY ONCE PER RESULT
# =========================================================

signature_text = (
    uploaded_file.name
    + str(uploaded_file.size)
    + str(confidence_threshold)
    + str(image_size)
)

result_signature = hashlib.md5(
    signature_text.encode()
).hexdigest()


if (
    "last_spoken_signature"
    not in st.session_state
):

    st.session_state.last_spoken_signature = None


if (
    st.session_state.last_spoken_signature
    != result_signature
):

    speak_text(announcement)

    st.session_state.last_spoken_signature = (
        result_signature
    )
