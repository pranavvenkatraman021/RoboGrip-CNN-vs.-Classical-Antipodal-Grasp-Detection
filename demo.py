#imports
import os
import json
import time
import urllib.request

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import (
    DATA_ROOT,
    BACKGROUNDS_DIR,
    MIN_DIST,
    MAX_DIST,
    PLATE_THICKNESS,
    WIDTH_MARGIN,
    TIE_BREAK,
    TIE_TOL,
    MASK_CHECK,
    PLATE_CAP_RATIO
)

from data_loading import (
    load_rgb,
    load_depth,
    parse_grasp_rectangles,
    convert
)

from baseline import load_backgrounds
from evaluate_baseline import (
    predict_one,
    rectangle_iou,
    angle_diff_deg,
    is_correct_grasp
)

from evaluate_cnn import (
    load_trained_model,
    preprocess,
    output_to_corners
)


#files and verified results
SPLIT_PATH = "dataset_split.json"
MODEL_PATH = "grasp_model_multigt_best.pth"

BASELINE_ACCURACY = 32.03

#change this only if your final verified evaluation prints 73.39%
CNN_ACCURACY = 72.39


#presentation-matched palette
DECK_NAVY = "#174A8B"
DECK_BLUE = "#2765B8"
DECK_LIGHT_BLUE = "#EAF2FC"
DECK_GREEN = "#2FA84F"
DECK_RED = "#D83B3B"
DECK_ORANGE = "#F28E2B"
DECK_TEXT = "#243B53"
DECK_GRAY = "#66788A"
DECK_BORDER = "#D7E2F0"
DECK_BACKGROUND = "#F8FAFD"
WHITE = "#FFFFFF"

#RGB values for drawing on RGB images
GROUND_TRUTH_RGB = (47, 168, 79)
PREDICTION_RGB = (216, 59, 59)


def configure_poppins_for_matplotlib():
    """
    Uses an installed Poppins font when available. If Poppins is not already
    installed, attempts to download its regular and semibold font files into
    a local cache. Matplotlib falls back safely if the download is blocked.
    """
    try:
        font_manager.findfont(
            "Poppins",
            fallback_to_default=False
        )
        plt.rcParams["font.family"] = "Poppins"
        return
    except ValueError:
        pass

    font_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".robogrip_fonts"
    )
    os.makedirs(font_directory, exist_ok=True)

    font_sources = {
        "Poppins-Regular.ttf": (
            "https://raw.githubusercontent.com/google/fonts/main/"
            "ofl/poppins/Poppins-Regular.ttf"
        ),
        "Poppins-SemiBold.ttf": (
            "https://raw.githubusercontent.com/google/fonts/main/"
            "ofl/poppins/Poppins-SemiBold.ttf"
        )
    }

    downloaded_fonts = []

    for filename, url in font_sources.items():
        destination = os.path.join(font_directory, filename)

        try:
            if not os.path.exists(destination):
                urllib.request.urlretrieve(url, destination)

            font_manager.fontManager.addfont(destination)
            downloaded_fonts.append(destination)
        except Exception:
            continue

    if downloaded_fonts:
        plt.rcParams["font.family"] = "Poppins"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"


configure_poppins_for_matplotlib()


#page setup
st.set_page_config(
    page_title="RoboGrip Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


#complete presentation-matched interface styling
st.markdown(
    f"""
    <style>
        @import url(
            'https://fonts.googleapis.com/css2?'
            'family=Poppins:wght@400;500;600;700&display=swap'
        );

        :root {{
            --deck-navy: {DECK_NAVY};
            --deck-blue: {DECK_BLUE};
            --deck-green: {DECK_GREEN};
            --deck-red: {DECK_RED};
            --deck-text: {DECK_TEXT};
            --deck-gray: {DECK_GRAY};
            --deck-border: {DECK_BORDER};
            --deck-background: {DECK_BACKGROUND};
        }}

        .stApp {{
            background-color: {DECK_BACKGROUND};
            background-image:
                linear-gradient(
                    135deg,
                    rgba(39, 101, 184, 0.028) 25%,
                    transparent 25%
                ),
                linear-gradient(
                    315deg,
                    rgba(39, 101, 184, 0.028) 25%,
                    transparent 25%
                );
            background-position: 0 0, 90px 90px;
            background-size: 180px 180px;
        }}

        .stApp,
        .stApp p,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp button,
        .stApp input,
        .stApp label,
        .stApp textarea {{
            font-family: "Poppins", sans-serif;
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }}

        .hero-banner {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(
                110deg,
                {DECK_NAVY} 0%,
                {DECK_BLUE} 100%
            );
            padding: 1.7rem 2rem 1.55rem 2rem;
            border-radius: 22px;
            margin-bottom: 1.5rem;
            box-shadow: 0 9px 24px rgba(23, 74, 139, 0.18);
            text-align: center;
        }}

        .hero-banner::after {{
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -90px;
            top: -145px;
            background: rgba(255, 255, 255, 0.07);
            transform: rotate(45deg);
        }}

        .hero-kicker {{
            position: relative;
            z-index: 2;
            color: #D7E6FA !important;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.14rem;
            margin: 0 0 0.35rem 0;
        }}

        .main-title {{
            position: relative;
            z-index: 2;
            color: white !important;
            font-size: 2.9rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0;
        }}

        .subtitle {{
            position: relative;
            z-index: 2;
            color: #F0F5FD !important;
            font-size: 1.08rem;
            font-weight: 400;
            margin: 0.55rem 0 0 0;
        }}

        h1, h2, h3, h4 {{
            color: {DECK_NAVY} !important;
            letter-spacing: -0.02rem;
        }}

        h2 {{
            font-weight: 700 !important;
        }}

        h3 {{
            font-weight: 600 !important;
        }}

        p, li {{
            color: {DECK_TEXT};
        }}

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    #F2F7FD 0%,
                    #EAF2FC 100%
                );
            border-right: 1px solid {DECK_BORDER};
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 1.25rem;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {DECK_NAVY} !important;
        }}

        div.stButton > button {{
            width: 100%;
            min-height: 3rem;
            color: white;
            background: linear-gradient(
                100deg,
                {DECK_NAVY},
                {DECK_BLUE}
            );
            border: none;
            border-radius: 12px;
            font-weight: 600;
            box-shadow: 0 5px 13px rgba(23, 74, 139, 0.20);
            transition: 0.18s ease;
        }}

        div.stButton > button:hover {{
            color: white;
            border: none;
            background: {DECK_NAVY};
            transform: translateY(-1px);
            box-shadow: 0 7px 16px rgba(23, 74, 139, 0.24);
        }}

        div.stButton > button:active {{
            transform: translateY(0);
        }}

        [data-testid="stSelectbox"] > div > div {{
            border-color: {DECK_BORDER};
            border-radius: 10px;
        }}

        [data-testid="stImage"] img {{
            border-radius: 14px;
            border: 1px solid {DECK_BORDER};
            box-shadow: 0 5px 15px rgba(23, 74, 139, 0.10);
        }}

        [data-testid="stImageCaption"] {{
            color: {DECK_GRAY};
            font-weight: 500;
        }}

        [data-testid="stMetric"] {{
            background: white;
            border: 1px solid {DECK_BORDER};
            border-left: 5px solid {DECK_BLUE};
            border-radius: 12px;
            padding: 0.8rem 1rem;
            box-shadow: 0 3px 10px rgba(23, 74, 139, 0.08);
        }}

        [data-testid="stMetricLabel"] {{
            color: {DECK_GRAY};
        }}

        [data-testid="stMetricValue"] {{
            color: {DECK_NAVY};
            font-weight: 700;
        }}

        [data-testid="stAlert"] {{
            border-radius: 12px;
            border: 1px solid {DECK_BORDER};
            box-shadow: 0 2px 7px rgba(23, 74, 139, 0.05);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {DECK_BORDER};
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(23, 74, 139, 0.07);
        }}

        .result-correct {{
            display: inline-block;
            color: white !important;
            background: {DECK_GREEN};
            padding: 0.38rem 0.9rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.015rem;
            box-shadow: 0 3px 8px rgba(47, 168, 79, 0.20);
        }}

        .result-incorrect {{
            display: inline-block;
            color: white !important;
            background: {DECK_RED};
            padding: 0.38rem 0.9rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.015rem;
            box-shadow: 0 3px 8px rgba(216, 59, 59, 0.20);
        }}

        .method-card {{
            background: white;
            border: 1px solid {DECK_BORDER};
            border-top: 6px solid {DECK_BLUE};
            border-radius: 16px;
            padding: 0.9rem 1rem 0.7rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 13px rgba(23, 74, 139, 0.08);
        }}

        .method-card.cnn {{
            border-top-color: {DECK_GREEN};
        }}

        .method-name {{
            color: {DECK_NAVY} !important;
            font-size: 1.22rem;
            font-weight: 600;
            margin: 0 0 0.35rem 0;
        }}

        .method-description {{
            color: {DECK_GRAY} !important;
            min-height: 3.1rem;
            line-height: 1.45;
            margin: 0;
        }}

        .input-card {{
            background: white;
            border: 1px solid {DECK_BORDER};
            border-radius: 15px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 13px rgba(23, 74, 139, 0.08);
        }}

        .criterion-formula {{
            display: block;
            width: 100%;
            box-sizing: border-box;
            color: {DECK_NAVY} !important;
            background: white;
            border: 1px solid {DECK_BORDER};
            border-left: 5px solid {DECK_BLUE};
            border-radius: 11px;
            padding: 0.72rem 0.55rem;
            margin: 0.55rem 0;
            text-align: center;
            font-family: "Poppins", sans-serif;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.35;
            white-space: nowrap;
            box-shadow: 0 3px 9px rgba(23, 74, 139, 0.07);
        }}

        .legend-row {{
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 0.8rem;
            color: {DECK_GRAY};
            font-size: 0.9rem;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .legend-green {{
            width: 23px;
            border-top: 3px dashed {DECK_GREEN};
        }}

        .legend-red {{
            width: 23px;
            border-top: 3px solid {DECK_RED};
        }}

        hr {{
            border-color: {DECK_BORDER};
        }}

        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


#presentation-style header
st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-kicker">RGB-D ROBOTIC GRASP DETECTION</p>
        <p class="main-title">RoboGrip</p>
        <p class="subtitle">
            Classical Antipodal Grasp Detection vs. ResNet-18 CNN
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_resources():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    backgrounds = load_backgrounds(BACKGROUNDS_DIR)

    model = load_trained_model(MODEL_PATH)
    model = model.to(device)
    model.eval()

    return backgrounds, model, device


@st.cache_data
def load_test_entries():
    with open(SPLIT_PATH, "r") as file:
        split = json.load(file)

    return split["test"]


def resolve_folder(entry):
    """
    Uses the path stored in dataset_split.json when it exists. Otherwise,
    reconstructs the Cornell folder from DATA_ROOT and the pcd identifier.
    """
    stored_folder = entry["folder"]
    pcd_id = entry["id"]

    if os.path.isdir(stored_folder):
        return stored_folder

    portable_folder = os.path.join(DATA_ROOT, pcd_id[:2])

    if os.path.isdir(portable_folder):
        return portable_folder

    raise FileNotFoundError(
        f"Could not locate the folder for pcd{pcd_id}. "
        f"Check CORNELL_ROOT or dataset_split.json."
    )


def draw_dashed_line(image, start, end, color, thickness=2):
    line_length = np.linalg.norm(end - start)
    number_segments = max(int(line_length // 10), 1)

    for segment in range(0, number_segments, 2):
        t1 = segment / number_segments
        t2 = min((segment + 1) / number_segments, 1.0)

        point_1 = np.round(
            start + t1 * (end - start)
        ).astype(int)

        point_2 = np.round(
            start + t2 * (end - start)
        ).astype(int)

        cv2.line(
            image,
            tuple(point_1),
            tuple(point_2),
            color,
            thickness,
            cv2.LINE_AA
        )


def draw_rectangles(
    rgb,
    ground_truths=None,
    prediction=None,
    show_ground_truth=True
):
    """
    Green dashed rectangles are human-labeled valid grasps.
    The red solid rectangle is the method's prediction.
    """
    output = rgb.copy()

    if show_ground_truth and ground_truths:
        for rectangle in ground_truths:
            points = np.round(rectangle).astype(np.int32)

            for index in range(4):
                draw_dashed_line(
                    output,
                    points[index],
                    points[(index + 1) % 4],
                    GROUND_TRUTH_RGB,
                    thickness=2
                )

    if prediction is not None:
        prediction_points = np.round(
            prediction
        ).astype(np.int32)

        cv2.polylines(
            output,
            [prediction_points],
            isClosed=True,
            color=PREDICTION_RGB,
            thickness=4,
            lineType=cv2.LINE_AA
        )

        center = np.round(
            np.mean(prediction, axis=0)
        ).astype(int)

        cv2.circle(
            output,
            tuple(center),
            radius=7,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        cv2.circle(
            output,
            tuple(center),
            radius=4,
            color=PREDICTION_RGB,
            thickness=-1,
            lineType=cv2.LINE_AA
        )

    return output


def evaluate_prediction(prediction, ground_truths):
    """
    If a prediction correctly matches any ground truth, selects the correct
    match with the greatest IoU. Otherwise, displays the greatest-IoU match.
    """
    if prediction is None:
        return {
            "correct": False,
            "iou": 0.0,
            "angle_error": None,
            "matched_index": None
        }

    comparisons = []

    for index, ground_truth in enumerate(ground_truths):
        iou = rectangle_iou(
            prediction,
            ground_truth
        )

        _, _, _, _, prediction_theta = convert(prediction)
        _, _, _, _, ground_truth_theta = convert(ground_truth)

        angle_error = angle_diff_deg(
            prediction_theta,
            ground_truth_theta
        )

        correct = is_correct_grasp(
            prediction,
            ground_truth
        )

        comparisons.append({
            "correct": correct,
            "iou": iou,
            "angle_error": angle_error,
            "matched_index": index
        })

    correct_matches = [
        result for result in comparisons
        if result["correct"]
    ]

    if correct_matches:
        return max(
            correct_matches,
            key=lambda result: result["iou"]
        )

    return max(
        comparisons,
        key=lambda result: result["iou"]
    )


def run_cnn(rgb, depth, model, device):
    tensor, scale, pad_x, pad_y, x_off, y_off = preprocess(
        rgb,
        depth
    )

    tensor = tensor.to(device)

    with torch.no_grad():
        output = (
            model(tensor)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    prediction = output_to_corners(
        output,
        scale,
        pad_x,
        pad_y,
        x_off,
        y_off
    )

    return prediction, output


def format_status(result):
    if result["correct"]:
        return "✓ CORRECT", "result-correct"

    return "✗ INCORRECT", "result-incorrect"


def show_metrics(result):
    status, css_class = format_status(result)

    st.markdown(
        f'<p class="{css_class}">{status}</p>',
        unsafe_allow_html=True
    )

    metric_column_1, metric_column_2 = st.columns(2)

    metric_column_1.metric(
        "Best IoU",
        f"{result['iou']:.3f}"
    )

    angle_value = (
        f"{result['angle_error']:.1f}°"
        if result["angle_error"] is not None
        else "N/A"
    )

    metric_column_2.metric(
        "Angular Error",
        angle_value
    )

    iou_passed = result["iou"] > 0.25

    angle_passed = (
        result["angle_error"] is not None
        and result["angle_error"] <= 30
    )

    st.caption(
        f"IoU criterion: {'passed' if iou_passed else 'failed'} · "
        f"Angle criterion: {'passed' if angle_passed else 'failed'}"
    )


def make_accuracy_chart():
    """
    Uses Poppins when available and the same blue/green visual language as
    the presentation's comparison charts.
    """
    labels = [
        "Classical\nBaseline",
        "ResNet-18\nCNN"
    ]

    values = [
        BASELINE_ACCURACY,
        CNN_ACCURACY
    ]

    colors = [
        DECK_BLUE,
        DECK_GREEN
    ]

    fig, ax = plt.subplots(figsize=(7, 3.7))

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.58
    )

    ax.set_ylabel(
        "Test Accuracy (%)",
        color=DECK_TEXT
    )

    ax.set_ylim(0, 100)

    ax.set_title(
        "Overall Test-Set Performance",
        color=DECK_NAVY,
        fontweight="semibold",
        pad=12
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DECK_BORDER)
    ax.spines["bottom"].set_color(DECK_BORDER)

    ax.tick_params(
        axis="both",
        colors=DECK_TEXT
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        color=DECK_BORDER,
        alpha=0.85
    )

    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            color=DECK_NAVY,
            fontweight="semibold"
        )

    fig.tight_layout()
    return fig


try:
    test_entries = load_test_entries()
except Exception as error:
    st.error(f"Could not load dataset split: {error}")
    st.stop()


#sidebar controls
with st.sidebar:
    st.header("Demo Controls")

    selected_index = st.selectbox(
        "Choose a held-out test image",
        options=range(len(test_entries)),
        format_func=lambda index:
            f"pcd{test_entries[index]['id']}"
    )

    selected_entry = test_entries[selected_index]

    show_ground_truth = st.checkbox(
        "Show ground-truth rectangles",
        value=True,
        help=(
            "Ground truth is used only after prediction to grade "
            "the output. It is not supplied to either method."
        )
    )

    run_button = st.button(
        "Run Grasp Detection",
        type="primary"
    )

    st.divider()
    st.subheader("Correctness Standard")

    st.write("A prediction is correct when:")

    st.markdown(
        """
        <div class="criterion-formula">IoU &gt; 0.25</div>
        <div class="criterion-formula">
            Angular error &le; 30&deg;
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "The prediction may match any valid human-labeled grasp."
    )


#load the selected test example
try:
    selected_folder = resolve_folder(selected_entry)
    selected_id = selected_entry["id"]

    rgb = load_rgb(
        selected_folder,
        selected_id
    )

    depth = load_depth(
        selected_folder,
        selected_id
    )

    ground_truths = parse_grasp_rectangles(
        f"{selected_folder}/pcd{selected_id}cpos.txt"
    )

except Exception as error:
    st.error(f"Could not load the selected example: {error}")
    st.stop()


st.subheader(f"Selected Input · pcd{selected_id}")

input_column_1, input_column_2, input_column_3 = st.columns(
    [1.2, 1.2, 1]
)

with input_column_1:
    st.image(
        rgb,
        caption="RGB image",
        use_container_width=True
    )

with input_column_2:
    depth_display = np.nan_to_num(
        depth.astype(np.float32),
        nan=0.0
    )

    if depth_display.max() > depth_display.min():
        depth_display = (
            depth_display - depth_display.min()
        ) / (
            depth_display.max() - depth_display.min()
        )

    st.image(
        depth_display,
        caption="Aligned depth image",
        clamp=True,
        use_container_width=True
    )

with input_column_3:
    st.markdown(
        f"""
        <div class="input-card">
            <h4>Input Information</h4>
            <p><strong>Image ID:</strong> pcd{selected_id}</p>
            <p>
                <strong>Valid labeled grasps:</strong>
                {len(ground_truths)}
            </p>
            <p><strong>RGB shape:</strong> {rgb.shape}</p>
            <p><strong>Depth shape:</strong> {depth.shape}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "The CNN receives RGB plus depth. The classical baseline "
        "uses RGB and empty-background reference images."
    )


if run_button:
    try:
        with st.spinner(
            "Running the classical baseline and ResNet-18 CNN..."
        ):
            backgrounds, model, device = load_resources()

            start_time = time.perf_counter()

            baseline_prediction, baseline_details = predict_one(
                selected_folder,
                selected_id,
                backgrounds,
                min_dist=MIN_DIST,
                max_dist=MAX_DIST,
                plate_thickness=PLATE_THICKNESS,
                width_margin=WIDTH_MARGIN,
                tie_break=TIE_BREAK,
                tie_tol=TIE_TOL,
                mask_check=MASK_CHECK,
                return_details=True
            )

            baseline_finish_time = time.perf_counter()

            cnn_prediction, raw_output = run_cnn(
                rgb,
                depth,
                model,
                device
            )

            cnn_finish_time = time.perf_counter()

        baseline_result = evaluate_prediction(
            baseline_prediction,
            ground_truths
        )

        cnn_result = evaluate_prediction(
            cnn_prediction,
            ground_truths
        )

        baseline_visual = draw_rectangles(
            rgb,
            ground_truths,
            baseline_prediction,
            show_ground_truth
        )

        cnn_visual = draw_rectangles(
            rgb,
            ground_truths,
            cnn_prediction,
            show_ground_truth
        )

        st.divider()
        st.header("Prediction Results")

        st.markdown(
            """
            <div class="legend-row">
                <div class="legend-item">
                    <span class="legend-green"></span>
                    Human-labeled valid grasp
                </div>
                <div class="legend-item">
                    <span class="legend-red"></span>
                    Method prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        baseline_column, cnn_column = st.columns(2)

        with baseline_column:
            st.markdown(
                """
                <div class="method-card">
                    <p class="method-name">
                        Classical Antipodal Baseline
                    </p>
                    <p class="method-description">
                        Segments the object, traces its contour,
                        estimates outward surface normals, and searches
                        for an antipodal contact pair.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                baseline_visual,
                use_container_width=True
            )

            show_metrics(baseline_result)

            if baseline_details is not None:
                st.write(
                    f"**Antipodal score:** "
                    f"{baseline_details['score']:.3f} / 2.000"
                )

                st.write(
                    f"**Predicted width:** "
                    f"{baseline_details['width']:.1f} pixels"
                )

            st.write(
                f"**Processing time:** "
                f"{baseline_finish_time - start_time:.3f} seconds"
            )

        with cnn_column:
            st.markdown(
                """
                <div class="method-card cnn">
                    <p class="method-name">ResNet-18 CNN</p>
                    <p class="method-description">
                        Processes a four-channel RGB-D tensor and
                        directly regresses the grasp center, dimensions,
                        and orientation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                cnn_visual,
                use_container_width=True
            )

            show_metrics(cnn_result)

            _, _, _, _, sin_2_theta, cos_2_theta = raw_output

            decoded_angle = np.degrees(
                0.5 * np.arctan2(
                    sin_2_theta,
                    cos_2_theta
                )
            )

            st.write(
                f"**Decoded grasp angle:** "
                f"{decoded_angle:.1f}°"
            )

            st.write(
                f"**Processing time:** "
                f"{cnn_finish_time - baseline_finish_time:.3f} seconds"
            )

        st.divider()
        st.subheader("Direct Comparison")

        comparison = pd.DataFrame({
            "Method": [
                "Classical baseline",
                "ResNet-18 CNN"
            ],
            "Correct": [
                "Yes" if baseline_result["correct"] else "No",
                "Yes" if cnn_result["correct"] else "No"
            ],
            "Best IoU": [
                round(baseline_result["iou"], 3),
                round(cnn_result["iou"], 3)
            ],
            "Angular error": [
                (
                    round(
                        baseline_result["angle_error"],
                        1
                    )
                    if baseline_result["angle_error"] is not None
                    else None
                ),
                (
                    round(
                        cnn_result["angle_error"],
                        1
                    )
                    if cnn_result["angle_error"] is not None
                    else None
                )
            ]
        })

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True
        )

        summary_column_1, summary_column_2 = st.columns(
            [1, 1]
        )

        with summary_column_1:
            st.pyplot(
                make_accuracy_chart(),
                use_container_width=True
            )

        with summary_column_2:
            st.markdown("#### Interpretation")

            if (
                cnn_result["correct"]
                and not baseline_result["correct"]
            ):
                st.success(
                    "For this image, the CNN finds a valid grasp "
                    "while the classical baseline fails."
                )

                st.write(
                    "This illustrates where learned RGB-D features "
                    "can handle geometry that does not produce a "
                    "clean contour-based antipodal pair."
                )

            elif (
                baseline_result["correct"]
                and cnn_result["correct"]
            ):
                st.success(
                    "Both approaches find a valid grasp."
                )

                st.write(
                    "This shows that classical geometry can remain "
                    "competitive when the object has a clear and "
                    "simple grasp structure."
                )

            elif (
                baseline_result["correct"]
                and not cnn_result["correct"]
            ):
                st.warning(
                    "The classical baseline succeeds while the CNN "
                    "does not on this example."
                )

                st.write(
                    "This failure case demonstrates that the learned "
                    "model is not universally better."
                )

            else:
                st.error(
                    "Neither method satisfies both correctness "
                    "criteria for this image."
                )

                st.write(
                    "The displayed IoU and angular error separate "
                    "localization, size, and orientation failures."
                )

            st.info(
                "Ground-truth rectangles were used only after "
                "inference to grade the outputs. They were not "
                "supplied to either prediction method."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

else:
    st.info(
        "Select a held-out test image in the sidebar, then click "
        "**Run Grasp Detection**."
    )