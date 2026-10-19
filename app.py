from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. 페이지 설정
# =========================================================

st.set_page_config(
    page_title="10-Year Chronic Health Risk",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# 2. 프로젝트 경로
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "results" / "models"
TABLE_DIR = PROJECT_DIR / "results" / "tables"


# =========================================================
# 3. 모델 불러오기
# =========================================================

@st.cache_resource
def load_models():

    overall_model = joblib.load(
        MODEL_DIR / "final_chronic_risk_model.joblib"
    )

    with open(
        MODEL_DIR / "final_model_features.json",
        "r",
        encoding="utf-8"
    ) as f:
        overall_features = json.load(f)

    with open(
        MODEL_DIR / "disease_model_info.json",
        "r",
        encoding="utf-8"
    ) as f:
        disease_info = json.load(f)

    with open(
        MODEL_DIR / "disease_model_features.json",
        "r",
        encoding="utf-8"
    ) as f:
        disease_features = json.load(f)

    disease_models = {}

    for key, info in disease_info.items():

        disease_models[key] = joblib.load(
            MODEL_DIR / info["model_file"]
        )

    return (
        overall_model,
        overall_features,
        disease_models,
        disease_features,
        disease_info
    )


(
    overall_model,
    overall_features,
    disease_models,
    disease_features,
    disease_info
) = load_models()


# =========================================================
# 4. BMI 계산
# =========================================================

def calculate_bmi(height_cm, weight_kg):

    height_m = height_cm / 100

    return weight_kg / (height_m ** 2)


# =========================================================
# 5. eGFR 계산
#    2021 CKD-EPI creatinine equation
# =========================================================

def calculate_egfr(creatinine, age, sex):

    if sex == 1:  # 남성

        k = 0.9
        alpha = -0.302
        sex_factor = 1.0

    else:  # 여성

        k = 0.7
        alpha = -0.241
        sex_factor = 1.012

    ratio = creatinine / k

    egfr = (
        142
        * min(ratio, 1) ** alpha
        * max(ratio, 1) ** (-1.200)
        * (0.9938 ** age)
        * sex_factor
    )

    return egfr


# =========================================================
# 6. 제목
# =========================================================

st.title(
    "🧠 10-Year Chronic Health Risk Predictor"
)

st.caption(
    "현재 건강정보를 기반으로 약 10년 후 "
    "만성 건강위험을 분석하는 AI 연구 모델"
)

st.divider()


# =========================================================
# 7. 사용자 입력
# =========================================================

st.subheader(
    "👤 현재 건강정보 입력"
)

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# 기본 정보
# ---------------------------------------------------------

with col1:

    st.markdown("### 👤 기본 정보")

    age = st.number_input(
        "나이",
        min_value=40,
        max_value=90,
        value=55
    )

    sex_text = st.selectbox(
        "성별",
        ["여성", "남성"]
    )

    sex = 2 if sex_text == "여성" else 1

    height = st.number_input(
        "키 (cm)",
        min_value=130.0,
        max_value=200.0,
        value=160.0,
        step=0.1
    )

    weight = st.number_input(
        "체중 (kg)",
        min_value=30.0,
        max_value=150.0,
        value=62.0,
        step=0.1
    )

    waist = st.number_input(
        "허리둘레 (cm)",
        min_value=50.0,
        max_value=150.0,
        value=82.0,
        step=0.1
    )


# ---------------------------------------------------------
# 혈압 / 혈당
# ---------------------------------------------------------

with col2:

    st.markdown("### ❤️ 혈압·혈당")

    sbp = st.number_input(
        "수축기 혈압 SBP (mmHg)",
        min_value=70,
        max_value=220,
        value=135
    )

    dbp = st.number_input(
        "이완기 혈압 DBP (mmHg)",
        min_value=40,
        max_value=140,
        value=85
    )

    glucose = st.number_input(
        "공복혈당 (mg/dL)",
        min_value=40.0,
        max_value=400.0,
        value=103.0,
        step=1.0
    )

    hba1c = st.number_input(
        "HbA1c (%)",
        min_value=3.0,
        max_value=15.0,
        value=6.0,
        step=0.1
    )


# ---------------------------------------------------------
# 지질 / 신장
# ---------------------------------------------------------

with col3:

    st.markdown("### 🧪 지질·신장")

    total_cholesterol = st.number_input(
        "총콜레스테롤 (mg/dL)",
        min_value=70.0,
        max_value=400.0,
        value=198.0,
        step=1.0
    )

    hdl = st.number_input(
        "HDL (mg/dL)",
        min_value=10.0,
        max_value=150.0,
        value=48.0,
        step=1.0
    )

    tg = st.number_input(
        "중성지방 TG (mg/dL)",
        min_value=20.0,
        max_value=1000.0,
        value=140.0,
        step=1.0
    )

    creatinine = st.number_input(
        "Creatinine (mg/dL)",
        min_value=0.1,
        max_value=10.0,
        value=0.9,
        step=0.1
    )


# =========================================================
# 8. 자동 계산
# =========================================================

bmi = calculate_bmi(
    height,
    weight
)

egfr = calculate_egfr(
    creatinine,
    age,
    sex
)


st.divider()

st.subheader(
    "📌 자동 계산 지표"
)

metric1, metric2 = st.columns(2)

with metric1:

    st.metric(
        "BMI",
        f"{bmi:.1f}"
    )

with metric2:

    st.metric(
        "eGFR",
        f"{egfr:.1f} mL/min/1.73m²"
    )


# =========================================================
# 9. 모델 입력 데이터
# =========================================================

person = pd.DataFrame(
    [
        {
            "T01_AGE": age,
            "T00_SEX": sex,

            "T01_SBP": sbp,
            "T01_DBP": dbp,

            "T01_BMI": bmi,
            "T01_WAIST": waist,

            "T01_GLU0": glucose,
            "T01_HBA1C": hba1c,

            "T01_TCHL": total_cholesterol,
            "T01_HDL": hdl,
            "T01_TG": tg,

            "T01_CREATININE": creatinine,
            "T01_EGFR": egfr
        }
    ]
)


# =========================================================
# 10. 예측 버튼
# =========================================================

st.divider()


if st.button(
    "🔍 10년 건강위험 분석하기",
    type="primary",
    use_container_width=True
):

    # =====================================================
    # 전체 위험 점수
    # =====================================================

    overall_input = person[
        overall_features
    ]

    overall_score = (
        overall_model
        .predict_proba(
            overall_input
        )[0, 1]
    )

    overall_percent = overall_score * 100


    # =====================================================
    # 질환별 AI 참고 점수
    # =====================================================

    disease_input = person[
        disease_features
    ]

    disease_scores = {}

    for key, model in disease_models.items():

        score = (
            model
            .predict_proba(
                disease_input
            )[0, 1]
        )

        disease_scores[key] = score * 100


    # =====================================================
    # 11. 전체 결과
    # =====================================================

    st.markdown(
        "## 📊 AI 예측 결과"
    )

    st.metric(
        "10-Year Chronic Health Risk Score",
        f"{overall_percent:.1f}%"
    )

    st.progress(
        float(
            min(overall_score, 1.0)
        )
    )

    st.caption(
        "AI 모델이 산출한 전체 만성 건강위험 참고 점수입니다. "
        "실제 질병 발생 확률을 의미하지 않습니다."
    )


    # =====================================================
    # 12. 질환별 결과
    # =====================================================

    st.markdown(
        "## 🔎 세부 건강위험"
    )

    d1, d2, d3, d4 = st.columns(4)


    with d1:

        st.markdown("### ❤️ 고혈압")

        st.metric(
            "AI 참고 위험 점수",
            f"{disease_scores['hypertension']:.1f}%"
        )


    with d2:

        st.markdown("### 🍬 당뇨")

        st.metric(
            "AI 참고 위험 점수",
            f"{disease_scores['diabetes']:.1f}%"
        )


    with d3:

        st.markdown("### 🩸 이상지질혈증")

        st.metric(
            "AI 참고 위험 점수",
            f"{disease_scores['dyslipidemia']:.1f}%"
        )


    with d4:

        st.markdown("### 🫘 신장기능 저하")

        st.metric(
            "AI 참고 위험 점수",
            f"{disease_scores['kidney']:.1f}%"
        )


    st.caption(
        "각 질환별 점수는 서로 다른 모델에서 산출된 연구용 참고 점수입니다. "
        "숫자를 서로 직접 비교해 가장 걸릴 가능성이 높은 질환으로 "
        "해석해서는 안 됩니다."
    )


    # =====================================================
    # 13. 개인 맞춤 건강관리 우선순위
    # =====================================================

    st.markdown(
        "## 🌱 나의 맞춤 건강관리 포인트"
    )

    st.caption(
        "현재 입력한 건강수치가 관리 기준에서 얼마나 벗어나 있는지를 "
        "바탕으로 개인별 관리 우선순위를 정합니다."
    )


    tips = []


    def add_tip(
        sort_score,
        level,
        title,
        reason,
        action
    ):

        tips.append(
            {
                "sort_score": sort_score,
                "level": level,
                "title": title,
                "reason": reason,
                "action": action
            }
        )


    # =====================================================
    # A. 혈압 우선순위 계산
    # =====================================================

    bp_score = 0


    if sbp >= 130:

        bp_score += (
            (sbp - 130) * 2.5
        )


    if dbp >= 80:

        bp_score += (
            (dbp - 80) * 3.0
        )


    if sbp >= 140 or dbp >= 90:

        bp_score += 40


    if sbp >= 160 or dbp >= 100:

        bp_score += 30


    if sbp >= 130 or dbp >= 80:

        if sbp >= 140 or dbp >= 90:

            level = 3

            title = (
                "❤️ 혈압을 우선적으로 확인해 보세요"
            )

        else:

            level = 2

            title = (
                "❤️ 혈압 관리에 신경 써 주세요"
            )


        add_tip(
            bp_score,
            level,
            title,

            (
                f"현재 입력한 혈압은 "
                f"{sbp}/{dbp} mmHg입니다."
            ),

            (
                "안정된 상태에서 혈압을 반복 측정해 기록해 보세요. "
                "짠 음식과 과도한 나트륨 섭취를 줄이고 "
                "규칙적인 신체활동을 유지하는 것이 도움이 됩니다. "
                "높은 수치가 반복되면 의료진과 상담해 주세요."
            )
        )


    # =====================================================
    # B. 혈당 우선순위 계산
    # =====================================================

    sugar_score = 0


    if glucose >= 100:

        sugar_score += (
            (glucose - 100) * 1.2
        )


    if hba1c >= 5.7:

        sugar_score += (
            (hba1c - 5.7) * 35
        )


    if glucose >= 126 or hba1c >= 6.5:

        sugar_score += 45


    if glucose >= 150 or hba1c >= 7.0:

        sugar_score += 25


    if glucose >= 100 or hba1c >= 5.7:

        if glucose >= 126 or hba1c >= 6.5:

            level = 3

            title = (
                "🍬 혈당 상태를 우선 확인해 보세요"
            )

        else:

            level = 2

            title = (
                "🍬 혈당 관리에 관심을 가져보세요"
            )


        add_tip(
            sugar_score,
            level,
            title,

            (
                f"공복혈당 {glucose:.0f} mg/dL, "
                f"HbA1c {hba1c:.1f}%가 입력되었습니다."
            ),

            (
                "당류가 많은 음료와 간식을 줄이고 "
                "규칙적인 식사와 신체활동을 유지해 보세요. "
                "혈당과 HbA1c 변화를 정기적으로 확인하는 것도 좋습니다. "
                "높은 수치가 반복되면 의료기관에서 확인해 주세요."
            )
        )


    # =====================================================
    # C. 지질 우선순위 계산
    # =====================================================

    lipid_score = 0
    lipid_reasons = []


    if tg >= 150:

        lipid_reasons.append(
            f"중성지방 {tg:.0f} mg/dL"
        )

        lipid_score += (
            (tg - 150) * 0.25
        )


    if tg >= 200:

        lipid_score += 25


    if total_cholesterol >= 200:

        lipid_reasons.append(
            f"총콜레스테롤 {total_cholesterol:.0f} mg/dL"
        )

        lipid_score += (
            (total_cholesterol - 200) * 0.2
        )


    if total_cholesterol >= 240:

        lipid_score += 25


    if sex == 1:

        hdl_limit = 40

    else:

        hdl_limit = 50


    if hdl < hdl_limit:

        lipid_reasons.append(
            f"HDL {hdl:.0f} mg/dL"
        )

        lipid_score += (
            (hdl_limit - hdl) * 2.0
        )


    if len(lipid_reasons) > 0:

        if (
            tg >= 200
            or total_cholesterol >= 240
        ):

            level = 3

        else:

            level = 2


        add_tip(
            lipid_score,
            level,

            "🩸 지질 수치를 관리해 주세요",

            (
                " / ".join(lipid_reasons)
                + "가 확인되었습니다."
            ),

            (
                "포화지방과 과도한 당류 섭취를 점검하고 "
                "규칙적인 유산소 활동을 유지해 보세요. "
                "정기적인 혈액검사로 지질 수치의 변화를 "
                "확인하는 것이 좋습니다."
            )
        )


    # =====================================================
    # D. 신장기능 우선순위 계산
    # =====================================================

    kidney_score = 0


    if egfr < 90:

        kidney_score += (
            (90 - egfr) * 1.2
        )


    if egfr < 60:

        kidney_score += 50


    if egfr < 45:

        kidney_score += 30


    if egfr < 90:

        if egfr < 60:

            level = 3

            title = (
                "🫘 신장기능을 우선 확인해 주세요"
            )

        else:

            level = 1

            title = (
                "🫘 신장기능 변화를 정기적으로 확인해 보세요"
            )


        add_tip(
            kidney_score,
            level,
            title,

            (
                f"계산된 eGFR은 "
                f"{egfr:.1f} mL/min/1.73m²입니다."
            ),

            (
                "eGFR은 연령과 다른 건강상태에 따라 "
                "해석이 달라질 수 있습니다. "
                "혈압과 혈당을 함께 관리하고 "
                "정기검사에서 변화가 지속되는지 확인해 보세요. "
                "한 번의 eGFR 값만으로 만성 신장질환을 "
                "진단할 수는 없습니다."
            )
        )


    # =====================================================
    # E. 체중 / 허리둘레 우선순위 계산
    # =====================================================

    body_score = 0
    body_reasons = []


    if sex == 1:

        waist_limit = 90

    else:

        waist_limit = 85


    if bmi >= 23:

        body_score += (
            (bmi - 23) * 8
        )

        body_reasons.append(
            f"BMI {bmi:.1f}"
        )


    if waist >= waist_limit:

        body_score += (
            (waist - waist_limit) * 4
        )

        body_reasons.append(
            f"허리둘레 {waist:.1f} cm"
        )


    if bmi >= 25:

        body_score += 20


    if bmi >= 30:

        body_score += 25


    if bmi < 18.5:

        body_score = (
            (18.5 - bmi) * 10 + 25
        )

        body_reasons = [
            f"BMI {bmi:.1f}"
        ]


    if len(body_reasons) > 0:

        if bmi >= 25 or waist >= waist_limit:

            level = 2

            title = (
                "⚖️ 체중과 허리둘레를 관리해 주세요"
            )

        elif bmi < 18.5:

            level = 2

            title = (
                "⚖️ 낮은 체중 상태를 확인해 주세요"
            )

        else:

            level = 1

            title = (
                "⚖️ 현재 체중을 꾸준히 관리해 주세요"
            )


        add_tip(
            body_score,
            level,
            title,

            (
                " / ".join(body_reasons)
                + "가 확인되었습니다."
            ),

            (
                "급격한 체중 변화보다는 "
                "꾸준한 신체활동과 균형 잡힌 식사를 통해 "
                "천천히 관리하는 것이 좋습니다."
            )
        )


    # =====================================================
    # F. 실제 개인별 우선순위 정렬
    # =====================================================

    tips = sorted(
        tips,
        key=lambda x: x["sort_score"],
        reverse=True
    )


    # =====================================================
    # 14. 개인 맞춤 결과 출력
    # =====================================================

    if len(tips) == 0:

        st.success(
            "✅ 현재 입력한 주요 건강수치에서는 "
            "뚜렷하게 우선 관리가 필요한 항목이 "
            "확인되지 않았습니다.\n\n"
            "현재 생활습관을 유지하면서 "
            "정기적인 건강검진을 받아주세요."
        )

    else:

        st.markdown(
            "### 🎯 지금 우선 확인하면 좋은 항목"
        )


        for i, tip in enumerate(
            tips[:3],
            start=1
        ):

            message = (
                f"**우선순위 {i}. "
                f"{tip['title']}**\n\n"
                f"📌 {tip['reason']}\n\n"
                f"👉 {tip['action']}"
            )


            if tip["level"] == 3:

                st.error(
                    message
                )

            elif tip["level"] == 2:

                st.warning(
                    message
                )

            else:

                st.info(
                    message
                )


        if len(tips) > 3:

            with st.expander(
                "다른 관리 항목도 보기"
            ):

                for tip in tips[3:]:

                    st.markdown(
                        f"### {tip['title']}"
                    )

                    st.write(
                        f"📌 {tip['reason']}"
                    )

                    st.write(
                        f"👉 {tip['action']}"
                    )

                    st.divider()


    st.caption(
        "※ 관리 우선순위는 입력한 현재 건강수치가 참고 기준에서 "
        "벗어난 정도를 이용해 정렬한 교육용 안내입니다. "
        "의학적 위험도 순위나 진단 결과를 의미하지 않습니다."
    )


    # =====================================================
    # 15. 현재 건강수치 요약
    # =====================================================

    st.markdown(
        "## 🧾 나의 현재 건강수치 요약"
    )

    summary1, summary2, summary3, summary4 = st.columns(4)


    with summary1:

        st.metric(
            "혈압",
            f"{sbp}/{dbp} mmHg"
        )

        st.metric(
            "BMI",
            f"{bmi:.1f}"
        )


    with summary2:

        st.metric(
            "공복혈당",
            f"{glucose:.0f} mg/dL"
        )

        st.metric(
            "HbA1c",
            f"{hba1c:.1f}%"
        )


    with summary3:

        st.metric(
            "총콜레스테롤",
            f"{total_cholesterol:.0f} mg/dL"
        )

        st.metric(
            "중성지방",
            f"{tg:.0f} mg/dL"
        )


    with summary4:

        st.metric(
            "Creatinine",
            f"{creatinine:.1f} mg/dL"
        )

        st.metric(
            "eGFR",
            f"{egfr:.1f}"
        )


    # =====================================================
    # 16. 전체 모델 주요 판단 변수
    # =====================================================

    importance_file = (
        TABLE_DIR
        / "model20_shap_importance.csv"
    )


    if importance_file.exists():

        importance = pd.read_csv(
            importance_file
        )

        feature_names = {

            "T01_AGE": "나이",
            "T00_SEX": "성별",

            "T01_SBP": "수축기혈압",
            "T01_DBP": "이완기혈압",

            "T01_BMI": "BMI",
            "T01_WAIST": "허리둘레",

            "T01_GLU0": "공복혈당",
            "T01_HBA1C": "HbA1c",

            "T01_TCHL": "총콜레스테롤",
            "T01_HDL": "HDL",
            "T01_TG": "중성지방",

            "T01_CREATININE": "Creatinine",
            "T01_EGFR": "eGFR"
        }


        importance["표시이름"] = (
            importance["FEATURE"]
            .map(feature_names)
        )

        top5 = importance.head(5)

        st.markdown(
            "## 🤖 전체 위험 모델의 주요 판단 변수"
        )

        st.caption(
            "아래 그래프는 현재 개인 한 사람의 원인이 아니라, "
            "전체 모델에서 평균적으로 중요했던 변수입니다."
        )

        chart_data = (
            top5[
                [
                    "표시이름",
                    "MEAN_ABS_SHAP"
                ]
            ]
            .set_index(
                "표시이름"
            )
        )

        st.bar_chart(
            chart_data
        )


# =========================================================
# 17. 모델 정보
# =========================================================

st.divider()


with st.expander(
    "📋 모델 정보"
):

    st.markdown(
        "### 전체 만성 건강위험 모델"
    )

    st.write(
        "모델: Random Forest"
    )

    st.write(
        "학습 대상자: 698명"
    )

    st.write(
        "입력 변수: 13개 임상 건강변수"
    )

    st.write(
        "Repeated 5-Fold Cross Validation"
    )

    st.write(
        "ROC-AUC: 0.716 ± 0.058"
    )

    st.write(
        "PR-AUC: 0.303 ± 0.060"
    )


    st.markdown(
        "### 질환별 모델 검증 성능"
    )

    st.write(
        "❤️ 고혈압 ROC-AUC: 0.603"
    )

    st.write(
        "🍬 당뇨 ROC-AUC: 0.704"
    )

    st.write(
        "🩸 이상지질혈증 ROC-AUC: 0.708"
    )

    st.write(
        "🫘 신장기능 저하 ROC-AUC: 0.875"
    )

    st.caption(
        "고혈압·당뇨·이상지질혈증 모델은 "
        "신규 사건 수가 적어 탐색적 참고 모델로 해석해야 합니다."
    )


# =========================================================
# 18. 최종 안내
# =========================================================

st.info(
    "본 서비스는 교육용·합성 데이터 기반 연구 모델의 "
    "시연용 결과입니다. AI 위험 점수와 관리 우선순위는 "
    "실제 질환 발생 확률이나 의학적 진단을 의미하지 않으며, "
    "실제 진단·치료 결정을 위한 의료기기가 아닙니다."
)