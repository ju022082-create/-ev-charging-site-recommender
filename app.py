import io
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="EV 급속충전소 입지 분석", page_icon="⚡", layout="wide")
st.title("⚡ EV 급속충전소 최적 입지 분석")
st.caption(
    "실제 공공데이터 기반 · 전국 탐색 → 선택 지역 상세 분석 · 연구용 의사결정 지원 도구"
)

# Persistent local cache. This folder contains data only, never API credentials.
DATA_DIR = Path(__file__).resolve().parent / "saved_data"
DATA_DIR.mkdir(exist_ok=True)


def saved_upload(upload, stem):
    """Save an uploaded file and return it, or restore the previously saved file."""
    if upload is not None:
        ext = Path(upload.name).suffix.lower()
        if ext not in (".csv", ".xlsx"):
            raise ValueError("CSV/XLSX 파일만 사용할 수 있습니다.")
        for old in DATA_DIR.glob(stem + ".*"):
            if old.is_file():
                old.unlink()
        (DATA_DIR / (stem + ext)).write_bytes(upload.getvalue())
        return upload
    for ext in (".csv", ".xlsx"):
        path = DATA_DIR / (stem + ext)
        if path.exists():
            return SimpleNamespace(name=path.name, getvalue=path.read_bytes)
    return None


def load_saved_chargers():
    csv_path = DATA_DIR / "chargers_api.csv"
    zip_path = DATA_DIR / "chargers_api.zip"

    # 기존 CSV 파일이 있으면 먼저 사용
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype=str, low_memory=False)

    # CSV 파일이 없으면 ZIP 파일에서 불러오기
    if zip_path.exists():
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as z:
            csv_files = [
                name for name in z.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                raise ValueError("ZIP 파일 안에 CSV 파일이 없습니다.")

            with z.open(csv_files[0]) as f:
                return pd.read_csv(f, dtype=str, low_memory=False)

    return None


PROVINCES = [
    "전체",
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
]
ALIASES = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_api(key, max_pages):
    url = "https://apis.data.go.kr/B552584/EvCharger/getChargerInfo"
    out = []
    for page in range(1, max_pages + 1):
        r = requests.get(
            url,
            params={"serviceKey": key, "pageNo": page, "numOfRows": 9999},
            timeout=40,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        code = root.findtext(".//resultCode")
        if code not in (None, "00", "0"):
            raise ValueError(
                "공공데이터 API 오류: " + str(root.findtext(".//resultMsg") or code)
            )
        items = root.findall(".//item")
        if not items:
            break
        for item in items:
            out.append({child.tag: child.text for child in item})
        total = int(root.findtext(".//totalCount") or 0)
        if total and len(out) >= total:
            break
        if len(items) < 9999:
            break
    return pd.DataFrame(out)


def load_table(file):
    content = file.getvalue()
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))
    for encoding in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("CSV 인코딩을 읽지 못했습니다.")


def normalize_chargers(df):
    d = df.copy()
    names = {
        "statNm": "충전소명",
        "addr": "주소",
        "lat": "위도",
        "lng": "경도",
        "powerType": "충전방식",
        "output": "출력",
        "statId": "충전소ID",
        "chgerId": "충전기ID",
        "useTime": "이용시간",
    }
    d = d.rename(columns=names)
    required = ["위도", "경도", "주소"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError("충전소 데이터에 필요한 열이 없습니다: " + ", ".join(missing))
    d["위도"] = pd.to_numeric(d["위도"], errors="coerce")
    d["경도"] = pd.to_numeric(d["경도"], errors="coerce")
    d = d.dropna(subset=["위도", "경도"])
    d = d[d["위도"].between(33, 39.5) & d["경도"].between(124, 132)]
    if "출력" in d:
        d["출력"] = pd.to_numeric(d["출력"], errors="coerce")
        d = d[d["출력"] >= 50]
    else:
        st.warning(
            "출력 열이 없어 급속충전기(50kW 이상)만 필터링할 수 없습니다. 결과는 전체 충전기 기준입니다."
        )
    d["주소"] = d["주소"].fillna("").astype(str)
    d["충전소명"] = d.get("충전소명", pd.Series("충전소", index=d.index)).fillna(
        "충전소"
    )
    d["시도"] = d["주소"].str.split().str[0]
    d["시군구"] = d["주소"].str.split().str[1].fillna("")
    return d


def score_table(chargers, ev, traffic, weights):
    # Aggregate by administrative name; charging supply is per charger, not station.
    supply = chargers.groupby(["시도", "시군구"], as_index=False).agg(
        급속충전기수=("주소", "size"), 위도=("위도", "mean"), 경도=("경도", "mean")
    )
    supply["지역"] = supply["시도"] + " " + supply["시군구"]
    data = supply[["지역", "급속충전기수", "위도", "경도"]]
    for frame in (ev, traffic):
        if frame is not None:
            if not {"지역", frame.columns[-1]}.issubset(frame.columns):
                raise ValueError("보조 데이터에 지역 열이 필요합니다.")
            data = data.merge(frame, on="지역", how="outer")
    if ev is None or traffic is None:
        return data, None
    # Only score rows with complete, valid measured inputs.
    for col in ["전기차등록대수", "일평균교통량", "급속충전기수"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    valid = data.dropna(
        subset=["전기차등록대수", "일평균교통량", "급속충전기수", "위도", "경도"]
    ).copy()
    valid = valid[
        (valid["전기차등록대수"] >= 0)
        & (valid["일평균교통량"] >= 0)
        & (valid["급속충전기수"] >= 0)
    ]
    if valid.empty:
        return data, valid
    valid["충전인프라부족지표"] = valid["전기차등록대수"] / (valid["급속충전기수"] + 1)

    def scale(x):
        a = x.min()
        b = x.max()
        return (
            pd.Series(np.full(len(x), 50.0), index=x.index)
            if a == b
            else 100 * (x - a) / (b - a)
        )

    for source, dest in [
        ("일평균교통량", "교통량점수"),
        ("전기차등록대수", "전기차수요점수"),
        ("충전인프라부족지표", "인프라부족점수"),
    ]:
        valid[dest] = scale(valid[source])
    valid["최종점수"] = sum(valid[c] * weights[c] for c in weights)
    return data, valid.sort_values("최종점수", ascending=False)


st.info(
    "실측 데이터가 없으면 점수를 만들어내지 않습니다. 충전소 API와 전기차 등록·교통량 자료를 입력하면 분석이 활성화됩니다."
)
with st.sidebar:
    st.header("① 데이터 연결")
    source = st.radio("충전소 데이터", ["공공데이터 API", "CSV/XLSX 업로드"])
    raw = None
    if source == "공공데이터 API":
        key = st.text_input(
            "공공데이터포털 일반 인증키",
            value=os.environ.get("EV_API_KEY", ""),
            type="password",
            help="인증키는 자동 저장하지 않습니다. 원한다면 본인 PC의 EV_API_KEY 환경변수로 관리하세요.",
        )
        max_pages = st.number_input(
            "API 최대 페이지 (9,999건/페이지)", min_value=1, max_value=100, value=5
        )
        if st.button("실제 충전소 데이터 불러오기", type="primary"):
            if not key:
                st.error("인증키를 입력해 주세요.")
            else:
                try:
                    fetched = fetch_api(key, int(max_pages))
                    if fetched.empty:
                        st.warning("API에서 데이터가 반환되지 않았습니다. 기존 저장 데이터는 유지합니다.")
                    else:
                        fetched.to_csv(DATA_DIR / "chargers_api.csv", index=False, encoding="utf-8-sig")
                        st.session_state["raw"] = fetched
                        st.success(f"충전기 데이터 {len(fetched):,}행을 PC에 저장했습니다.")
                except Exception as e:
                    st.error(f"API 요청 실패: {e}")
        raw = st.session_state.get("raw")
        if raw is None:
            try:
                raw = load_saved_chargers()
            except Exception as e:
                st.error(f"저장된 충전기 데이터를 읽지 못했습니다: {e}")
        if raw is not None and not raw.empty:
            st.caption("저장된 충전기 데이터 사용 중 · 새로 불러오면 갱신됩니다.")
    else:
        upload = st.file_uploader(
            "충전소 CSV 또는 XLSX", type=["csv", "xlsx"], key="charger"
        )
        try:
            upload = saved_upload(upload, "chargers_upload")
        except Exception as e:
            st.error(f"충전소 파일 저장 오류: {e}")
        if upload:
            try:
                raw = load_table(upload)
            except Exception as e:
                st.error(str(e))
    st.header("② 수요 데이터")
    st.caption(
        "CSV/XLSX에 지역, 전기차등록대수 / 지역, 일평균교통량 열을 넣으세요. 지역은 예: 전라남도 나주시"
    )
    evfile = st.file_uploader("전기차 등록 대수", type=["csv", "xlsx"], key="ev")
    trfile = st.file_uploader("지역별 일평균 교통량", type=["csv", "xlsx"], key="tr")
    try:
        evfile = saved_upload(evfile, "ev_registration")
        trfile = saved_upload(trfile, "traffic")
    except Exception as e:
        st.error(f"보조 데이터 저장 오류: {e}")
    if evfile is not None:
        st.caption("전기차 등록 데이터: PC에 저장됨")
    if trfile is not None:
        st.caption("교통량 데이터: PC에 저장됨")
    st.header("③ 평가 가중치")
    w1 = st.slider("교통량", 0, 100, 35)
    w2 = st.slider("전기차 수요", 0, 100, 35)
    w3 = st.slider("인프라 부족", 0, 100, 30)
    if w1 + w2 + w3 == 0:
        st.error("가중치 중 하나 이상을 양수로 설정하세요.")
    st.caption(
        "가중치는 합계와 관계없이 자동 정규화됩니다. 현재 모델은 3개 데이터 지표만 평가합니다."
    )

st.markdown("### 분석 범위")
region = st.selectbox("시·도 선택", PROVINCES)
if raw is None or raw.empty:
    st.warning(
        "먼저 사이드바에서 실제 충전소 데이터를 연결하세요. API 키가 없다면 공식 포털에서 CSV/XLSX를 받아 업로드할 수 있습니다."
    )
    st.markdown(
        "**공식 데이터 출처:** [한국환경공단 충전소 API](https://www.data.go.kr/data/15076352/openapi.do) · [국토교통부 교통량](https://www.road.re.kr/) · [국토교통 통계누리](https://stat.molit.go.kr/)"
    )
    st.stop()
try:
    chargers = normalize_chargers(raw)
except Exception as e:
    st.error(str(e))
    st.stop()
if region != "전체":
    chargers = chargers[chargers["시도"].isin([region, ALIASES[region]])]
if chargers.empty:
    st.warning("해당 지역의 충전기 데이터가 없습니다.")
    st.stop()


def read_aux(file, value_col):
    if not file:
        return None
    frame = load_table(file)
    if not {"지역", value_col}.issubset(frame.columns):
        raise ValueError(f"필수 열: 지역, {value_col}")
    frame = frame[["지역", value_col]].copy()
    frame["지역"] = frame["지역"].astype(str).str.strip()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    if frame["지역"].duplicated().any():
        raise ValueError(
            "지역 값이 중복됩니다. 지역별로 먼저 합계 또는 적절한 집계를 해주세요."
        )
    return frame


try:
    ev = read_aux(evfile, "전기차등록대수")
    traffic = read_aux(trfile, "일평균교통량")
    denom = max(w1 + w2 + w3, 1)
    overview, ranked = score_table(
        chargers,
        ev,
        traffic,
        {
            "교통량점수": w1 / denom,
            "전기차수요점수": w2 / denom,
            "인프라부족점수": w3 / denom,
        },
    )
except Exception as e:
    st.error(f"분석 데이터 오류: {e}")
    st.stop()
a, b, c = st.columns(3)
a.metric("급속충전기 데이터 행 수", f"{len(chargers):,}")
b.metric("행정구역 수", f"{len(overview):,}")
c.metric("입지 점수 계산 가능 지역", f"{len(ranked) if ranked is not None else 0:,}")
st.caption(
    "충전기 1대당 1행인 데이터 기준입니다. API 페이지 제한이 있으면 전국 전체를 대표하지 않을 수 있습니다."
)
left, right = st.columns([1.4, 1])
with left:
    st.subheader("기존 급속충전 인프라 지도")
    center = [float(chargers["위도"].mean()), float(chargers["경도"].mean())]
    m = folium.Map(
        location=center,
        zoom_start=7 if region == "전체" else 10,
        tiles="OpenStreetMap",
    )
    for _, r in chargers.head(2000).iterrows():
        folium.CircleMarker(
            [r["위도"], r["경도"]],
            radius=3,
            color="#1678c2",
            fill=True,
            fill_opacity=0.5,
            tooltip=str(r["충전소명"]),
        ).add_to(m)
    if ranked is not None:
        for _, r in ranked.head(10).iterrows():
            folium.Marker(
                [r["위도"], r["경도"]],
                tooltip=f"{r['지역']}: {r['최종점수']:.1f}점",
                icon=folium.Icon(color="green", icon="star"),
            ).add_to(m)
    st_folium(m, height=540, width=None, returned_objects=[])
    if len(chargers) > 2000:
        st.caption(
            "지도 성능을 위해 처음 2,000개 충전기만 표시합니다. 통계에는 전체 불러온 데이터가 반영됩니다."
        )
with right:
    st.subheader("지역별 충전기 현황")
    st.dataframe(
        overview[["지역", "급속충전기수"]].sort_values("급속충전기수", ascending=False),
        width="stretch",
        hide_index=True,
        height=440,
    )
    st.download_button(
        "지역별 현황 CSV 다운로드",
        overview.to_csv(index=False).encode("utf-8-sig"),
        "charger_overview.csv",
        "text/csv",
    )
if ranked is None:
    st.warning(
        "전기차 등록 대수와 교통량 파일을 모두 올리면 실제 데이터 기반 추천 순위가 계산됩니다."
    )
elif ranked.empty:
    st.warning(
        "행정구역 이름이 일치하는 데이터가 없습니다. 예: 전라남도 나주시 형태로 통일하세요."
    )
else:
    st.subheader("📍 신규 급속충전소 입지 우선 검토 지역")
    st.dataframe(
        ranked[
            [
                "지역",
                "최종점수",
                "일평균교통량",
                "전기차등록대수",
                "급속충전기수",
                "충전인프라부족지표",
            ]
        ].round(2),
        hide_index=True,
        width="stretch",
    )
    st.bar_chart(ranked.head(15).set_index("지역")["최종점수"])
    st.download_button(
        "입지 추천 결과 다운로드",
        ranked.to_csv(index=False).encode("utf-8-sig"),
        "site_ranking.csv",
        "text/csv",
    )
st.divider()
st.caption(
    "주의: 행정구역 단위 1차 스크리닝 결과입니다. 교통량은 전기차 통행량과 다르며, 급속충전기 수는 실제 가용 용량·이용률을 대변하지 않습니다. 지도 표식은 해당 행정구역의 기존 충전기 좌표 평균이며 실제 설치 부지가 아닙니다. 전력계통 여유, 부지 소유권, 진출입 안전성 등은 2차 현장 검토가 필요합니다."
)
