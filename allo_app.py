import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="[글로벌자산배분전략위원회] Quantitative Sleeve Allocation",
    layout="wide"
)

# 폰트 크기 조정 CSS
st.markdown("""
    <style>
        /* 전체 텍스트 크기 조정 */
        html, body, [class*="css"] {
            font-size: 12px !important;
        }

        /* 헤더 크기 조정 */
        h1 {
            font-size: 1.4rem !important;
        }

        h2 {
            font-size: 1.2rem !important;
        }

        h3 {
            font-size: 1.1rem !important;
        }

        /* 본문 텍스트 */
        p, div, span {
            font-size: 12px !important;
        }

        /* 메트릭 컴포넌트 */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }

        /* 데이터프레임 */
        .dataframe {
            font-size: 11px !important;
        }

        /* 사이드바 */
        [data-testid="stSidebar"] {
            font-size: 12px !important;
        }

        /* 라디오 버튼, 체크박스 등 */
        label {
            font-size: 12px !important;
        }

        /* 입력 필드 */
        input, select, textarea {
            font-size: 12px !important;
        }

        /* 버튼 */
        button {
            font-size: 12px !important;
        }

        /* 정보/경고 메시지 */
        [data-baseweb="notification"] {
            font-size: 11px !important;
        }

        /* 테이블 스타일 개선 */
        .dataframe {
            width: 100% !important;
            border-collapse: collapse !important;
            table-layout: fixed !important;
        }

        .dataframe th {
            background: linear-gradient(180deg, #1f2937 0%, #111827 100%) !important;
            color: #fafafa !important;
            font-weight: 600 !important;
            padding: 10px 8px !important;
            text-align: center !important;
            border: 1px solid #374151 !important;
            font-size: 11px !important;
        }

        .dataframe td {
            padding: 10px 8px !important;
            text-align: center !important;
            border: 1px solid #374151 !important;
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
            font-size: 11px !important;
        }

        .dataframe tbody tr:nth-child(even) {
            background-color: #1a1f2e !important;
        }

        .dataframe tbody tr:nth-child(even) td {
            background-color: #1a1f2e !important;
        }

        .dataframe tbody tr:hover {
            background-color: #374151 !important;
        }

        .dataframe tbody tr:hover td {
            background-color: #374151 !important;
        }

        /* 칼럼 너비 동일하게 설정 - 모든 칼럼 동일한 너비 */
        .dataframe {
            table-layout: fixed !important;
        }

        .dataframe th,
        .dataframe td {
            width: 12.5% !important;
            word-wrap: break-word !important;
        }

        /* 첫 번째 칼럼(SLEEVE)과 마지막 칼럼(GROUP)도 동일한 너비 */
        .dataframe th:first-child,
        .dataframe td:first-child,
        .dataframe th:last-child,
        .dataframe td:last-child {
            width: 12.5% !important;
            text-align: left !important;
            font-weight: 500 !important;
        }

        /* 숫자 칼럼 우측 정렬 및 폰트 */
        .dataframe td:nth-child(n+2):not(:last-child) {
            text-align: right !important;
            font-family: 'Courier New', monospace !important;
            font-weight: 500 !important;
        }

        .dataframe th:nth-child(n+2):not(:last-child) {
            text-align: right !important;
        }

        /* 3열 설정 부분 세로 구분선 */
        [data-testid="column"]:not(:last-child) {
            border-right: 2px solid #000000 !important;
            padding-right: 20px !important;
            margin-right: 0 !important;
        }

        [data-testid="column"]:not(:first-child) {
            padding-left: 20px !important;
            margin-left: 0 !important;
        }

        /* 컬럼 컨테이너에 구분선 추가 */
        div[data-testid="column"]:not(:last-child) {
            position: relative;
        }

        div[data-testid="column"]:not(:last-child)::after {
            content: "";
            position: absolute;
            right: -1px;
            top: 0;
            bottom: 0;
            width: 2px;
            background-color: #000000;
            z-index: 1;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="font-size: 6.0rem;">[글로벌자산배분전략위원회] Quantitative Sleeve Allocation</h1>', unsafe_allow_html=True)

# ReadMe 섹션
st.markdown("""
    <div style="font-size: 12px;">
        <strong>ReadMe</strong><br>
        1. 정해진 양식의 엑셀파일을 업로드한 후 최적화를 수행합니다. <br> 
        2. 사이드바에서 펀드(530810 or 530950)를 선택할 수 있습니다. <br> 
        3. 최적화의 기대수익률은 과거수익률 활용, 위원회 스코어링 결과 적용, 몬테칼로 시뮬레이션 방법을 선택할 수 있습니다. <br> 
        4. 최적화는 Max Sharpe, Min Risk, Risk Parity 세 가지를 적용합니다. <br> 
        5. 최적화 결과는 테이블로 조회할 수 있고, CSV 파일로 다운로드할 수 있습니다. <br>
        <br>
    </div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'fund_selected' not in st.session_state:
    st.session_state.fund_selected = None


def load_excel_data(uploaded_file):
    """엑셀 파일에서 데이터 로드"""
    try:
        # 각 시트 읽기
        price_df = pd.read_excel(uploaded_file, sheet_name='기준가')
        Current_df = pd.read_excel(uploaded_file, sheet_name='Current')
        Gr_MinMax_df = pd.read_excel(uploaded_file, sheet_name='Gr_MinMax')

        return price_df, Current_df, Gr_MinMax_df
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {str(e)}")
        return None, None, None


def filter_data_by_fund(price_df, Current_df, Gr_MinMax_df, fund_type):
    """펀드 타입에 따라 데이터 필터링"""
    if fund_type == '530810':
        # 530950 관련 칼럼 제거
        Current_filtered = Current_df.drop(columns=[col for col in Current_df.columns if '530950' in str(col)],
                                           errors='ignore')
        Gr_MinMax_filtered = Gr_MinMax_df.drop(columns=[col for col in Gr_MinMax_df.columns if '530950' in str(col)],
                                               errors='ignore')
        weight_col = 'F530810'
        min_col = 'MIN_530810'
        max_col = 'MAX_530810'
    else:  # 530950
        # 530810 관련 칼럼 제거
        Current_filtered = Current_df.drop(columns=[col for col in Current_df.columns if '530810' in str(col)],
                                           errors='ignore')
        Gr_MinMax_filtered = Gr_MinMax_df.drop(columns=[col for col in Gr_MinMax_df.columns if '530810' in str(col)],
                                               errors='ignore')
        weight_col = 'F530950'
        min_col = 'MIN_530950'
        max_col = 'MAX_530950'

    return price_df, Current_filtered, Gr_MinMax_filtered, weight_col, min_col, max_col


def calculate_historical_returns(price_df, Current_filtered, return_period, calc_period=3):
    """과거수익률 계산 (return_period 개월의 과거 데이터를 사용하여 3개월 기간 수익률로 변환)"""
    # DATE 칼럼을 날짜로 변환
    if 'DATE' in price_df.columns:
        price_df['DATE'] = pd.to_datetime(price_df['DATE'])
        price_df = price_df.sort_values('DATE')
    else:
        # DATE 칼럼이 없으면 인덱스 기준으로 정렬
        price_df = price_df.sort_index()

    # CODE별로 수익률 계산
    returns_dict = {}
    sleeves = Current_filtered['SLEEVE'].unique()

    # 항상 3개월 기간 수익률로 계산
    # 캘린더 기준으로 정확히 calc_period 개월 전 날짜를 찾아서 계산

    # DATE를 제외한 칼럼들이 CODE 값들
    code_columns = [col for col in price_df.columns if col != 'DATE']

    for sleeve in sleeves:
        sleeve_codes = Current_filtered[Current_filtered['SLEEVE'] == sleeve]['CODE'].astype(str).tolist()

        # 해당 sleeve의 CODE와 매칭되는 칼럼들
        matched_cols = [col for col in code_columns if str(col) in sleeve_codes]

        if len(matched_cols) == 0:
            returns_dict[sleeve] = 0.0
            continue

        # 각 CODE(칼럼)별 수익률 계산 후 평균
        sleeve_returns = []
        for code_col in matched_cols:
            if code_col not in price_df.columns:
                continue

            # 해당 칼럼의 시계열 데이터 (DATE와 함께)
            if 'DATE' in price_df.columns:
                code_data = price_df[['DATE', code_col]].dropna(subset=[code_col]).copy()
            else:
                code_data = pd.DataFrame({code_col: price_df[code_col].dropna()})
                code_data['DATE'] = code_data.index

            if len(code_data) < 2:
                continue

            # return_period 개월 전부터 시작 (캘린더 기준)
            max_date = code_data['DATE'].max()
            cutoff_date = max_date - pd.DateOffset(months=return_period)
            code_data_filtered = code_data[code_data['DATE'] >= cutoff_date].copy()

            if len(code_data_filtered) < 2:
                continue

            # 3개월 기간 수익률들을 계산
            # 각 날짜에 대해 정확히 3개월 전 날짜를 찾아서 수익률 계산
            period_returns = []
            for i in range(len(code_data_filtered)):
                current_date = code_data_filtered.iloc[i]['DATE']
                target_date = current_date - pd.DateOffset(months=calc_period)

                # target_date와 가장 가까운 날짜 찾기 (target_date 이전 또는 같은 날짜)
                valid_dates = code_data[code_data['DATE'] <= current_date]
                if len(valid_dates) == 0:
                    continue

                date_diff = (valid_dates['DATE'] - target_date).abs()
                if len(date_diff) == 0:
                    continue

                closest_idx = date_diff.idxmin()

                if closest_idx is not None:
                    current_price = code_data_filtered.iloc[i][code_col]
                    prev_price = code_data.loc[closest_idx, code_col]

                    # 가격이 0이 아닌 경우만 계산
                    if current_price > 0 and prev_price > 0:
                        # 3개월 수익률 계산 (퍼센트)
                        period_return = (current_price / prev_price - 1) * 100
                        period_returns.append(period_return)

            if period_returns:
                sleeve_returns.append(np.mean(period_returns))

        returns_dict[sleeve] = np.mean(sleeve_returns) if sleeve_returns else 0.0

    return returns_dict


def calculate_monte_carlo_returns(price_df, Current_filtered, return_period, corr_matrix, sleeves_list, calc_period=3,
                                  n_simulations=1000):
    """몬테칼로 시뮬레이션을 통한 기대수익률 계산 (상관관계 고려, 3개월 기간 수익률로 산출)"""
    # 랜덤 시드 설정 (재현 가능성)
    np.random.seed(42)

    # DATE 칼럼을 날짜로 변환
    if 'DATE' in price_df.columns:
        price_df['DATE'] = pd.to_datetime(price_df['DATE'])
        price_df = price_df.sort_values('DATE')
    else:
        # DATE 칼럼이 없으면 인덱스 기준으로 정렬
        price_df = price_df.sort_index()

    # 항상 3개월 기간 수익률로 계산
    # 캘린더 기준으로 정확히 calc_period 개월 전 날짜를 찾아서 계산

    # DATE를 제외한 칼럼들이 CODE 값들
    code_columns = [col for col in price_df.columns if col != 'DATE']

    # 각 sleeve별 수익률 시계열 계산
    sleeve_returns_series = {}

    for sleeve in sleeves_list:
        sleeve_codes = Current_filtered[Current_filtered['SLEEVE'] == sleeve]['CODE'].astype(str).tolist()

        # 해당 sleeve의 CODE와 매칭되는 칼럼들
        matched_cols = [col for col in code_columns if str(col) in sleeve_codes]

        if len(matched_cols) == 0:
            sleeve_returns_series[sleeve] = []
            continue

        # 각 CODE(칼럼)별 3개월 수익률 계산
        all_returns = []
        for code_col in matched_cols:
            if code_col not in price_df.columns:
                continue

            # 해당 칼럼의 시계열 데이터 (DATE와 함께)
            if 'DATE' in price_df.columns:
                code_data = price_df[['DATE', code_col]].dropna(subset=[code_col]).copy()
            else:
                code_data = pd.DataFrame({code_col: price_df[code_col].dropna()})
                code_data['DATE'] = code_data.index

            if len(code_data) < 2:
                continue

            # return_period 개월 전부터 시작 (캘린더 기준)
            max_date = code_data['DATE'].max()
            cutoff_date = max_date - pd.DateOffset(months=return_period)
            code_data_filtered = code_data[code_data['DATE'] >= cutoff_date].copy()

            if len(code_data_filtered) < 2:
                continue

            # 3개월 기간 수익률들을 계산
            # 각 날짜에 대해 정확히 3개월 전 날짜를 찾아서 수익률 계산
            for i in range(len(code_data_filtered)):
                current_date = code_data_filtered.iloc[i]['DATE']
                target_date = current_date - pd.DateOffset(months=calc_period)

                # target_date와 가장 가까운 날짜 찾기
                date_diff = (code_data['DATE'] - target_date).abs()
                closest_idx = date_diff.idxmin()

                if closest_idx is not None and code_data.loc[closest_idx, 'DATE'] <= current_date:
                    current_price = code_data_filtered.iloc[i][code_col]
                    prev_price = code_data.loc[closest_idx, code_col]

                    if current_price > 0 and prev_price > 0:
                        # 3개월 수익률 계산 (퍼센트)
                        period_return = (current_price / prev_price - 1) * 100
                        all_returns.append(period_return)

        sleeve_returns_series[sleeve] = all_returns

    # 공통 기간 찾기 (최소 길이로 맞춤)
    min_len = min([len(returns) for returns in sleeve_returns_series.values() if len(returns) > 0], default=0)
    if min_len == 0:
        # 공통 기간이 없으면 각 sleeve별 평균 사용
        returns_dict = {sleeve: np.mean(returns) if returns else 0.0
                        for sleeve, returns in sleeve_returns_series.items()}
        return returns_dict

    # 공통 기간의 수익률 행렬 구성
    n_sleeves = len(sleeves_list)
    returns_matrix = np.zeros((min_len, n_sleeves))

    for i, sleeve in enumerate(sleeves_list):
        returns = sleeve_returns_series.get(sleeve, [])
        if len(returns) >= min_len:
            returns_matrix[:, i] = returns[:min_len]
        else:
            # 데이터가 부족하면 평균으로 채움
            returns_matrix[:, i] = np.mean(returns) if returns else 0.0

    # 평균 수익률 벡터 (3개월 기간 수익률)
    mean_returns = np.mean(returns_matrix, axis=0)

    # 공분산 행렬 계산 (3개월 기간 변동성)
    cov_matrix = np.cov(returns_matrix.T)

    # 상관관계 행렬과 일치하도록 조정
    std_returns = np.std(returns_matrix, axis=0)
    for i in range(n_sleeves):
        for j in range(n_sleeves):
            if std_returns[i] > 0 and std_returns[j] > 0:
                cov_matrix[i, j] = corr_matrix[i, j] * std_returns[i] * std_returns[j]

    # 몬테칼로 시뮬레이션: 다변량 정규분포
    try:
        simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)
        returns_dict = {sleeves_list[i]: np.mean(simulated_returns[:, i]) for i in range(n_sleeves)}
    except:
        # 공분산 행렬이 양정부호가 아닌 경우, 각 sleeve별 독립적으로 시뮬레이션
        returns_dict = {}
        for i, sleeve in enumerate(sleeves_list):
            returns = sleeve_returns_series.get(sleeve, [])
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                returns_dict[sleeve] = np.mean(simulated_returns)
            else:
                returns_dict[sleeve] = 0.0

    return returns_dict


def calculate_volatility(price_df, Current_filtered, vol_period, calc_period=3):
    """변동성 계산 (vol_period 개월의 과거 데이터를 사용하여 3개월 기간 변동성으로 변환)"""
    # DATE 칼럼을 날짜로 변환
    if 'DATE' in price_df.columns:
        price_df['DATE'] = pd.to_datetime(price_df['DATE'])
        price_df = price_df.sort_values('DATE')
    else:
        # DATE 칼럼이 없으면 인덱스 기준으로 정렬
        price_df = price_df.sort_index()

    volatility_dict = {}
    sleeves = Current_filtered['SLEEVE'].unique()

    # 항상 3개월 기간 변동성으로 계산
    # 캘린더 기준으로 정확히 calc_period 개월 전 날짜를 찾아서 계산

    # DATE를 제외한 칼럼들이 CODE 값들
    code_columns = [col for col in price_df.columns if col != 'DATE']

    for sleeve in sleeves:
        sleeve_codes = Current_filtered[Current_filtered['SLEEVE'] == sleeve]['CODE'].astype(str).tolist()

        # 해당 sleeve의 CODE와 매칭되는 칼럼들
        matched_cols = [col for col in code_columns if str(col) in sleeve_codes]

        if len(matched_cols) == 0:
            volatility_dict[sleeve] = 0.0
            continue

        # 각 CODE(칼럼)별 변동성 계산 후 평균
        sleeve_vols = []
        for code_col in matched_cols:
            if code_col not in price_df.columns:
                continue

            # 해당 칼럼의 시계열 데이터 (DATE와 함께)
            if 'DATE' in price_df.columns:
                code_data = price_df[['DATE', code_col]].dropna(subset=[code_col]).copy()
            else:
                code_data = pd.DataFrame({code_col: price_df[code_col].dropna()})
                code_data['DATE'] = code_data.index

            if len(code_data) < 2:
                continue

            # vol_period 개월 전부터 시작 (캘린더 기준)
            max_date = code_data['DATE'].max()
            cutoff_date = max_date - pd.DateOffset(months=vol_period)
            code_data_filtered = code_data[code_data['DATE'] >= cutoff_date].copy()

            if len(code_data_filtered) < 2:
                continue

            # 3개월 기간 수익률들을 계산
            # 각 날짜에 대해 정확히 3개월 전 날짜를 찾아서 수익률 계산
            period_returns = []
            for i in range(len(code_data_filtered)):
                current_date = code_data_filtered.iloc[i]['DATE']
                target_date = current_date - pd.DateOffset(months=calc_period)

                # target_date와 가장 가까운 날짜 찾기
                date_diff = (code_data['DATE'] - target_date).abs()
                closest_idx = date_diff.idxmin()

                if closest_idx is not None and code_data.loc[closest_idx, 'DATE'] <= current_date:
                    current_price = code_data_filtered.iloc[i][code_col]
                    prev_price = code_data.loc[closest_idx, code_col]

                    if current_price > 0 and prev_price > 0:
                        # 3개월 수익률 계산 (퍼센트)
                        period_return = (current_price / prev_price - 1) * 100
                        period_returns.append(period_return)

            if len(period_returns) > 1:
                # 3개월 기간 수익률의 변동성 (퍼센트, 연율화하지 않음)
                vol = np.std(period_returns)
                sleeve_vols.append(vol)

        volatility_dict[sleeve] = np.mean(sleeve_vols) if sleeve_vols else 0.0

    return volatility_dict


def calculate_correlation_matrix(price_df, Current_filtered, sleeves):
    """Sleeve 간 상관관계 행렬 계산"""
    if 'DATE' in price_df.columns:
        price_df['DATE'] = pd.to_datetime(price_df['DATE'])
        price_df = price_df.sort_values('DATE')
    else:
        # DATE 칼럼이 없으면 인덱스 기준으로 정렬
        price_df = price_df.sort_index()

    # DATE를 제외한 칼럼들이 CODE 값들
    code_columns = [col for col in price_df.columns if col != 'DATE']

    # 각 sleeve별 수익률 시계열 계산
    sleeve_returns = {}

    for sleeve in sleeves:
        sleeve_codes = Current_filtered[Current_filtered['SLEEVE'] == sleeve]['CODE'].astype(str).tolist()

        # 해당 sleeve의 CODE와 매칭되는 칼럼들
        matched_cols = [col for col in code_columns if str(col) in sleeve_codes]

        daily_returns = []
        for code_col in matched_cols:
            if code_col not in price_df.columns:
                continue

            # 해당 칼럼의 시계열 데이터 (DATE와 함께)
            if 'DATE' in price_df.columns:
                code_data = price_df[['DATE', code_col]].dropna(subset=[code_col]).copy()
            else:
                code_data = pd.DataFrame({code_col: price_df[code_col].dropna()})
                code_data['DATE'] = code_data.index

            if len(code_data) < 2:
                continue

            # 일별 수익률 계산
            for i in range(1, len(code_data)):
                latest_price = code_data.iloc[i][code_col]
                prev_price = code_data.iloc[i - 1][code_col]

                if latest_price > 0 and prev_price > 0:
                    period_return = (latest_price / prev_price - 1) * 100
                    daily_returns.append(period_return)

        if daily_returns:
            # 최근 1년치 데이터 사용 (약 252 영업일, 최대 365일)
            sleeve_returns[sleeve] = daily_returns[-365:] if len(daily_returns) > 365 else daily_returns

    # 상관관계 행렬 계산
    n_sleeves = len(sleeves)
    corr_matrix = np.eye(n_sleeves)

    for i, sleeve1 in enumerate(sleeves):
        for j, sleeve2 in enumerate(sleeves):
            if i != j and sleeve1 in sleeve_returns and sleeve2 in sleeve_returns:
                returns1 = sleeve_returns[sleeve1]
                returns2 = sleeve_returns[sleeve2]
                min_len = min(len(returns1), len(returns2))
                if min_len > 1:
                    corr = np.corrcoef(returns1[:min_len], returns2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        corr_matrix[i, j] = corr

    return corr_matrix


def optimize_portfolio(Current_filtered, Gr_MinMax_filtered, expected_returns, volatilities,
                       corr_matrix, weight_col, min_col, max_col, objective, risk_free_rate,
                       dur_buffer, portfolio_duration, return_period):
    """포트폴리오 최적화"""
    sleeves = Current_filtered['SLEEVE'].unique().tolist()
    n = len(sleeves)

    # 현재 비중
    current_weights = Current_filtered.set_index('SLEEVE')[weight_col].to_dict()
    current_weights_array = np.array([current_weights.get(s, 0) for s in sleeves])
    current_weights_normalized = current_weights_array / current_weights_array.sum()

    # 기대수익률 벡터
    mu = np.array([expected_returns.get(s, 0) for s in sleeves]) / 100  # 퍼센트를 소수로 변환

    # 변동성 벡터
    sigma = np.array([volatilities.get(s, 0) for s in sleeves]) / 100  # 퍼센트를 소수로 변환

    # 변동성이 모두 0인 경우 처리
    if np.all(sigma == 0):
        st.warning("⚠️ 모든 Sleeve의 변동성이 0입니다. 최소 변동성을 0.01%로 설정합니다.")
        sigma = np.where(sigma == 0, 0.0001, sigma)  # 0.01% = 0.0001

    # 공분산 행렬
    cov_matrix = np.outer(sigma, sigma) * corr_matrix

    # 공분산 행렬이 모두 0인 경우 처리
    if np.all(cov_matrix == 0):
        st.warning("⚠️ 공분산 행렬이 0입니다. 대각 행렬로 대체합니다.")
        cov_matrix = np.diag(sigma ** 2)

    # 제약조건 설정
    # 1. 개별 SLEEVE별 비중 제약
    # 엑셀의 % 형식 데이터는 이미 소수로 읽히므로 / 100 불필요
    min_weights = Current_filtered.set_index('SLEEVE')[min_col].to_dict()
    max_weights = Current_filtered.set_index('SLEEVE')[max_col].to_dict()
    bounds = [(min_weights.get(s, 0), max_weights.get(s, 1.0)) for s in sleeves]

    # 2. DUR 제약
    dur_values = Current_filtered.set_index('SLEEVE')['DUR'].to_dict()
    dur_array = np.array([dur_values.get(s, 0) for s in sleeves])

    dur_min = portfolio_duration * (1 - dur_buffer / 100)
    dur_max = portfolio_duration * (1 + dur_buffer / 100)

    # 3. 그룹별 비중 제약
    # 엑셀의 % 형식 데이터는 이미 소수로 읽히므로 / 100 불필요
    group_constraints = {}
    for _, row in Gr_MinMax_filtered.iterrows():
        group = row['GROUP']
        group_min = row.get('MIN_' + weight_col.replace('F', ''), 0)
        group_max = row.get('MAX_' + weight_col.replace('F', ''), 1.0)
        group_constraints[group] = (group_min, group_max)

    # 그룹 매핑
    group_mapping = Current_filtered.set_index('SLEEVE')['GROUP'].to_dict()

    # 목적함수 정의
    def objective_function(w):
        w = np.array(w)
        portfolio_return = np.dot(w, mu)
        portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))

        if objective == "Max Sharpe":
            sharpe = (portfolio_return - risk_free_rate / 100) / portfolio_vol if portfolio_vol > 0 else -1e10
            return -sharpe  # 최소화를 위해 음수
        elif objective == "Min Risk":
            return portfolio_vol
        else:  # Risk Parity
            # Risk Parity: 각 자산의 기여도가 동일하도록
            risk_contributions = w * (np.dot(cov_matrix, w) / portfolio_vol) if portfolio_vol > 0 else w
            target_risk = portfolio_vol / n
            return np.sum((risk_contributions - target_risk) ** 2)

    # 제약조건 함수
    constraints = []

    # 합계 = 1 (100%로 보정)
    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # DUR 제약 (이미 합계=1이므로 w를 그대로 사용)
    constraints.append({'type': 'ineq', 'fun': lambda w: np.dot(w, dur_array) - dur_min})
    constraints.append({'type': 'ineq', 'fun': lambda w: dur_max - np.dot(w, dur_array)})

    # 그룹별 비중 제약 (이미 합계=1이므로 w를 그대로 사용)
    for group, (group_min, group_max) in group_constraints.items():
        group_sleeves = [s for s in sleeves if group_mapping.get(s) == group]
        if group_sleeves:
            group_indices = [sleeves.index(s) for s in group_sleeves]

            # 클로저 문제 해결을 위해 함수 생성
            def make_group_min_constraint(idx, g_min):
                return lambda w: np.sum([w[i] for i in idx]) - g_min

            def make_group_max_constraint(idx, g_max):
                return lambda w: g_max - np.sum([w[i] for i in idx])

            constraints.append({'type': 'ineq', 'fun': make_group_min_constraint(group_indices, group_min)})
            constraints.append({'type': 'ineq', 'fun': make_group_max_constraint(group_indices, group_max)})

    # 회전율 제약 (100%) - 이미 합계=1이므로 w를 그대로 사용
    turnover_limit = 1.0
    constraints.append(
        {'type': 'ineq', 'fun': lambda w: turnover_limit - np.sum(np.abs(w - current_weights_normalized))})

    # 초기값이 제약조건을 만족하는지 확인하고 조정
    x0 = current_weights_normalized.copy()

    # 초기값이 bounds를 만족하는지 확인
    for i in range(n):
        if x0[i] < bounds[i][0]:
            x0[i] = bounds[i][0]
        elif x0[i] > bounds[i][1]:
            x0[i] = bounds[i][1]

    # 합계를 1로 정규화
    x0 = x0 / np.sum(x0) if np.sum(x0) > 0 else x0

    # 최적화 실행 (여러 방법 시도)
    optimal_weights_raw = None
    methods = ['SLSQP', 'trust-constr']

    for method in methods:
        try:
            if method == 'SLSQP':
                result = minimize(
                    objective_function,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-6, 'disp': False}
                )
            else:  # trust-constr
                result = minimize(
                    objective_function,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 2000, 'gtol': 1e-6, 'disp': False}
                )

            if result.success:
                # 결과가 제약조건을 만족하는지 확인
                w_test = result.x
                w_test = w_test / np.sum(w_test) if np.sum(w_test) > 0 else w_test

                # bounds 확인
                tol = 1e-8
                bounds_ok = all(bounds[i][0] - tol <= w_test[i] <= bounds[i][1] + tol for i in range(n))

                # DUR 제약 확인
                dur_test = np.dot(w_test, dur_array)
                dur_ok = (dur_min - tol) <= dur_test <= (dur_max + tol)

                if bounds_ok and dur_ok:
                    optimal_weights_raw = w_test
                    break
        except Exception as e:
            continue

    # 모든 방법이 실패한 경우, 제약조건을 완화하여 재시도
    if optimal_weights_raw is None:
        try:
            # 제약조건을 완화 (DUR 버퍼를 10% 더 늘림)
            dur_min_relaxed = portfolio_duration * (1 - (dur_buffer + 10) / 100)
            dur_max_relaxed = portfolio_duration * (1 + (dur_buffer + 10) / 100)

            constraints_relaxed = []
            constraints_relaxed.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            constraints_relaxed.append({'type': 'ineq', 'fun': lambda w: np.dot(w, dur_array) - dur_min_relaxed})
            constraints_relaxed.append({'type': 'ineq', 'fun': lambda w: dur_max_relaxed - np.dot(w, dur_array)})

            # 그룹 제약은 유지
            for group, (group_min, group_max) in group_constraints.items():
                group_sleeves = [s for s in sleeves if group_mapping.get(s) == group]
                if group_sleeves:
                    group_indices = [sleeves.index(s) for s in group_sleeves]

                    def make_group_min_constraint(idx, g_min):
                        return lambda w: np.sum([w[i] for i in idx]) - g_min

                    def make_group_max_constraint(idx, g_max):
                        return lambda w: g_max - np.sum([w[i] for i in idx])

                    constraints_relaxed.append(
                        {'type': 'ineq', 'fun': make_group_min_constraint(group_indices, group_min)})
                    constraints_relaxed.append(
                        {'type': 'ineq', 'fun': make_group_max_constraint(group_indices, group_max)})

            constraints_relaxed.append(
                {'type': 'ineq', 'fun': lambda w: turnover_limit - np.sum(np.abs(w - current_weights_normalized))})

            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_relaxed,
                options={'maxiter': 2000, 'ftol': 1e-6, 'disp': False}
            )

            if result.success:
                optimal_weights_raw = result.x / np.sum(result.x) if np.sum(result.x) > 0 else result.x
                st.warning("⚠️ DUR 제약을 완화하여 최적화를 수행했습니다.")
            else:
                st.warning("⚠️ 최적화가 수렴하지 않았습니다. 현재 비중을 사용합니다.")
                optimal_weights_raw = current_weights_normalized
        except Exception as e:
            st.warning(f"⚠️ 최적화 중 오류 발생: {str(e)}. 현재 비중을 사용합니다.")
            optimal_weights_raw = current_weights_normalized

    if optimal_weights_raw is None:
        optimal_weights_raw = current_weights_normalized

    # 최적화는 100%로 환산한 비중(정규화된 비중)으로 수행됨
    optimal_weights_normalized = optimal_weights_raw / np.sum(optimal_weights_raw) if np.sum(
        optimal_weights_raw) > 0 else optimal_weights_raw

    # 원본 Current 시트의 비중 합계 계산 (100%로 환산하기 전)
    current_weights_total = current_weights_array.sum()

    # 최적화된 정규화 비중을 원본 비중 합계에 맞춰서 변환 (원본 기준 비중)
    optimal_weights_original_scale = optimal_weights_normalized * current_weights_total

    # 결과 계산 (정규화된 비중 기준으로 포트폴리오 지표 계산)
    portfolio_return = np.dot(optimal_weights_normalized, mu) * 100
    portfolio_vol = np.sqrt(np.dot(optimal_weights_normalized, np.dot(cov_matrix, optimal_weights_normalized))) * 100
    optimal_duration = np.dot(optimal_weights_normalized, dur_array)

    # 원본 기준 비중 변화 계산
    weight_changes = {sleeves[i]: (optimal_weights_original_scale[i] - current_weights_array[i])
                      for i in range(n)}

    # 반환값: 정규화된 비중(최적화에 사용), 원본 기준 비중(결과 표시용)
    optimal_weights_normalized_dict = {sleeves[i]: optimal_weights_normalized[i] for i in range(n)}
    optimal_weights_original_dict = {sleeves[i]: optimal_weights_original_scale[i] for i in range(n)}

    return optimal_weights_original_dict, optimal_weights_normalized_dict, portfolio_return, portfolio_vol, optimal_duration, weight_changes


# 사이드바: 파일 업로드
with st.sidebar:
    # 이미지 표시
    image_path = "images/miraeasset.png"
    try:
        st.image(image_path, use_container_width=True)
    except:
        st.warning("이미지를 불러올 수 없습니다.")

    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=['xlsx', 'xls'])

    # 예제 파일 다운로드 링크
    example_file_path = "images/example.xlsx"
    if os.path.exists(example_file_path):
        with open(example_file_path, "rb") as f:
            example_file_data = f.read()
            # 텍스트 링크처럼 보이도록 다운로드 버튼 생성
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            st.download_button(
                label="📥 예제 파일 다운로드",
                data=example_file_data,
                file_name="example.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="example_file_download"
            )
    else:
        st.caption("예제 파일을 찾을 수 없습니다.")

    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.data_loaded = False

        if not st.session_state.data_loaded:
            with st.spinner("데이터 로딩 중..."):
                price_df, Current_df, Gr_MinMax_df = load_excel_data(uploaded_file)
                if price_df is not None:
                    st.session_state.price_df = price_df
                    st.session_state.Current_df = Current_df
                    st.session_state.Gr_MinMax_df = Gr_MinMax_df
                    st.session_state.data_loaded = True

                    # 최근자료일 계산
                    if 'DATE' in price_df.columns:
                        price_df_date = price_df.copy()
                        price_df_date['DATE'] = pd.to_datetime(price_df_date['DATE'], errors='coerce')
                        latest_date = price_df_date['DATE'].max()
                        if pd.isna(latest_date):
                            latest_date = pd.Timestamp.today()
                        st.session_state.latest_date = latest_date
                    else:
                        # DATE 칼럼이 없으면 오늘 날짜 사용
                        latest_date = pd.Timestamp.today()
                        st.session_state.latest_date = latest_date

                    st.success("데이터 로드 완료!")
                    st.info(f"as of: {st.session_state.latest_date.strftime('%Y-%m-%d')}")

        if st.session_state.data_loaded:
            st.header("⚙️ 설정")
            fund_type = st.radio(
                "대상 펀드 선택",
                ['530810', '530950'],
                index=0 if st.session_state.fund_selected is None else (
                    0 if st.session_state.fund_selected == '530810' else 1)
            )
            st.session_state.fund_selected = fund_type

# 메인 영역
if st.session_state.data_loaded and st.session_state.fund_selected:
    price_df = st.session_state.price_df
    Current_df = st.session_state.Current_df
    Gr_MinMax_df = st.session_state.Gr_MinMax_df
    fund_type = st.session_state.fund_selected

    # 데이터 필터링
    price_df, Current_filtered, Gr_MinMax_filtered, weight_col, min_col, max_col = filter_data_by_fund(
        price_df, Current_df, Gr_MinMax_df, fund_type
    )

    # 기준가 시트의 칼럼 이름들과 Current 시트의 CODE 매칭
    valid_codes = set(Current_filtered['CODE'].astype(str))
    # DATE 칼럼을 제외한 모든 칼럼이 CODE 후보
    # 각 칼럼 이름이 CODE 값과 매칭됨
    if 'DATE' in price_df.columns:
        code_columns = [col for col in price_df.columns if col != 'DATE']
    else:
        code_columns = list(price_df.columns)

    # 매칭되는 칼럼만 필터링
    matched_columns = [col for col in code_columns if str(col) in valid_codes]

    if len(matched_columns) == 0:
        st.warning(f"⚠️ 기준가 시트에서 Current 시트의 CODE와 매칭되는 칼럼이 없습니다.")
        price_filtered = pd.DataFrame()
    else:
        # DATE와 매칭된 칼럼들만 선택
        price_filtered = price_df[['DATE'] + matched_columns].copy() if 'DATE' in price_df.columns else price_df[
            matched_columns].copy()

    st.header("📈 요약 정보")

    # Sleeve별 비중 표시 (하나의 테이블로 통합)
    st.subheader("Sleeve별 정보")
    weight_df = Current_filtered[['SLEEVE', weight_col, 'DUR', min_col, max_col, 'GROUP']].copy()
    total_weight = weight_df[weight_col].sum()

    # 원본 비중을 % 형식으로 변환 (0.11 -> 11.00%)
    weight_df['원본 비중 (%)'] = (weight_df[weight_col] * 100).round(2).apply(lambda x: f"{x:.2f}")

    # 100% 환산 비중 계산
    weight_df['100% 환산 비중 (%)'] = (weight_df[weight_col] / total_weight * 100).round(2).apply(lambda x: f"{x:.2f}")

    # MIN/MAX 비중을 % 형식으로 변환
    weight_df['최소 비중 (%)'] = (weight_df[min_col] * 100).round(2).apply(lambda x: f"{x:.2f}")
    weight_df['최대 비중 (%)'] = (weight_df[max_col] * 100).round(2).apply(lambda x: f"{x:.2f}")

    # DUR 포맷팅
    weight_df['DUR'] = weight_df['DUR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

    # EXPECTED_R이 있으면 추가
    if 'EXPECTED_R' in Current_filtered.columns:
        expected_r_dict = Current_filtered.groupby('SLEEVE')['EXPECTED_R'].first().to_dict()
        weight_df['스코어링 기대수익률 (%)'] = weight_df['SLEEVE'].map(
            lambda
                x: f"{round(expected_r_dict.get(x, 0) * 100 if expected_r_dict.get(x, 0) < 1.0 else expected_r_dict.get(x, 0), 2):.2f}"
        )
        # 최종 테이블 (SLEEVE, 원본 비중, 100% 환산 비중, DUR, 최소 비중, 최대 비중, 스코어링 기대수익률, GROUP)
        weight_display_df = weight_df[['SLEEVE', '원본 비중 (%)', '100% 환산 비중 (%)', 'DUR',
                                       '최소 비중 (%)', '최대 비중 (%)', '스코어링 기대수익률 (%)', 'GROUP']].copy()
    else:
        # 최종 테이블 (EXPECTED_R이 없는 경우)
        weight_display_df = weight_df[['SLEEVE', '원본 비중 (%)', '100% 환산 비중 (%)', 'DUR',
                                       '최소 비중 (%)', '최대 비중 (%)', 'GROUP']].copy()

    st.dataframe(weight_display_df, use_container_width=True, hide_index=True)

    # 그룹별 비중 제약 표시 및 듀레이션 계산을 2열로 배치
    col1, col2 = st.columns(2)

    with col1:
        # 그룹별 비중 제약 표시
        st.subheader("그룹별 비중 제약")
        group_min_col = 'MIN_' + weight_col.replace('F', '')
        group_max_col = 'MAX_' + weight_col.replace('F', '')

        # 컬럼 존재 여부 확인
        if group_min_col in Gr_MinMax_filtered.columns and group_max_col in Gr_MinMax_filtered.columns:
            # 그룹별 MIN/MAX 비중을 % 형식으로 변환
            group_df = Gr_MinMax_filtered[['GROUP', group_min_col, group_max_col]].copy()
            group_df['최소 비중 (%)'] = (group_df[group_min_col] * 100).round(2)
            group_df['최대 비중 (%)'] = (group_df[group_max_col] * 100).round(2)

            group_display_df = group_df[['GROUP', '최소 비중 (%)', '최대 비중 (%)']].copy()
            st.dataframe(group_display_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"⚠️ 그룹별 비중 제약 컬럼({group_min_col}, {group_max_col})을 찾을 수 없습니다.")

    with col2:
        # 듀레이션 계산
        st.subheader("펀드 듀레이션")
        dur_df = Current_filtered[['SLEEVE', 'DUR', weight_col]].copy()
        dur_df['비중'] = dur_df[weight_col] / dur_df[weight_col].sum()
        portfolio_duration = (dur_df['DUR'] * dur_df['비중']).sum()
        st.markdown(f'<p style="font-size: 14px;">포트폴리오 듀레이션: {portfolio_duration:.2f}</p>', unsafe_allow_html=True)

    # 최적화 섹션
    st.header("최적화 설정")

    # 설명 텍스트
    st.markdown("""
    - 수익률/변동성은 3개월 기간 수익률/변동성으로 계산됩니다.
    - 개별 Sleeve별 비중은 위의 'Sleeve별 정보' 테이블의 최소, 최대비중을 적용합니다.
    - 그룹비중합 제약은 '그룹별 비중 제약' 테이블의 최소, 최대 비중을 적용합니다.
    """)

    # 3열 레이아웃으로 설정 표시
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("기대수익률 설정")
        return_method = st.radio(
            "기대수익률 계산 방법",
            ["과거수익률", "위원회 스코어링 결과", "몬테칼로 시뮬레이션"]
        )

        if return_method != "위원회 스코어링 결과":
            return_period = st.number_input(
                "참조 기간 (개월)",
                min_value=1,
                value=36,
                step=1,
                help="과거수익률이나 몬테칼로 시뮬레이션 분석에 사용할 과거 데이터 기간 (개월 수). 이 기간의 데이터를 사용하여 3개월 기간 수익률을 계산합니다."
            )
        else:
            # 위원회 스코어링 결과를 선택한 경우에도 return_period는 필요 없지만,
            # 코드 일관성을 위해 기본값 설정 (실제로는 사용되지 않음)
            return_period = 3

    with col2:
        st.subheader("변동성 설정")
        vol_period = st.number_input(
            "변동성 참조 기간 (개월)",
            min_value=1,
            value=36,
            step=1,
            help="과거변동성을 계산할 때 참조할 과거 데이터 기간 (개월 수). 이 기간의 데이터를 사용하여 3개월 기간 변동성을 계산합니다."
        )

    with col3:
        st.subheader("제약조건 설정")
        dur_buffer = st.number_input(
            "DUR 제약 버퍼 (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="현재 DUR에 플러스 마이너스 가능한 퍼센트"
        )

    # 텍스트 메시지는 3열 아래 행에 표시
    if return_method == "위원회 스코어링 결과":
        st.info("스코어링 기준 기대수익률은 'Sleeve별 정보' 테이블에 표시되어 있습니다.")

    # Risk-free Rate 입력 (Max Sharpe에 필요)
    st.subheader("Risk-free Rate")
    risk_free_rate = st.number_input(
        "Risk-free Rate (%)",
        value=0.0,
        step=0.1,
        help="Max Sharpe 최적화에 사용됩니다."
    )

    # 최적화 실행 버튼
    if st.button("Optimization", type="primary"):
        with st.spinner("최적화 진행 중..."):
            # 상관관계 행렬 계산 (몬테칼로 시뮬레이션에 필요)
            sleeves_list = Current_filtered['SLEEVE'].unique().tolist()
            corr_matrix = calculate_correlation_matrix(price_filtered, Current_filtered, sleeves_list)

            # 기대수익률 계산 (항상 3개월 기간 수익률로 계산)
            calc_period = 3  # 고정값
            if return_method == "과거수익률":
                expected_returns = calculate_historical_returns(price_filtered, Current_filtered, return_period,
                                                                calc_period)
            elif return_method == "위원회 스코어링 결과":
                # Current 시트의 EXPECTED_R 칼럼에서 읽어오기 (3개월 기간 수익률)
                # 엑셀의 % 형식 데이터는 이미 소수로 읽히므로, 퍼센트로 변환 필요
                expected_returns = {}
                if 'EXPECTED_R' in Current_filtered.columns:
                    for sleeve in Current_filtered['SLEEVE'].unique():
                        sleeve_data = Current_filtered[Current_filtered['SLEEVE'] == sleeve]
                        expected_r_values = sleeve_data['EXPECTED_R'].dropna()
                        if len(expected_r_values) > 0:
                            # 엑셀에서 읽은 값이 소수(0.0123)이면 퍼센트(1.23)로 변환
                            val = expected_r_values.iloc[0]
                            # 값이 1보다 작으면 소수로 간주하고 퍼센트로 변환
                            expected_returns[sleeve] = val * 100 if val < 1.0 else val
                        else:
                            expected_returns[sleeve] = 0.0
                else:
                    st.error("Current 시트에 EXPECTED_R 칼럼이 없습니다.")
                    expected_returns = {sleeve: 0.0 for sleeve in sleeves_list}
            else:  # 몬테칼로 시뮬레이션
                expected_returns = calculate_monte_carlo_returns(
                    price_filtered, Current_filtered, return_period,
                    corr_matrix, sleeves_list, calc_period
                )

            # 변동성 계산 (항상 3개월 기간 변동성으로 계산)
            vol_calc_period = 3  # 고정값
            volatilities = calculate_volatility(price_filtered, Current_filtered, vol_period, vol_calc_period)

            # 변동성 디버깅 정보
            if any(v == 0.0 for v in volatilities.values()):
                st.warning("⚠️ 일부 Sleeve의 변동성이 0입니다. 데이터가 충분한지 확인해주세요.")
                with st.expander("변동성 계산 결과 확인"):
                    for sleeve, vol in volatilities.items():
                        st.write(f"{sleeve}: {vol:.4f}%")

            # 세 가지 목적함수 모두 실행
            objectives = ["Max Sharpe", "Min Risk", "Risk Parity"]
            results = {}

            for obj in objectives:
                optimal_weights_raw, optimal_weights_normalized, portfolio_return, portfolio_vol, optimal_duration, weight_changes = optimize_portfolio(
                    Current_filtered, Gr_MinMax_filtered, expected_returns, volatilities,
                    corr_matrix, weight_col, min_col, max_col, obj,
                    risk_free_rate if obj == "Max Sharpe" else 0.0,
                    dur_buffer, portfolio_duration, return_period
                )

                # 샤프 비율 계산
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

                results[obj] = {
                    'optimal_weights': optimal_weights_raw,
                    'optimal_weights_normalized': optimal_weights_normalized,
                    'portfolio_return': portfolio_return,
                    'portfolio_vol': portfolio_vol,
                    'optimal_duration': optimal_duration,
                    'weight_changes': weight_changes,
                    'sharpe_ratio': sharpe_ratio
                }

            # 결과 저장
            st.session_state.optimization_results = results
            st.session_state.expected_returns = expected_returns
            st.session_state.volatilities = volatilities
            st.session_state.risk_free_rate = risk_free_rate

    # 최적화 결과 표시
    if 'optimization_results' in st.session_state:
        st.header("📊 최적화 결과")

        results = st.session_state.optimization_results
        current_weights_dict = Current_filtered.set_index('SLEEVE')[weight_col].to_dict()
        total_current = sum(current_weights_dict.values())

        # 통합 결과 테이블 생성
        comparison_data = []
        sleeves = Current_filtered['SLEEVE'].unique()

        for sleeve in sleeves:
            current_w = current_weights_dict.get(sleeve, 0)

            row_data = {
                'SLEEVE': sleeve,
                '현재 비중 (%)': f"{current_w * 100:.2f}%",
                'Max Sharpe 비중 (%)': f"{results['Max Sharpe']['optimal_weights'].get(sleeve, 0) * 100:.2f}%",
                'Min Risk 비중 (%)': f"{results['Min Risk']['optimal_weights'].get(sleeve, 0) * 100:.2f}%",
                'Risk Parity 비중 (%)': f"{results['Risk Parity']['optimal_weights'].get(sleeve, 0) * 100:.2f}%",
            }

            # 각 목적함수별 변화량
            row_data['Max Sharpe 변화 (%)'] = f"{results['Max Sharpe']['weight_changes'].get(sleeve, 0) * 100:+.2f}%"
            row_data['Min Risk 변화 (%)'] = f"{results['Min Risk']['weight_changes'].get(sleeve, 0) * 100:+.2f}%"
            row_data['Risk Parity 변화 (%)'] = f"{results['Risk Parity']['weight_changes'].get(sleeve, 0) * 100:+.2f}%"

            comparison_data.append(row_data)

        comparison_df = pd.DataFrame(comparison_data)

        # 통합 결과 테이블 표시 (CSV 다운로드 버튼 포함)
        col_title, col_csv = st.columns([10, 1])
        with col_title:
            st.subheader("비중 비교 (세 가지 목적함수)")
        with col_csv:
            # 한글 인코딩 문제 해결: UTF-8 BOM으로 인코딩
            csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
            csv_bytes = csv.encode('utf-8-sig')

            # 파일명 생성: 위원회_최적화결과_530810_yymmdd.csv
            latest_date = st.session_state.get('latest_date', pd.Timestamp.today())
            date_str = latest_date.strftime('%y%m%d')
            fund_type = st.session_state.fund_selected
            file_name = f"위원회_최적화결과_{fund_type}_{date_str}.csv"

            st.download_button(
                label="CSV",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv;charset=utf-8",
                key="download_comparison_csv"
            )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # 각 목적함수별 포트폴리오 지표 표시 (행과 열 전치)
        st.subheader("목적함수별 포트폴리오 지표")
        metrics_data = {
            '기대수익률 (%)': [
                f"{results['Max Sharpe']['portfolio_return']:.2f}",
                f"{results['Min Risk']['portfolio_return']:.2f}",
                f"{results['Risk Parity']['portfolio_return']:.2f}"
            ],
            '기대변동성 (%)': [
                f"{results['Max Sharpe']['portfolio_vol']:.2f}",
                f"{results['Min Risk']['portfolio_vol']:.2f}",
                f"{results['Risk Parity']['portfolio_vol']:.2f}"
            ],
            '듀레이션': [
                f"{results['Max Sharpe']['optimal_duration']:.2f}",
                f"{results['Min Risk']['optimal_duration']:.2f}",
                f"{results['Risk Parity']['optimal_duration']:.2f}"
            ],
            '샤프 비율': [
                f"{results['Max Sharpe']['sharpe_ratio']:.2f}",
                "-",
                "-"
            ]
        }

        metrics_df = pd.DataFrame(metrics_data, index=["Max Sharpe", "Min Risk", "Risk Parity"])
        metrics_df = metrics_df.T  # 행과 열 전치
        st.dataframe(metrics_df, use_container_width=True, hide_index=False)

else:
    st.info("사이드바에서 엑셀 파일을 업로드하고 펀드(530810 or 530950)를 선택해주세요.")

