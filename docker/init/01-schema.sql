CREATE TABLE prices (
    ticker   VARCHAR(20)  NOT NULL,
    date     DATE         NOT NULL,
    open     DOUBLE PRECISION,
    high     DOUBLE PRECISION,
    low      DOUBLE PRECISION,
    close    DOUBLE PRECISION,
    volume   DOUBLE PRECISION,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE fundamentals_sources (
    ticker  VARCHAR(20) PRIMARY KEY,
    source  VARCHAR(20) NOT NULL
);

CREATE TABLE fundamentals (
    ticker                    VARCHAR(20) NOT NULL,
    end_date                  DATE,
    quarter                   VARCHAR(10),
    fy                        INTEGER,
    fp                        VARCHAR(5),
    form                      VARCHAR(20),
    revenue                   DOUBLE PRECISION,
    net_income_loss           DOUBLE PRECISION,
    earnings_per_share_basic  DOUBLE PRECISION,
    operating_cash_flow       DOUBLE PRECISION,
    capex_raw                 DOUBLE PRECISION,
    dividends_per_share       DOUBLE PRECISION,
    cash_and_cash_equivalents DOUBLE PRECISION,
    total_debt                DOUBLE PRECISION,
    stockholders_equity       DOUBLE PRECISION,
    interest_expense          DOUBLE PRECISION,
    shares_outstanding        DOUBLE PRECISION,
    free_cash_flow            DOUBLE PRECISION,
    earnings_payout_ratio     DOUBLE PRECISION,
    net_debt                  DOUBLE PRECISION,
    debt_to_equity            DOUBLE PRECISION,
    PRIMARY KEY (ticker, end_date, quarter)
);

CREATE TABLE dividendology_compare (
    ticker              VARCHAR(20) PRIMARY KEY,
    name                TEXT,
    dividendology_topic TEXT,
    category            TEXT,
    dl_stance           TEXT,
    price               DOUBLE PRECISION,
    yield               DOUBLE PRECISION,
    fwd_pe              DOUBLE PRECISION,
    div_streak          INTEGER,
    div_cagr_5y         DOUBLE PRECISION,
    latest_dps          DOUBLE PRECISION,
    fy                  DOUBLE PRECISION,
    earn_payout         DOUBLE PRECISION,
    fcf_payout_sec      DOUBLE PRECISION,
    fcf_payout_yf       DOUBLE PRECISION,
    fcf_payout_used     DOUBLE PRECISION,
    fcf_source          TEXT,
    safety_score        DOUBLE PRECISION,
    safety_label        TEXT,
    ddm_fv              DOUBLE PRECISION,
    ddm_mos             DOUBLE PRECISION,
    dcf_fv              DOUBLE PRECISION,
    dcf_mos             DOUBLE PRECISION,
    notebook_signal     TEXT,
    alignment           TEXT,
    sec_data            BOOLEAN,
    notes               TEXT
);

CREATE INDEX idx_prices_ticker ON prices (ticker);
CREATE INDEX idx_fundamentals_ticker ON fundamentals (ticker);
CREATE INDEX idx_fundamentals_end_date ON fundamentals (end_date);
