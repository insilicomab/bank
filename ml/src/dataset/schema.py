from pandera import Check, Column, DataFrameSchema, Index, SeriesSchema

JOB = [
    "blue-collar",
    "entrepreneur",
    "management",
    "retired",
    "services",
    "technician",
    "admin.",
    "self-employed",
    "housemaid",
    "unemployed",
    "unknown",
    "student",
]

MARITAL = ["married", "single", "divorced"]

EDUCATION = ["secondary", "primary", "tertiary", "unknown"]

DEFAULT = ["no", "yes"]

HOUSING = ["yes", "no"]

LOAN = ["no", "yes"]

CONTACT = ["cellular", "unknown", "telephone"]

MONTH = [
    "apr",
    "feb",
    "jan",
    "jun",
    "sep",
    "may",
    "aug",
    "mar",
    "jul",
    "nov",
    "oct",
    "dec",
]

POUTCOME = ["unknown", "failure", "success", "other"]

DAY = [i for i in range(1, 32, 1)]

BASE_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(int),
        "age": Column(int),
        "job": Column(str, checks=Check.isin(JOB)),
        "marital": Column(str, checks=Check.isin(MARITAL)),
        "education": Column(str, checks=Check.isin(EDUCATION)),
        "default": Column(str, checks=Check.isin(DEFAULT)),
        "balance": Column(int),
        "housing": Column(str, checks=Check.isin(HOUSING)),
        "loan": Column(str, checks=Check.isin(LOAN)),
        "contact": Column(str, checks=Check.isin(CONTACT)),
        "day": Column(int, checks=Check.isin(DAY)),
        "month": Column(str, checks=Check.isin(MONTH)),
        "duration": Column(int),
        "campaign": Column(int),
        "pdays": Column(int),
        "previous": Column(int),
        "poutcome": Column(str, checks=Check.isin(POUTCOME)),
        "y": Column(int, required=False),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

PREPROCESSED_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(int),
        "age": Column(int),
        "job": Column(str, checks=Check.isin(JOB)),
        "marital": Column(str, checks=Check.isin(MARITAL)),
        "education": Column(str, checks=Check.isin(EDUCATION)),
        "default": Column(str, checks=Check.isin(DEFAULT)),
        "balance": Column(int),
        "housing": Column(str, checks=Check.isin(HOUSING)),
        "loan": Column(str, checks=Check.isin(LOAN)),
        "contact": Column(str, checks=Check.isin(CONTACT)),
        "duration": Column(int),
        "campaign": Column(int),
        "pdays": Column(int),
        "previous": Column(int),
        "poutcome": Column(str, checks=Check.isin(POUTCOME)),
        "isOver60yr": Column(int),
        "datetime_int": Column(float),
        "y": Column(int, required=False),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

X_SCHEMA = DataFrameSchema(
    columns={
        "age": Column(int),
        "job": Column(float),
        "marital": Column(float),
        "education": Column(float),
        "default": Column(float),
        "balance": Column(int),
        "housing": Column(float),
        "loan": Column(float),
        "contact": Column(float),
        "duration": Column(int),
        "campaign": Column(int),
        "pdays": Column(int),
        "previous": Column(int),
        "poutcome": Column(float),
        "isOver60yr": Column(int),
        "datetime_int": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

Y_SCHEMA = SeriesSchema(
    int,
    checks=Check.isin((0, 1)),
    nullable=False,
    coerce=True,
)

RAW_PREDICTION_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(int),
        "age": Column(int),
        "job": Column(str, checks=Check.isin(JOB)),
        "marital": Column(str, checks=Check.isin(MARITAL)),
        "education": Column(str, checks=Check.isin(EDUCATION)),
        "default": Column(str, checks=Check.isin(DEFAULT)),
        "balance": Column(int),
        "housing": Column(str, checks=Check.isin(HOUSING)),
        "loan": Column(str, checks=Check.isin(LOAN)),
        "contact": Column(str, checks=Check.isin(CONTACT)),
        "day": Column(int, checks=Check.isin(DAY)),
        "month": Column(str, checks=Check.isin(MONTH)),
        "duration": Column(int),
        "campaign": Column(int),
        "pdays": Column(int),
        "previous": Column(int),
        "poutcome": Column(str, checks=Check.isin(POUTCOME)),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

PREPROCESSED_PREDICTION_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(int),
        "age": Column(int),
        "job": Column(float),
        "marital": Column(float),
        "education": Column(float),
        "default": Column(float),
        "balance": Column(int),
        "housing": Column(float),
        "loan": Column(float),
        "contact": Column(float),
        "duration": Column(int),
        "campaign": Column(int),
        "pdays": Column(int),
        "previous": Column(int),
        "poutcome": Column(float),
        "isOver60yr": Column(int),
        "datetime_int": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

PREDICTION_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(int),
        "prediction": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)
