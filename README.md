# ModGuard AI --- Context-Aware AI Content Moderation Engine

ModGuard AI is an academic prototype for context-aware online content
moderation. It combines multi-label text classification,
promotional-spam detection, configurable community policies, and
moderator review in a single moderation workflow.

> **Project status:** Academic prototype. The classifier was trained and
> validated using a synthetic, template-based dataset. Reported
> validation results should not be interpreted as evidence of real-world
> moderation performance. Evaluation on a separate test set and
> human-annotated real-world comments is still required.

## Key Features

-   **Multi-label harmful-content classification:** A fine-tuned
    DistilBERT model predicts seven labels independently: `toxicity`,
    `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`,
    and `sexual_explicit`.
-   **Parent-comment context:** A moderation request can include a
    parent comment to provide additional context. This is limited
    contextual input, not full conversation-history analysis.
-   **Promotional-spam detection:** A separate component checks
    promotional content and repeated promotional behaviour; this is
    distinct from the seven-label DistilBERT classifier.
-   **Configurable moderation policies:** YAML-based community settings
    define label thresholds and policy configuration.
-   **Human-in-the-loop review:** The system records moderation outcomes
    and supports a moderator review endpoint.
-   **API and dashboard:** FastAPI exposes moderation endpoints, while
    Streamlit provides an interface for submitting comments and viewing
    moderation results.

## Architecture

``` text
User / Moderator
       |
       v
Streamlit Dashboard
       |
       v
FastAPI Moderation API
       |
       +---- Promotional Spam Detection
       |
       +---- Repeated Promotional Behaviour Check
       |
       +---- DistilBERT Multi-label Classifier
       |          |
       |          +---- Optional Parent-comment Context
       |
       +---- YAML Policy Evaluation
       |
       +---- SQLite Moderation Records
       |
       v
Moderation Result / Moderator Review
```

## Technology Stack

  Area                            Technologies
  ------------------------------- ---------------------------------------
  Programming language            Python
  NLP model                       DistilBERT, Hugging Face Transformers
  Deep learning                   PyTorch
  Model training and evaluation   Scikit-learn, Pandas
  Backend API                     FastAPI
  Frontend                        Streamlit
  Policy configuration            YAML
  Prototype database              SQLite
  Training environment            Google Colab (T4 GPU)

## Model and Training

The text classifier uses DistilBERT for multi-label classification. Each
moderation label is predicted independently, allowing a comment to
receive more than one label.

Training configuration used for the current experimental model:

-   Base architecture: DistilBERT
-   Number of labels: 7
-   Maximum sequence length: 192 tokens
-   Batch size: 4
-   Epochs: 3
-   Learning rate: `2e-5`
-   Optimizer: AdamW
-   Loss function: `BCEWithLogitsLoss`

The current V3 dataset contains 1,500 synthetic examples split into
training (1,000), validation (250), and test (250) partitions. It
includes a limited set of contextual comment pairs.

The reported validation metrics for the current experiment were 1.00 for
precision, recall, F1-score, and average precision across the seven
labels. Because the dataset is synthetic and template-based, these
results may reflect patterns in the generated examples. They are not
proof of generalisation to real online discussions. The held-out test
set and a human-annotated real-world dataset should be evaluated before
making performance claims.

## Moderation Workflow

1.  A client submits a comment and metadata to the FastAPI moderation
    endpoint.
2.  The application checks the comment for promotional spam and repeated
    promotional behaviour.
3.  For supported English input, the DistilBERT classifier produces
    scores for the seven moderation labels. A parent comment may also be
    supplied.
4.  The policy component compares model scores with configured
    thresholds.
5.  The system returns a moderation decision, triggered rules,
    classifier scores, and version information, and stores the record in
    SQLite.
6.  A moderator can submit a review through the review endpoint.

**Important implementation note:** In the current prototype, detected
policy violations are routed to `manual_review`. Do not interpret YAML
action labels as proof that automatic removal or escalation is currently
enforced. The review endpoint also does not implement authentication, so
this project is not ready for production deployment.

## Project Structure

``` text
modguard-ai/
├── app/
│   ├── api.py
│   ├── context.py
│   ├── database.py
│   ├── moderation.py
│   ├── policy.py
│   ├── schemas.py
│   ├── spam_detection.py
│   └── spam_history.py
├── src/
│   ├── data/
│   │   ├── validate_contextual_data.py
│   │   ├── generate_contextual_v2.py
│   │   └── generate_contextual_v3.py
│   └── models/
│       └── train_contextual.py
├── data/
│   ├── contextual/
│   └── experiments/
│       ├── v2/
│       └── v3/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

Model checkpoints and local database files are intentionally excluded
from this repository. Large trained model artefacts should be hosted
separately (for example, on the Hugging Face Hub) and
downloaded/configured for the deployment environment.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/SemalJohari/modguard-ai.git
cd modguard-ai
```

### 2. Create and activate a virtual environment

On Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the model

Place the trained model directory at the path expected by
`app/moderation.py` (currently `models/contextual_02/`), or update the
model path to match your local setup.

The model directory is not included in this Git repository. You must
obtain the trained model artefacts separately before running model
inference.

### 5. Start the FastAPI backend

``` bash
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

The API documentation is available at:

``` text
http://127.0.0.1:8000/docs
```

### 6. Start the Streamlit dashboard

Open a second terminal, activate the same virtual environment, navigate
to the project directory, and run:

``` bash
streamlit run streamlit_app.py
```

Streamlit will print the local dashboard URL in the terminal.

## API Endpoints

  -----------------------------------------------------------------------------------
  Method                  Endpoint                            Purpose
  ----------------------- ----------------------------------- -----------------------
  `POST`                  `/moderate`                         Submit a comment for
                                                              moderation

  `POST`                  `/moderation/{comment_id}/review`   Record a moderator's
                                                              review
  -----------------------------------------------------------------------------------

The moderation request supports fields such as `comment_id`,
`author_id`, `text`, optional `parent_comment`, optional
`parent_comment_id`, `community_id`, and `language`.

## Limitations and Future Work

-   Replace or supplement synthetic training data with diverse,
    human-annotated real-world comments.
-   Evaluate on the held-out test split and report per-label precision,
    recall, F1-score, average precision, and threshold behaviour.
-   Assess generalisation across communities, writing styles, slang,
    spelling variations, and adversarial inputs.
-   Extend context handling beyond a single parent comment where
    appropriate.
-   Improve probability calibration and tune decision thresholds against
    moderation costs.
-   Add authentication, authorization, audit controls, rate limiting,
    and robust input validation.
-   Use a production database such as PostgreSQL and persistent storage
    for deployed moderation records.
-   Implement and test explicit policy actions if automatic removal or
    escalation is required.

## Disclaimer

ModGuard AI is developed for academic and demonstration purposes. It is
not a production-ready content moderation service. Model predictions can
be incorrect and should not be treated as definitive judgments about a
user or comment. Human review and further evaluation are necessary
before real-world use.

## Author

**Semal Johari**
