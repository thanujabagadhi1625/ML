"""
schemas.py
----------
Typed data contracts for the three core entities in the system:
Question, User, Submission.

These dataclasses aren't strictly required by pandas, but they document
the exact schema every downstream module assumes, and can be used with
`pydantic` at an API boundary (e.g. FastAPI request/response models)
without changing anything else in the pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class Question:
    question_id: int
    title: str
    description: str          # free-text problem statement -> consumed by nlp_cluster.py
    difficulty: str            # one of {"Easy", "Medium", "Hard"}
    topic_tags: List[str] = field(default_factory=list)
    acceptance_rate: float = 0.5  # global site-wide acceptance rate, 0-1


@dataclass
class User:
    user_id: int
    account_age_days: int
    latent_skill: float        # NOT exposed to models directly; drives synthetic generation only
    contest_rating: float       # regression TARGET for predictor.py


@dataclass
class Submission:
    user_id: int
    question_id: int
    timestamp: datetime
    status: str                 # one of config.SUBMISSION_STATUSES
    runtime_ms: float
    language: str

    @property
    def is_accepted(self) -> int:
        return int(self.status == "Accepted")


# Canonical column ordering used when constructing pandas DataFrames.
QUESTION_COLUMNS = ["question_id", "title", "description", "difficulty", "topic_tags", "acceptance_rate"]
USER_COLUMNS = ["user_id", "account_age_days", "latent_skill", "contest_rating"]
SUBMISSION_COLUMNS = ["user_id", "question_id", "timestamp", "status", "runtime_ms", "language"]
