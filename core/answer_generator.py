import logging
import dspy
from typing import List

from core.lm import create_lm


logger = logging.getLogger(__name__)


class _ProofToAnswer(dspy.Signature):
    """
    You are a helpful assistant. You are given a user's natural language
    question and a logical proof trace from a Probabilistic Logic Network.

    Answer the question based strictly on the proof. If the proof is empty
    or does not contain enough information, say you don't know.
    Do not use any outside knowledge — only what the proof provides.
    Translate technical PLN terms into plain language.
    """
    question: str = dspy.InputField(desc="The original question")
    proof: str = dspy.InputField(desc="Raw PLN proof trace")
    answer: str = dspy.OutputField(desc="Natural language answer")


class AnswerGenerator:
    """
    Translates a PLN proof trace into a natural language response.
    This is the only LLM call in the query path.
    """

    def __init__(self):
        self._predict = dspy.Predict(_ProofToAnswer)
        self._predict.lm = create_lm()

    def generate(self, question: str, proof_traces: List[str]) -> str:
        if not proof_traces:
            return "I don't know — no proof was found for this question."

        proof_str = "\n".join(proof_traces)
        try:
            result = self._predict(question=question, proof=proof_str)
            return result.answer
        except Exception:
            logger.exception("Answer generation failed.")
            return "I was unable to generate an answer due to an internal error."
