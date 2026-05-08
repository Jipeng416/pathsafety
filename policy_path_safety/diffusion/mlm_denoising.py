from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from policy_path_safety.core.types import MASK, PAD, Scenario
from policy_path_safety.diffusion.interfaces import CandidateProposal
from policy_path_safety.monitors.recoverability import is_consistent, text_from_state


class HFMaskedLMDenoisingSampler:
    """
    Real masked-LM-backed denoising proposal backend.

    This class exposes the interface needed by the path-safety monitor:
    a partial masked state u_t is mapped to a small candidate set of
    partially denoised states u'. The recoverability monitor then decides
    which u' states may survive.

    This is still not a large MDLM/LLaDA backend. It is the Phase 2
    engineering interface that replaces the pure random proposal proxy
    with a real masked-language-model scorer.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "auto",
        max_wordpiece_length: int = 1,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.max_wordpiece_length = int(max_wordpiece_length)
        self._score_cache: Dict[Tuple[str, Tuple[str, ...], int, str], float] = {}

        if self.mask_token is None or self.mask_token_id is None:
            raise ValueError(f"Model {model_name} does not expose a mask token.")

    def _wordpiece_ids(self, word: str):
        pieces = self.tokenizer.tokenize(word)
        if len(pieces) == 0:
            return None
        if len(pieces) > self.max_wordpiece_length:
            return None
        if any(piece.startswith("##") for piece in pieces[1:]):
            return None
        ids = self.tokenizer.convert_tokens_to_ids(pieces)
        if isinstance(ids, int):
            ids = [ids]
        return ids

    def score_word_at_position(
        self,
        context: str,
        state: Sequence[str],
        pos: int,
        candidate_word: str,
    ) -> float:
        if candidate_word == PAD:
            return -0.05

        key = (context, tuple(state), pos, candidate_word)
        if key in self._score_cache:
            return self._score_cache[key]

        ids = self._wordpiece_ids(candidate_word)
        if ids is None or len(ids) != 1:
            self._score_cache[key] = -20.0
            return -20.0

        candidate_id = ids[0]

        response_parts = []
        selected_mask_ordinal = None
        mask_count = 0

        for i, tok in enumerate(state):
            if tok == PAD:
                continue

            if i == pos:
                response_parts.append(self.mask_token)
                selected_mask_ordinal = mask_count
                mask_count += 1
            elif tok == MASK:
                response_parts.append(self.mask_token)
                mask_count += 1
            else:
                response_parts.append(tok)

        if selected_mask_ordinal is None:
            self._score_cache[key] = -20.0
            return -20.0

        text = context.lower() + " [SEP] " + " ".join(response_parts)

        with torch.no_grad():
            enc = self.tokenizer(text, return_tensors="pt").to(self.device)
            mask_positions = (enc.input_ids[0] == self.mask_token_id).nonzero(as_tuple=False).flatten()

            if len(mask_positions) <= selected_mask_ordinal:
                self._score_cache[key] = -20.0
                return -20.0

            selected_pos = mask_positions[selected_mask_ordinal]
            logits = self.model(**enc).logits[0, selected_pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            score = float(log_probs[candidate_id].detach().cpu())

        self._score_cache[key] = score
        return score

    @staticmethod
    def _weighted_choice(items, rng: random.Random):
        total = sum(max(item.weight, 0.0) for item in items)
        if total <= 0:
            return rng.choice(list(items))

        r = rng.random() * total
        acc = 0.0

        for item in items:
            acc += max(item.weight, 0.0)
            if r <= acc:
                return item

        return items[-1]

    def propose(
        self,
        scenario: Scenario,
        state: Sequence[str],
        rng: random.Random,
        block_size: int,
        num_candidates: int,
        prior_strength: float,
        model_score_weight: float,
    ) -> List[CandidateProposal]:
        consistent = [
            response for response in scenario.responses
            if is_consistent(state, scenario.padded_tokens(response))
        ]

        if not consistent:
            return []

        masked_positions = [i for i, tok in enumerate(state) if tok == MASK]

        if not masked_positions:
            return [
                CandidateProposal(
                    state=tuple(state),
                    family="terminal",
                    score=0.0,
                    filled_positions=tuple(),
                    source_text=text_from_state(state),
                )
            ]

        k = max(1, min(int(block_size), len(masked_positions)))
        chosen_positions = tuple(sorted(rng.sample(masked_positions, k=k)))

        proposals: List[CandidateProposal] = []
        seen = set()

        for _ in range(int(num_candidates)):
            response = self._weighted_choice(consistent, rng)
            full = scenario.padded_tokens(response)
            new_state = list(state)

            score = float(prior_strength) * math.log(max(response.weight, 1e-12))

            for pos in chosen_positions:
                word = full[pos]
                new_state[pos] = word
                score += float(model_score_weight) * self.score_word_at_position(
                    context=scenario.context,
                    state=state,
                    pos=pos,
                    candidate_word=word,
                )

            key = tuple(new_state)
            if key in seen:
                continue

            seen.add(key)
            proposals.append(
                CandidateProposal(
                    state=tuple(new_state),
                    family=response.family,
                    score=float(score),
                    filled_positions=chosen_positions,
                    source_text=response.text,
                )
            )

        return proposals
