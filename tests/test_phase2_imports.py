from policy_path_safety.diffusion.interfaces import CandidateProposal
from policy_path_safety.diffusion.mlm_denoising import HFMaskedLMDenoisingSampler
from policy_path_safety.experiments.run_phase2_diffusion import summarize_phase2


def test_phase2_imports():
    assert CandidateProposal is not None
    assert HFMaskedLMDenoisingSampler is not None
    assert summarize_phase2 is not None
