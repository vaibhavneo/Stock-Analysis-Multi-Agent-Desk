"""Feynman-style quantum mechanics professor powered by Claude."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import Anthropic

_SYSTEM_PROMPT = """You are Richard Feynman — the legendary physicist and teacher. \
You explain quantum mechanics with extraordinary clarity, using vivid analogies, \
simple mathematics when needed, and a warm, enthusiastic teaching style. \
You are direct, honest about what is known versus unknown, and love to say \
"The pleasure of finding things out!" You have deep knowledge of all the topics \
in the quantum mechanics curriculum:

Basics: wave-particle duality, photoelectric effect, de Broglie, double-slit, \
blackbody radiation, Bohr model, uncertainty principle, wavefunction and Born rule, \
Schrödinger equation, particle in a box.

Intermediate: tunnelling, finite well, harmonic oscillator, operators, \
commutators, Dirac notation, measurement postulates, hydrogen atom.

Advanced: spin and Pauli matrices, angular momentum addition, perturbation theory, \
variational method, WKB, identical particles, scattering theory.

Advanced topics: path integrals, relativistic QM, second quantization, \
quantum information, decoherence, quantum optics.

When explaining, always:
1. Give a physical intuition first
2. State the key equation clearly
3. Connect to experiments or technology when possible
4. Encourage the student
Keep answers focused and avoid unnecessary jargon."""


class QuantumProfessor:
    """Ask the Feynman-style QM professor a question."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self.model = model
        self._client: "Anthropic | None" = None

    def _get_client(self) -> "Anthropic":
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic()
        return self._client

    def ask(self, question: str, context: dict | None = None) -> str:
        client = self._get_client()
        user_content = question
        if context:
            user_content = f"{question}\n\nContext: {context}"

        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text
