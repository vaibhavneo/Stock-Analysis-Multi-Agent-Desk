"""Curated quantum mechanics knowledge base: books, topics, curriculum graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

LEVELS = ["basics", "intermediate", "advanced", "advanced_topics"]


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    authors: str
    level: str
    topics: List[str]
    note: str
    why_read: str


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    level: str
    prerequisites: List[str]
    key_concepts: List[str]
    key_equations: List[str]
    intuition: str
    book_refs: List[str]
    problem_ids: List[str]


BOOKS: Dict[str, Book] = {b.id: b for b in [
    Book(
        id="feynman-vol3",
        title="The Feynman Lectures on Physics, Vol. III",
        authors="Richard P. Feynman, Robert B. Leighton, Matthew Sands",
        level="basics",
        topics=["wave-particle-duality", "double-slit", "quantum-behavior", "uncertainty-principle"],
        note="Free online at feynmanlectures.caltech.edu",
        why_read="The best intuition-first introduction to QM ever written. Feynman explains quantum behavior from scratch with zero hand-waving.",
    ),
    Book(
        id="griffiths",
        title="Introduction to Quantum Mechanics",
        authors="David J. Griffiths",
        level="intermediate",
        topics=["schrodinger-equation", "particle-in-a-box", "harmonic-oscillator", "hydrogen-atom", "spin-pauli"],
        note="3rd edition (2018) recommended",
        why_read="The standard undergraduate text. Clear derivations, great problems, covers all core QM in one volume.",
    ),
    Book(
        id="shankar",
        title="Principles of Quantum Mechanics",
        authors="R. Shankar",
        level="intermediate",
        topics=["dirac-notation", "operators-observables", "harmonic-oscillator", "hydrogen-atom", "time-independent-perturbation"],
        note="2nd edition",
        why_read="Rigorous mathematical foundations with excellent Dirac notation treatment. Best bridge from undergrad to graduate QM.",
    ),
    Book(
        id="sakurai",
        title="Modern Quantum Mechanics",
        authors="J. J. Sakurai, Jim Napolitano",
        level="advanced",
        topics=["spin-pauli", "angular-momentum-addition", "time-independent-perturbation", "time-dependent-perturbation", "scattering-theory"],
        note="3rd edition (2020)",
        why_read="The graduate standard. Starts with spin rather than waves, develops formalism powerfully. Essential for PhD students.",
    ),
    Book(
        id="cohen-tannoudji",
        title="Quantum Mechanics (2 vols.)",
        authors="Claude Cohen-Tannoudji, Bernard Diu, Franck Laloë",
        level="advanced",
        topics=["operators-observables", "harmonic-oscillator", "angular-momentum-addition", "time-dependent-perturbation", "identical-particles"],
        note="Nobel Prize author",
        why_read="Encyclopedic coverage with detailed complements on every chapter. A comprehensive reference for the working physicist.",
    ),
    Book(
        id="dirac",
        title="The Principles of Quantum Mechanics",
        authors="Paul A. M. Dirac",
        level="advanced",
        topics=["dirac-notation", "operators-observables", "measurement-postulates", "relativistic-qm"],
        note="4th edition (1958), historically foundational",
        why_read="Dirac invented much of modern QM notation. Reading the original source builds deep appreciation for the formalism.",
    ),
    Book(
        id="zettili",
        title="Quantum Mechanics: Concepts and Applications",
        authors="Nouredine Zettili",
        level="intermediate",
        topics=["schrodinger-equation", "particle-in-a-box", "harmonic-oscillator", "hydrogen-atom", "spin-pauli"],
        note="2nd edition; contains many solved problems",
        why_read="Excellent problem-solving focus with fully worked examples. Great companion to Griffiths for building calculation skills.",
    ),
    Book(
        id="ballentine",
        title="Quantum Mechanics: A Modern Development",
        authors="Leslie E. Ballentine",
        level="advanced",
        topics=["measurement-postulates", "identical-particles", "scattering-theory", "decoherence"],
        note="Emphasizes the statistical (ensemble) interpretation",
        why_read="Provides the most rigorous and philosophically careful treatment of QM foundations and measurement theory.",
    ),
    Book(
        id="townsend",
        title="A Modern Approach to Quantum Mechanics",
        authors="John S. Townsend",
        level="intermediate",
        topics=["spin-pauli", "dirac-notation", "operators-observables", "angular-momentum-addition"],
        note="Starts with spin-½, like Sakurai but at undergrad level",
        why_read="Elegant spin-first approach that builds formalism early. Excellent for developing physical intuition alongside the math.",
    ),
    Book(
        id="weinberg-qm",
        title="Lectures on Quantum Mechanics",
        authors="Steven Weinberg",
        level="advanced",
        topics=["measurement-postulates", "time-independent-perturbation", "scattering-theory", "decoherence"],
        note="Reflects Weinberg's view on interpretations",
        why_read="A Nobel Laureate's perspective on QM fundamentals and what the formalism really means. Valuable for advanced students.",
    ),
    Book(
        id="landau-lifshitz-qm",
        title="Quantum Mechanics: Non-Relativistic Theory",
        authors="Lev D. Landau, Evgeny M. Lifshitz",
        level="advanced",
        topics=["schrodinger-equation", "harmonic-oscillator", "angular-momentum-addition", "wkb", "scattering-theory"],
        note="Course of Theoretical Physics Vol. 3",
        why_read="The Russian master class in theoretical physics. Dense but rewarding; every problem is a gem.",
    ),
    Book(
        id="nielsen-chuang",
        title="Quantum Computation and Quantum Information",
        authors="Michael A. Nielsen, Isaac L. Chuang",
        level="advanced_topics",
        topics=["quantum-information", "decoherence"],
        note="'Mike and Ike' — the bible of QC",
        why_read="The definitive textbook on quantum computing and information theory. Essential for anyone working in quantum technology.",
    ),
    Book(
        id="feynman-hibbs",
        title="Quantum Mechanics and Path Integrals",
        authors="Richard P. Feynman, Albert R. Hibbs",
        level="advanced_topics",
        topics=["path-integrals"],
        note="Emended edition by Styer (2010)",
        why_read="Feynman's own development of path integrals — a completely different and beautiful way to think about quantum amplitudes.",
    ),
]}

TOPICS: Dict[str, Topic] = {t.id: t for t in [
    # ─── BASICS ──────────────────────────────────────────────────────────────
    Topic(
        id="wave-particle-duality",
        title="Wave–Particle Duality",
        level="basics",
        prerequisites=[],
        key_concepts=["photons behave as both wave and particle", "matter has wave properties", "complementarity principle"],
        key_equations=[r"E = h\nu", r"\lambda = h/p"],
        intuition="Light is neither a pure wave nor a pure particle — it is something new that shows wave behavior in some experiments and particle behavior in others. The question 'which is it?' is the wrong question.",
        book_refs=["feynman-vol3", "griffiths"],
        problem_ids=["duality-p1"],
    ),
    Topic(
        id="photoelectric-effect",
        title="The Photoelectric Effect",
        level="basics",
        prerequisites=["wave-particle-duality"],
        key_concepts=["photon energy quantization", "work function", "threshold frequency", "Einstein's 1905 explanation"],
        key_equations=[r"E_{\text{photon}} = h\nu", r"KE_{\max} = h\nu - \phi"],
        intuition="Shining light on metal ejects electrons only if the frequency is high enough — not the intensity. Einstein realized light must come in discrete packets (photons) each with energy hν.",
        book_refs=["feynman-vol3", "griffiths"],
        problem_ids=["photo-p1"],
    ),
    Topic(
        id="de-broglie",
        title="de Broglie Wavelength",
        level="basics",
        prerequisites=["wave-particle-duality"],
        key_concepts=["matter waves", "de Broglie relation", "wave vector k", "electron diffraction"],
        key_equations=[r"\lambda = \frac{h}{p}", r"k = \frac{p}{\hbar} = \frac{2\pi}{\lambda}"],
        intuition="Just as Einstein showed light has momentum p = E/c = h/λ, de Broglie reversed it: particles with momentum p have an associated wavelength λ = h/p. Electrons really do diffract like waves.",
        book_refs=["feynman-vol3", "griffiths", "zettili"],
        problem_ids=["debroglie-p1", "debroglie-p2"],
    ),
    Topic(
        id="double-slit",
        title="The Double-Slit Experiment",
        level="basics",
        prerequisites=["wave-particle-duality", "de-broglie"],
        key_concepts=["interference of probability amplitudes", "which-path information destroys interference", "quantum measurement affects the system"],
        key_equations=[r"P_{12} = |A_1 + A_2|^2 \neq P_1 + P_2"],
        intuition="When you don't know which slit the electron went through, its probability amplitudes interfere. The moment you measure which path, interference disappears. This is the central mystery of quantum mechanics.",
        book_refs=["feynman-vol3"],
        problem_ids=[],
    ),
    Topic(
        id="blackbody-radiation",
        title="Blackbody Radiation & Planck's Law",
        level="basics",
        prerequisites=[],
        key_concepts=["ultraviolet catastrophe", "energy quantization", "Planck constant", "Stefan-Boltzmann law"],
        key_equations=[r"E_n = n h\nu", r"u(\nu,T) = \frac{8\pi h\nu^3}{c^3} \frac{1}{e^{h\nu/k_B T}-1}"],
        intuition="Classical physics predicted infinite energy at high frequencies (the UV catastrophe). Planck's radical fix: energy is only exchanged in discrete chunks E = nhν. This was the birth of quantum theory.",
        book_refs=["griffiths", "zettili"],
        problem_ids=[],
    ),
    Topic(
        id="bohr-model",
        title="The Bohr Model of the Atom",
        level="basics",
        prerequisites=["photoelectric-effect", "de-broglie"],
        key_concepts=["quantized orbits", "energy levels", "emission and absorption spectra", "Rydberg formula"],
        key_equations=[r"E_n = -\frac{13.6\,\text{eV}}{n^2}", r"\frac{1}{\lambda} = R_\infty\!\left(\frac{1}{n_f^2} - \frac{1}{n_i^2}\right)"],
        intuition="Bohr postulated that electrons only occupy orbits where their de Broglie wave fits exactly around the circle. This explains hydrogen's spectral lines but fails for multi-electron atoms.",
        book_refs=["griffiths", "zettili", "feynman-vol3"],
        problem_ids=["bohr-p1"],
    ),
    Topic(
        id="uncertainty-principle",
        title="Heisenberg Uncertainty Principle",
        level="basics",
        prerequisites=["wave-particle-duality", "de-broglie"],
        key_concepts=["position-momentum uncertainty", "energy-time uncertainty", "not an experimental limitation but fundamental"],
        key_equations=[r"\Delta x \,\Delta p \geq \frac{\hbar}{2}", r"\Delta E \,\Delta t \geq \frac{\hbar}{2}"],
        intuition="To know where a particle is precisely, you need a narrow wave packet — but a narrow packet requires many wavelengths, meaning many momenta. Localizing position unavoidably spreads momentum. This is geometry, not ignorance.",
        book_refs=["feynman-vol3", "griffiths", "shankar"],
        problem_ids=["uncertainty-p1", "uncertainty-p2"],
    ),
    Topic(
        id="wavefunction-born-rule",
        title="The Wavefunction and Born Rule",
        level="basics",
        prerequisites=["wave-particle-duality", "double-slit"],
        key_concepts=["wavefunction ψ(x,t)", "probability density |ψ|²", "normalization", "superposition principle"],
        key_equations=[r"P(x) = |\psi(x,t)|^2", r"\int_{-\infty}^{\infty}|\psi|^2 \,dx = 1"],
        intuition="The wavefunction is a complex probability amplitude. Its squared magnitude gives the probability of finding the particle at a location. Before measurement the particle is in a superposition; measurement 'collapses' the wavefunction.",
        book_refs=["griffiths", "feynman-vol3", "shankar"],
        problem_ids=[],
    ),
    Topic(
        id="schrodinger-equation",
        title="The Schrödinger Equation",
        level="basics",
        prerequisites=["wavefunction-born-rule", "de-broglie"],
        key_concepts=["time-dependent Schrödinger equation", "time-independent form", "stationary states", "energy eigenvalues"],
        key_equations=[
            r"i\hbar\frac{\partial\psi}{\partial t} = \hat{H}\psi",
            r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V\psi = E\psi",
        ],
        intuition="Schrödinger's equation is the quantum version of F=ma. It tells us how the wavefunction evolves in time. For stationary states with definite energy, it reduces to a spatial eigenvalue equation.",
        book_refs=["griffiths", "shankar", "zettili"],
        problem_ids=[],
    ),
    Topic(
        id="particle-in-a-box",
        title="Infinite Square Well (Particle in a Box)",
        level="basics",
        prerequisites=["schrodinger-equation", "wavefunction-born-rule"],
        key_concepts=["boundary conditions", "quantized energies", "standing waves", "zero-point energy"],
        key_equations=[
            r"E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}",
            r"\psi_n(x) = \sqrt{\frac{2}{L}}\sin\!\left(\frac{n\pi x}{L}\right)",
        ],
        intuition="Confine a particle between two walls. Only standing-wave solutions fit — exactly like a guitar string. The energy is quantized as n², and even the ground state (n=1) has non-zero energy: the zero-point energy.",
        book_refs=["griffiths", "shankar", "zettili", "feynman-vol3"],
        problem_ids=["box-p1", "box-p2"],
    ),
    # ─── INTERMEDIATE ─────────────────────────────────────────────────────────
    Topic(
        id="step-barrier-tunneling",
        title="Potential Step, Barrier & Quantum Tunneling",
        level="intermediate",
        prerequisites=["particle-in-a-box", "schrodinger-equation"],
        key_concepts=["reflection and transmission coefficients", "evanescent wave", "tunneling through classically forbidden region", "WKB tunneling rate"],
        key_equations=[
            r"T \approx e^{-2\kappa L},\quad \kappa = \sqrt{2m(V_0-E)}/\hbar",
        ],
        intuition="Classically, a ball with less energy than a hill cannot cross. Quantum mechanically, the wavefunction decays exponentially inside the barrier but doesn't vanish — there's a non-zero probability of appearing on the other side. Tunnel diodes and nuclear fusion rely on this.",
        book_refs=["griffiths", "shankar", "zettili"],
        problem_ids=["tunnel-p1"],
    ),
    Topic(
        id="finite-well",
        title="Finite Square Well",
        level="intermediate",
        prerequisites=["particle-in-a-box", "step-barrier-tunneling"],
        key_concepts=["bound states", "tunneling into classically forbidden region", "transcendental equations for eigenvalues"],
        key_equations=[r"k\tan(ka) = \kappa,\quad k = \sqrt{2mE}/\hbar,\quad \kappa = \sqrt{2m(V_0-E)}/\hbar"],
        intuition="A finite well has fewer bound states than the infinite well, and they're slightly lower in energy. The wavefunction leaks outside the well — the particle can 'be found' in classically forbidden regions.",
        book_refs=["griffiths", "shankar"],
        problem_ids=[],
    ),
    Topic(
        id="harmonic-oscillator",
        title="Quantum Harmonic Oscillator",
        level="intermediate",
        prerequisites=["schrodinger-equation", "particle-in-a-box"],
        key_concepts=["ladder operators (creation and annihilation)", "equally-spaced energy levels", "zero-point energy", "coherent states"],
        key_equations=[
            r"E_n = \hbar\omega\!\left(n + \tfrac{1}{2}\right)",
            r"\hat{H} = \hbar\omega\!\left(\hat{a}^\dagger\hat{a} + \tfrac{1}{2}\right)",
        ],
        intuition="The harmonic oscillator is the most important quantum system — it appears in every quantum field theory. Remarkably, the algebra of raising/lowering operators solves it without ever writing down the wavefunction.",
        book_refs=["griffiths", "shankar", "sakurai", "cohen-tannoudji"],
        problem_ids=["ho-p1", "ho-p2"],
    ),
    Topic(
        id="operators-observables",
        title="Operators and Observables",
        level="intermediate",
        prerequisites=["wavefunction-born-rule", "schrodinger-equation"],
        key_concepts=["Hermitian operators", "eigenvalues and eigenstates", "expectation values", "spectrum of observables"],
        key_equations=[r"\langle A \rangle = \langle\psi|\hat{A}|\psi\rangle", r"\hat{A}|\phi_n\rangle = a_n|\phi_n\rangle"],
        intuition="In QM every measurable quantity is represented by a Hermitian operator. Measuring it gives one of its eigenvalues, and the state 'collapses' to the corresponding eigenstate. The expectation value is the average over many measurements.",
        book_refs=["griffiths", "shankar", "dirac", "sakurai"],
        problem_ids=["op-p1"],
    ),
    Topic(
        id="commutators",
        title="Commutators and Compatible Observables",
        level="intermediate",
        prerequisites=["operators-observables"],
        key_concepts=["commutator [A,B]=AB-BA", "canonical commutation relation [x,p]=iℏ", "simultaneous eigenstates iff [A,B]=0", "generalized uncertainty principle"],
        key_equations=[r"[\hat{x},\hat{p}] = i\hbar", r"\Delta A\,\Delta B \geq \tfrac{1}{2}|{\langle[\hat{A},\hat{B}]\rangle}|"],
        intuition="Two operators commute iff they can be simultaneously measured precisely. Position and momentum famously do NOT commute — this is the algebraic root of the uncertainty principle, not just a measurement disturbance story.",
        book_refs=["griffiths", "shankar", "sakurai"],
        problem_ids=["comm-p1"],
    ),
    Topic(
        id="dirac-notation",
        title="Dirac Notation & Hilbert Space",
        level="intermediate",
        prerequisites=["operators-observables", "wavefunction-born-rule"],
        key_concepts=["kets |ψ⟩ and bras ⟨ψ|", "inner product ⟨φ|ψ⟩", "completeness relation", "representation in different bases"],
        key_equations=[r"\langle x|\psi\rangle = \psi(x)", r"\sum_n |n\rangle\langle n| = \hat{I}"],
        intuition="Dirac notation is a powerful shorthand that separates physics from any particular representation. A state vector |ψ⟩ exists independently of whether we write it as a wavefunction or a spin vector.",
        book_refs=["shankar", "dirac", "sakurai", "townsend"],
        problem_ids=[],
    ),
    Topic(
        id="measurement-postulates",
        title="Postulates of Quantum Mechanics",
        level="intermediate",
        prerequisites=["dirac-notation", "operators-observables", "wavefunction-born-rule"],
        key_concepts=["state vector postulate", "observable postulate", "measurement postulate (projection)", "time evolution postulate", "Born rule"],
        key_equations=[r"P(a_n) = |\langle a_n|\psi\rangle|^2", r"|\psi\rangle \xrightarrow{\text{measure}} |a_n\rangle"],
        intuition="QM has a small set of axioms: (1) state is a vector; (2) observables are Hermitian operators; (3) measurement gives an eigenvalue with probability = squared amplitude; (4) between measurements, Schrödinger evolves the state.",
        book_refs=["shankar", "dirac", "ballentine", "weinberg-qm"],
        problem_ids=[],
    ),
    Topic(
        id="hydrogen-atom",
        title="The Hydrogen Atom",
        level="intermediate",
        prerequisites=["schrodinger-equation", "operators-observables", "bohr-model"],
        key_concepts=["3D Schrödinger equation in spherical coordinates", "quantum numbers n, l, m", "orbital shapes", "exact energy spectrum"],
        key_equations=[
            r"E_n = -\frac{13.6\,\text{eV}}{n^2}",
            r"\psi_{nlm}(r,\theta,\phi) = R_{nl}(r)Y_l^m(\theta,\phi)",
        ],
        intuition="Solving Schrödinger's equation for the Coulomb potential gives the exact hydrogen spectrum — a triumph of QM. Three quantum numbers (n, l, m) emerge naturally from the boundary conditions in 3D.",
        book_refs=["griffiths", "shankar", "cohen-tannoudji", "zettili"],
        problem_ids=["hydrogen-p1", "hydrogen-p2"],
    ),
    # ─── ADVANCED ─────────────────────────────────────────────────────────────
    Topic(
        id="spin-pauli",
        title="Spin and Pauli Matrices",
        level="advanced",
        prerequisites=["operators-observables", "commutators", "dirac-notation"],
        key_concepts=["intrinsic angular momentum", "spin-½", "Pauli matrices σₓ σᵧ σᵤ", "Stern-Gerlach experiment", "spinors"],
        key_equations=[
            r"S_z|\pm\rangle = \pm\frac{\hbar}{2}|\pm\rangle",
            r"\boldsymbol{\sigma}_x = \begin{pmatrix}0&1\\1&0\end{pmatrix},\quad \boldsymbol{\sigma}_z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}",
        ],
        intuition="Electrons have an intrinsic angular momentum called spin that has no classical analogue. A spin-½ particle in a magnetic field can only be 'up' or 'down'. Pauli matrices encode the algebra of spin and underpin all of quantum information.",
        book_refs=["sakurai", "townsend", "griffiths", "cohen-tannoudji"],
        problem_ids=["spin-p1"],
    ),
    Topic(
        id="angular-momentum-addition",
        title="Addition of Angular Momentum & Clebsch–Gordan",
        level="advanced",
        prerequisites=["spin-pauli", "hydrogen-atom"],
        key_concepts=["total angular momentum J = L + S", "Clebsch–Gordan coefficients", "triplet and singlet states", "j = |l-s| to l+s"],
        key_equations=[r"|j_1 j_2; j m\rangle = \sum_{m_1 m_2} C^{jm}_{j_1 m_1 j_2 m_2}|j_1 m_1\rangle|j_2 m_2\rangle"],
        intuition="When two angular momenta couple, the total can range from |j₁-j₂| to j₁+j₂. The Clebsch–Gordan coefficients tell you the overlap between the coupled and uncoupled bases — they appear throughout atomic, nuclear, and particle physics.",
        book_refs=["sakurai", "cohen-tannoudji", "griffiths"],
        problem_ids=[],
    ),
    Topic(
        id="time-independent-perturbation",
        title="Time-Independent Perturbation Theory",
        level="advanced",
        prerequisites=["operators-observables", "hydrogen-atom", "spin-pauli"],
        key_concepts=["small perturbation H'", "first and second order energy corrections", "first order state corrections", "degenerate perturbation theory"],
        key_equations=[
            r"E_n^{(1)} = \langle n^{(0)}|H'|n^{(0)}\rangle",
            r"E_n^{(2)} = \sum_{m\neq n}\frac{|\langle m|H'|n\rangle|^2}{E_n^{(0)}-E_m^{(0)}}",
        ],
        intuition="Most quantum systems can't be solved exactly. If H = H₀ + λH' with λ small, we can expand in powers of λ. First-order corrections involve only the expectation value of the perturbation in the unperturbed state.",
        book_refs=["griffiths", "shankar", "sakurai", "cohen-tannoudji"],
        problem_ids=["pt-p1"],
    ),
    Topic(
        id="time-dependent-perturbation",
        title="Time-Dependent Perturbation Theory",
        level="advanced",
        prerequisites=["time-independent-perturbation", "measurement-postulates"],
        key_concepts=["Fermi's golden rule", "transition rate", "selection rules", "absorption and stimulated emission"],
        key_equations=[
            r"\Gamma_{i\to f} = \frac{2\pi}{\hbar}|\langle f|H'|i\rangle|^2\rho(E_f)",
        ],
        intuition="When a perturbation oscillates at just the right frequency, it drives transitions between energy levels. Fermi's golden rule gives the transition rate — the foundation of spectroscopy, laser physics, and quantum optics.",
        book_refs=["griffiths", "sakurai", "cohen-tannoudji"],
        problem_ids=[],
    ),
    Topic(
        id="variational-method",
        title="The Variational Method",
        level="advanced",
        prerequisites=["time-independent-perturbation", "operators-observables"],
        key_concepts=["variational principle", "trial wavefunction", "upper bound on ground state energy", "Hartree-Fock method"],
        key_equations=[r"E_{\text{gs}} \leq \frac{\langle\psi_T|\hat{H}|\psi_T\rangle}{\langle\psi_T|\psi_T\rangle}"],
        intuition="For any trial wavefunction, the expectation value of H is always ≥ the true ground state energy. So we can find approximate ground states by minimizing over a family of trial functions — the basis of all quantum chemistry calculations.",
        book_refs=["griffiths", "shankar", "cohen-tannoudji"],
        problem_ids=[],
    ),
    Topic(
        id="wkb",
        title="WKB Approximation",
        level="advanced",
        prerequisites=["step-barrier-tunneling", "schrodinger-equation"],
        key_concepts=["semiclassical limit", "slowly varying potential", "connection formulas", "tunneling rate", "Bohr-Sommerfeld quantization"],
        key_equations=[
            r"\psi(x) \approx \frac{A}{\sqrt{p(x)}}e^{\pm i\int p(x')\,dx'/\hbar}",
            r"\oint p\,dq = \left(n+\tfrac{1}{2}\right)h",
        ],
        intuition="When the potential varies slowly compared to the de Broglie wavelength, the wavefunction locally looks like a plane wave with a position-dependent wavelength. This semiclassical approximation is exact in the ℏ→0 limit.",
        book_refs=["griffiths", "landau-lifshitz-qm"],
        problem_ids=[],
    ),
    Topic(
        id="identical-particles",
        title="Identical Particles & Symmetrization",
        level="advanced",
        prerequisites=["measurement-postulates", "spin-pauli", "wavefunction-born-rule"],
        key_concepts=["indistinguishable particles", "exchange symmetry", "bosons (symmetric) vs fermions (antisymmetric)", "Pauli exclusion principle", "Slater determinant"],
        key_equations=[
            r"\psi(1,2) = \begin{cases} +\psi(2,1) & \text{bosons} \\ -\psi(2,1) & \text{fermions}\end{cases}",
        ],
        intuition="In QM, identical particles literally have no identity — swapping two electrons gives the same physical state. Fermions (half-integer spin) are antisymmetric under exchange, giving the Pauli exclusion principle that underlies the entire periodic table.",
        book_refs=["griffiths", "sakurai", "cohen-tannoudji", "ballentine"],
        problem_ids=[],
    ),
    Topic(
        id="scattering-theory",
        title="Scattering Theory",
        level="advanced",
        prerequisites=["step-barrier-tunneling", "time-independent-perturbation"],
        key_concepts=["cross section", "Born approximation", "partial waves", "S-matrix", "optical theorem"],
        key_equations=[
            r"\frac{d\sigma}{d\Omega} = |f(\theta)|^2",
            r"f(\theta) \approx -\frac{m}{2\pi\hbar^2}\int e^{i(\mathbf{k}-\mathbf{k}')\cdot\mathbf{r}}V(\mathbf{r})\,d^3r",
        ],
        intuition="Scattering experiments are how we probe quantum systems. The differential cross section |f(θ)|² tells us how particles scatter at each angle. Born approximation works when the potential is weak compared to the incident energy.",
        book_refs=["sakurai", "griffiths", "landau-lifshitz-qm"],
        problem_ids=[],
    ),
    # ─── ADVANCED TOPICS ──────────────────────────────────────────────────────
    Topic(
        id="path-integrals",
        title="Feynman Path Integrals",
        level="advanced_topics",
        prerequisites=["schrodinger-equation", "wkb", "time-dependent-perturbation"],
        key_concepts=["sum over all paths", "action S[x(t)]", "stationary phase = classical path", "propagator", "equivalent to Schrödinger equation"],
        key_equations=[
            r"\langle x_f,t_f|x_i,t_i\rangle = \int \mathcal{D}[x(t)]\,e^{iS[x(t)]/\hbar}",
        ],
        intuition="Feynman's formulation: a quantum particle explores ALL possible paths simultaneously. Each path contributes with amplitude exp(iS/ℏ). Paths near the classical action interfere constructively; wild paths cancel. In the classical limit, only the stationary-action path survives.",
        book_refs=["feynman-hibbs", "sakurai"],
        problem_ids=[],
    ),
    Topic(
        id="relativistic-qm",
        title="Relativistic Quantum Mechanics & the Dirac Equation",
        level="advanced_topics",
        prerequisites=["spin-pauli", "schrodinger-equation"],
        key_concepts=["Dirac equation", "spinors", "antiparticles (positrons)", "fine structure", "Klein-Gordon equation"],
        key_equations=[
            r"(i\gamma^\mu\partial_\mu - mc)\psi = 0",
            r"\gamma^\mu = (\beta, \beta\boldsymbol{\alpha})",
        ],
        intuition="Combining special relativity with QM forced Dirac to write a first-order equation for spin-½ particles. It automatically predicted spin, correctly computed the electron's magnetic moment, and predicted antimatter — one of the greatest achievements in theoretical physics.",
        book_refs=["dirac", "sakurai"],
        problem_ids=[],
    ),
    Topic(
        id="second-quantization",
        title="Second Quantization & QFT Introduction",
        level="advanced_topics",
        prerequisites=["identical-particles", "harmonic-oscillator", "relativistic-qm"],
        key_concepts=["field operators", "creation and annihilation operators for fields", "Fock space", "quantization of EM field", "photons as quanta"],
        key_equations=[
            r"[\hat{a}_k, \hat{a}_{k'}^\dagger] = \delta_{kk'}",
            r"\hat{\phi}(x) = \int \frac{d^3k}{(2\pi)^3}\frac{1}{\sqrt{2\omega_k}}\left(\hat{a}_k e^{ikx} + \hat{a}_k^\dagger e^{-ikx}\right)",
        ],
        intuition="The harmonic oscillator's ladder operators generalize to fields. Instead of one oscillator with levels n=0,1,2,..., you get an oscillator for each mode, and each excitation is a particle. This is the mathematical language of the Standard Model.",
        book_refs=["sakurai", "weinberg-qm"],
        problem_ids=[],
    ),
    Topic(
        id="quantum-information",
        title="Quantum Information & Computing",
        level="advanced_topics",
        prerequisites=["spin-pauli", "measurement-postulates", "identical-particles"],
        key_concepts=["qubits", "quantum gates", "entanglement as resource", "no-cloning theorem", "quantum error correction", "Grover's and Shor's algorithms"],
        key_equations=[
            r"|0\rangle = \begin{pmatrix}1\\0\end{pmatrix},\quad |1\rangle = \begin{pmatrix}0\\1\end{pmatrix}",
            r"H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)",
        ],
        intuition="A qubit is a quantum two-level system. Unlike a classical bit, it can be in a superposition. Entangled qubits exhibit correlations with no classical explanation. Quantum algorithms exploit interference to solve problems faster than any classical computer.",
        book_refs=["nielsen-chuang"],
        problem_ids=["qinfo-p1"],
    ),
    Topic(
        id="decoherence",
        title="Decoherence & Open Quantum Systems",
        level="advanced_topics",
        prerequisites=["measurement-postulates", "identical-particles"],
        key_concepts=["entanglement with environment", "density matrix", "Lindblad equation", "pointer states", "emergence of classicality"],
        key_equations=[
            r"\rho = \sum_k p_k |\psi_k\rangle\langle\psi_k|",
            r"\dot{\rho} = -\frac{i}{\hbar}[H,\rho] + \sum_k\!\left(L_k\rho L_k^\dagger - \tfrac{1}{2}L_k^\dagger L_k\rho - \tfrac{1}{2}\rho L_k^\dagger L_k\right)",
        ],
        intuition="Quantum systems don't live in isolation. When a quantum system entangles with its environment, superpositions appear to 'collapse' from our local perspective — without any mysterious wavefunction collapse. Decoherence explains why we never see cats in superpositions.",
        book_refs=["ballentine", "weinberg-qm", "nielsen-chuang"],
        problem_ids=[],
    ),
    Topic(
        id="quantum-optics",
        title="Quantum Optics",
        level="advanced_topics",
        prerequisites=["second-quantization", "time-dependent-perturbation"],
        key_concepts=["coherent states", "squeezed states", "photon statistics", "cavity QED", "Jaynes-Cummings model"],
        key_equations=[
            r"\hat{a}|\alpha\rangle = \alpha|\alpha\rangle\quad(\text{coherent state})",
            r"g^{(2)}(\tau) = \frac{\langle \hat{a}^\dagger\hat{a}^\dagger\hat{a}\hat{a}\rangle}{\langle\hat{a}^\dagger\hat{a}\rangle^2}",
        ],
        intuition="Quantum optics studies light at the single-photon level. Coherent states (laser light) are the most 'classical' quantum states. Squeezed states have noise below the vacuum level in one quadrature — used in gravitational wave detectors (LIGO).",
        book_refs=["cohen-tannoudji", "sakurai"],
        problem_ids=[],
    ),
]}


def topic(topic_id: str) -> Optional[Topic]:
    """Return a Topic by ID, or None if not found."""
    return TOPICS.get(topic_id)


def topics_at_level(level: str) -> List[Topic]:
    """Return all topics at a given level, in curriculum order."""
    order = {
        "basics": ["wave-particle-duality", "photoelectric-effect", "de-broglie", "double-slit",
                   "blackbody-radiation", "bohr-model", "uncertainty-principle", "wavefunction-born-rule",
                   "schrodinger-equation", "particle-in-a-box"],
        "intermediate": ["step-barrier-tunneling", "finite-well", "harmonic-oscillator",
                         "operators-observables", "commutators", "dirac-notation",
                         "measurement-postulates", "hydrogen-atom"],
        "advanced": ["spin-pauli", "angular-momentum-addition", "time-independent-perturbation",
                     "time-dependent-perturbation", "variational-method", "wkb",
                     "identical-particles", "scattering-theory"],
        "advanced_topics": ["path-integrals", "relativistic-qm", "second-quantization",
                            "quantum-information", "decoherence", "quantum-optics"],
    }
    ids = order.get(level, [])
    return [TOPICS[tid] for tid in ids if tid in TOPICS]


def books_at_level(level: str) -> List[Book]:
    """Return books aimed at a given level."""
    return [b for b in BOOKS.values() if b.level == level]


def books_for_topic(topic_id: str) -> List[Book]:
    """Return books that cover a given topic."""
    t = TOPICS.get(topic_id)
    if not t:
        return []
    return [BOOKS[bid] for bid in t.book_refs if bid in BOOKS]


def learning_path(topic_id: str) -> List[Topic]:
    """Return prerequisite-ordered list of topics ending at topic_id (DAG walk, cycle-safe)."""
    result: List[Topic] = []
    visited: set = set()

    def _visit(tid: str) -> None:
        if tid in visited:
            return
        visited.add(tid)
        t = TOPICS.get(tid)
        if not t:
            return
        for prereq in t.prerequisites:
            _visit(prereq)
        result.append(t)

    _visit(topic_id)
    return result
