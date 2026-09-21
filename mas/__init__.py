"""
mas — the Multi-Agent System layer.

Stock Agent is the PRIMARY orchestrator. This package is how it classifies a
request, plans which specialist sub-agents can answer it, runs them under
isolation, and folds their answers into one synthesis.

Named `mas` rather than `orchestration` because `intelligence/orchestration.py`
already exists and means something else entirely — it is a compute-budget
router that decides which *intelligence sections* to run for one equity. This
package routes across *agents*, and across *asset classes*.

Layout
------
    asset_class.py   what a symbol IS, and therefore what may be computed
    registry.json    the sub-agent roster — DATA, not code
    registry.py      reading and validating that roster
    contract.py      the uniform sub-agent interface every adapter implements
    plan.py          request -> inspectable execution plan (no I/O)
    run.py           plan -> traced execution, isolated per step
    synthesis.py     sub-agent results -> one answer
    agents/          the adapters
    pricing/         the local derivatives engine
"""
