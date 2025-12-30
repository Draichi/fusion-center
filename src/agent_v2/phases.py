"""
Prompt Builders for each research phase.

These functions build the prompts that get sent to each specialized agent.
"""

from src.agent_v2.state import ResearchContext


def build_gather_prompt(ctx: ResearchContext) -> str:
    """Build prompt for gathering phase."""
    queries_info = ""
    if ctx.research_plan:
        queries = ctx.research_plan.get("initial_queries", [])
        if queries:
            queries_info = f"\nPlanned queries to execute: {queries}"
    
    hypotheses_info = ctx.get_hypotheses_summary()
    
    return f"""Gather intelligence for the following research task.

Task: {ctx.task}

Hypotheses to test:
{hypotheses_info}
{queries_info}

## Instructions

Call 3-5 relevant tools to gather evidence, then STOP and provide your summary.

Prioritize:
1. Testing the hypotheses with targeted queries
2. Covering different intelligence domains (news, satellite, cyber, social)
3. Looking for corroborating or contradicting evidence

Current findings count: {len(ctx.findings)}
Iteration: {ctx.iteration}/{ctx.max_iterations}

**After calling tools, return your structured output with tools_called, summary, and data_quality.**
"""


def build_analysis_prompt(ctx: ResearchContext) -> str:
    """Build prompt for analysis phase."""
    findings_summary = ctx.get_findings_summary()
    hypotheses_summary = ctx.get_hypotheses_summary()
    
    return f"""Analyze the collected intelligence findings.

Task: {ctx.task}

Findings collected:
{findings_summary}

Hypotheses being tested:
{hypotheses_summary}

Previous insights: {ctx.key_insights if ctx.key_insights else "None yet"}

Identify key patterns, test hypotheses against evidence, and determine what additional information is needed.
"""


def build_reflection_prompt(ctx: ResearchContext) -> str:
    """Build prompt for reflection phase."""
    return f"""Reflect on the current analysis state.

Task: {ctx.task}

Key insights so far:
{chr(10).join(f'- {i}' for i in ctx.key_insights) if ctx.key_insights else 'None'}

Uncertainties identified:
{chr(10).join(f'- {u}' for u in ctx.uncertainties) if ctx.uncertainties else 'None'}

Hypotheses status:
{ctx.get_hypotheses_summary()}

Findings count: {len(ctx.findings)}
Iteration: {ctx.iteration}/{ctx.max_iterations}

Critically examine for biases, gaps, and alternative explanations.
Determine if more investigation is needed or if we're ready for verification.
"""


def build_verification_prompt(ctx: ResearchContext) -> str:
    """Build prompt for verification phase."""
    insights_list = "\n".join(f"[{i}] {insight}" for i, insight in enumerate(ctx.key_insights))
    
    return f"""Verify the consistency and validity of conclusions.

Task: {ctx.task}

Key insights to verify:
{insights_list}

Correlations found: {len(ctx.correlations)}

For each insight, verify:
1. Is it supported by actual evidence?
2. Are there contradictions?
3. Is the confidence level appropriate?

Flag any issues that need addressing.
"""


def build_synthesis_prompt(ctx: ResearchContext) -> str:
    """Build prompt for final synthesis."""
    return f"""Synthesize a complete SITREP intelligence report.

## User Query
{ctx.task}

## Verified Insights
{chr(10).join(f'- {i}' for i in ctx.verified_insights) if ctx.verified_insights else 'No verified insights'}

## All Key Insights (for reference)
{chr(10).join(f'- {i}' for i in ctx.key_insights) if ctx.key_insights else 'None'}

## Correlations Found
{len(ctx.correlations)} correlations identified

## Hypotheses Final Status
{ctx.get_hypotheses_summary()}

## Reflection Notes
{len(ctx.reflection_notes)} notes recorded

## Research Metadata
- Task complexity: {ctx.task_complexity}
- Total findings: {len(ctx.findings)}
- Iterations completed: {ctx.iteration}

Create a comprehensive SITREP with all 6 sections. Cite sources for every claim.
"""
