# Research Algorithm - Deep Technical Specification

This document describes the step-by-step research process used by the Deep Research Agent. It details how each node in the LangGraph processes data and which prompts guide the LLM's reasoning.

---

## Graph Flow Overview

```
START → PLANNING → DECOMPOSING → HYPOTHESIZING → GATHERING ⟷ ANALYZING → REFLECTING → CORRELATING → VERIFYING → SYNTHESIZING → END
                                                    ↑______________|         ↑_________|
```

The router function (`route_next_step`) determines transitions based on `current_phase` in `AgentState`.

---

## Phase 1: PLANNING

**Node:** `plan_research(state, llm)`  
**Output:** `research_plan`, `pending_queries`, transitions to `DECOMPOSING`

### Prompt Composition

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"""
        {PLANNER_PROMPT}

        ## Task
        {state["task"]}

        ## Context
        {json.dumps(state["context"])}

        ## Available Tools
        {json.dumps(get_tool_definitions())}

        Create a research plan as JSON...
    """),
]
```

### SYSTEM_PROMPT Role

Establishes the agent as an OSINT analyst with strict data integrity rules:

```python
SYSTEM_PROMPT = """You are an expert OSINT analyst working for Project Overwatch...

## CRITICAL: Data Integrity Rules
**NEVER FABRICATE OR HALLUCINATE DATA.**
1. Only report data that was ACTUALLY returned by tools
2. If a tool fails, explicitly state it failed
3. If no data is available, say "No data available"
..."""
```

### PLANNER_PROMPT Role

Guides the LLM to create a structured research plan:

```python
PLANNER_PROMPT = """You are planning a deep research investigation...

Your job is to create a comprehensive research plan that:
1. Breaks down the task into specific investigative objectives
2. Identifies relevant geographic regions and coordinates
3. Determines keywords for news searches
4. Prioritizes which intelligence sources to query first
5. Plans initial queries to execute
..."""
```

### Expected Output Structure

```json
{
    "objectives": ["Identify military activity patterns", "Track infrastructure damage"],
    "regions_of_interest": ["Ukraine", "Kharkiv Oblast"],
    "keywords": ["military", "strike", "explosion"],
    "coordinates": [{"lat": 49.99, "lon": 36.23, "radius_km": 50}],
    "time_range": "7d",
    "priority_sources": ["news", "satellite", "cyber"],
    "initial_queries": [
        {"tool": "search_news", "args": {"keywords": "military strike", "country_code": "UA"}}
    ]
}
```

---

## Phase 2: DECOMPOSING

**Node:** `decompose_task(state, llm)`  
**Output:** `sub_tasks`, `task_complexity`, transitions to `HYPOTHESIZING`

### Prompt Composition

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"""
        {TASK_DECOMPOSITION_PROMPT}

        ## Research Task
        {state["task"]}

        ## Context
        {json.dumps(state["context"])}

        Analyze this task and decompose it into sub-tasks.
    """),
]
```

### TASK_DECOMPOSITION_PROMPT Role

Breaks complex tasks into 2-5 focused sub-tasks:

```python
TASK_DECOMPOSITION_PROMPT = """You are decomposing a complex research task...

For each sub-task:
1. Give it a unique ID (e.g., "subtask_1")
2. Write a clear, focused description
3. Identify any dependencies
4. Estimate the complexity (simple, moderate, complex)

Consider: geographic aspects, temporal aspects, actor/entity aspects, thematic aspects
..."""
```

### Expected Output Structure

```json
{
    "task_complexity": "moderate",
    "decomposition_reasoning": "Task involves multiple geographic regions and temporal analysis",
    "sub_tasks": [
        {
            "id": "subtask_1",
            "description": "Identify thermal anomalies in Kharkiv region",
            "focus_area": "geographic",
            "dependencies": [],
            "complexity": "simple"
        },
        {
            "id": "subtask_2",
            "description": "Correlate thermal events with news reports",
            "focus_area": "thematic",
            "dependencies": ["subtask_1"],
            "complexity": "moderate"
        }
    ]
}
```

---

## Phase 3: HYPOTHESIZING

**Node:** `generate_hypotheses(state, llm)`  
**Output:** `hypotheses`, `pending_queries` (test queries), transitions to `GATHERING`

### Prompt Composition

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"""
        {HYPOTHESIS_GENERATION_PROMPT}

        ## Research Task
        {state["task"]}

        ## Context
        {json.dumps(state["context"])}

        ## Sub-tasks (if decomposed)
        {json.dumps(state.get("sub_tasks", []))}

        ## Existing Findings
        {json.dumps(state.get("findings", [])[:10])}

        ## Available Tools for Testing Hypotheses
        {json.dumps(get_tool_definitions())}

        Generate hypotheses that can be tested with these tools.
    """),
]
```

### HYPOTHESIS_GENERATION_PROMPT Role

Forms testable hypotheses using Chain of Thought:

```python
HYPOTHESIS_GENERATION_PROMPT = """You are forming hypotheses to guide the research...

## Chain of Thought Process
Before generating hypotheses, think through:
1. What do we already know?
2. What are the possible explanations?
3. What would we expect to see if each explanation is true?
4. Which explanations are most likely given the context?

For each hypothesis:
1. State it clearly and specifically
2. Explain what evidence would SUPPORT it
3. Explain what evidence would REFUTE it
4. Assign an initial confidence (0.0 to 1.0)
5. Suggest specific queries to test it
..."""
```

### Expected Output Structure

```json
{
    "reasoning_chain": [
        "First, I observe that the task mentions military activity in eastern Ukraine",
        "This suggests possible ongoing conflict escalation",
        "An alternative explanation could be routine exercises",
        "To distinguish, I would look for thermal anomalies and civilian impact reports"
    ],
    "hypotheses": [
        {
            "id": "h1",
            "statement": "Military strikes have increased in Kharkiv region in the past 7 days",
            "supporting_evidence_criteria": ["Multiple thermal anomalies detected", "News reports of explosions"],
            "refuting_evidence_criteria": ["No thermal anomalies", "Reports indicate ceasefire"],
            "initial_confidence": 0.6,
            "test_queries": [
                {"tool": "detect_thermal_anomalies", "args": {"latitude": 49.99, "longitude": 36.23, "radius_km": 50}, "reason": "Check for fire/explosion signatures"}
            ]
        }
    ]
}
```

---

## Phase 4: GATHERING

**Node:** `gather_intelligence(state, tool_executor)`  
**Output:** `findings`, `executed_queries`, transitions to `ANALYZING` (or stays in `GATHERING` if more queries)

### Process

1. Executes up to 5 queries per iteration from `pending_queries`
2. Calls MCP tools via `tool_executor.execute(tool_name, args)`
3. Extracts structured findings from tool results using `_extract_findings()`

### Finding Extraction Logic

```python
def _extract_findings(tool_name, args, result):
    findings = []
    
    if tool_name == "search_news":
        for article in result.get("articles", [])[:10]:
            findings.append({
                "source": article.get("domain"),
                "source_type": IntelligenceType.NEWS.value,
                "timestamp": datetime.utcnow().isoformat(),
                "content": {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "date": article.get("seendate"),
                },
                "relevance_score": 0.8,
                "confidence": "medium",
            })
    
    elif tool_name == "detect_thermal_anomalies":
        for anomaly in result.get("anomalies", [])[:20]:
            findings.append({
                "source": "NASA FIRMS",
                "source_type": IntelligenceType.SATELLITE.value,
                "content": {
                    "brightness": anomaly.get("brightness"),
                    "confidence": anomaly.get("confidence"),
                },
                "location": {
                    "lat": anomaly.get("latitude"),
                    "lon": anomaly.get("longitude"),
                },
            })
    # ... similar for cyber, sanctions
```

### Hypothesis Update (Post-Gathering)

After gathering, if hypotheses exist, `update_hypotheses()` is called:

```python
if state.get("hypotheses"):
    updates = await update_hypotheses(state, llm)
```

**HYPOTHESIS_UPDATE_PROMPT** uses Bayesian reasoning:

```python
HYPOTHESIS_UPDATE_PROMPT = """You are updating hypotheses based on new evidence.

## Bayesian Reasoning
1. Prior: What was your confidence before this evidence?
2. Likelihood: How likely is this evidence if the hypothesis is TRUE?
3. Base rate: How likely is this evidence in general?
4. Posterior: Updated confidence after seeing the evidence
..."""
```

---

## Phase 5: ANALYZING

**Node:** `analyze_findings(state, llm)`  
**Output:** `key_insights`, `correlations`, `uncertainties`, `pending_queries` (follow-ups), transitions to `REFLECTING`

### Prompt Composition

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"""
        {ENHANCED_ANALYST_PROMPT}

        ## Original Task
        {state["task"]}

        ## Current Hypotheses
        {json.dumps(state.get("hypotheses", []))}

        ## Collected Findings ({len(findings)} total)
        {json.dumps(findings[:30])}

        ## Available Tools for Follow-up Queries
        {json.dumps(tool_names)}

        ## Instructions
        Think step by step:
        1. First, survey all the evidence
        2. What patterns emerge from individual sources?
        3. How does this evidence relate to our hypotheses?
        4. What areas still need investigation?
        ...
    """),
]
```

### ENHANCED_ANALYST_PROMPT Role

Combines Chain-of-Thought wrapper with analysis instructions:

```python
ENHANCED_ANALYST_PROMPT = f"""You are analyzing collected intelligence...

{CHAIN_OF_THOUGHT_WRAPPER}

## Multi-Step Analysis Process
1. Survey the evidence: What types of data do we have?
2. Pattern recognition: What patterns emerge?
3. Cross-source analysis: How do patterns compare across sources?
4. Hypothesis testing: How does this evidence relate to hypotheses?
5. Gap identification: What do we still need to know?
..."""
```

### CHAIN_OF_THOUGHT_WRAPPER

```python
CHAIN_OF_THOUGHT_WRAPPER = """
## Think Step by Step

Before providing your final answer, work through the problem:
1. Understand: What exactly is being asked?
2. Gather: What information do we have?
3. Analyze: What patterns or connections exist?
4. Evaluate: What are the strengths/weaknesses?
5. Conclude: What's the most supported conclusion?

Show your reasoning in a "thinking" section:
```thinking
[Your step-by-step reasoning here]
```

Then provide your final JSON response.
"""
```

### Expected Output Structure

```json
{
    "thinking": "First, I observe 15 news articles about explosions. Thermal data shows 8 anomalies...",
    "key_insights": [
        "Thermal anomalies cluster around industrial areas in Kharkiv",
        "News reports correlate temporally with FIRMS detections"
    ],
    "hypothesis_implications": [
        {"hypothesis_id": "h1", "evidence_type": "supporting", "explanation": "Multiple thermal anomalies detected..."}
    ],
    "correlations": [
        {
            "finding_ids": [0, 5, 12],
            "correlation_type": "temporal",
            "description": "News report at 14:00 UTC matches thermal anomaly at 14:05 UTC",
            "confidence": "high",
            "implications": ["Confirms active military engagement"]
        }
    ],
    "uncertainties": ["Unable to determine if fires are military or industrial"],
    "follow_up_queries": [
        {"tool": "check_connectivity", "args": {"country_code": "UA"}, "reason": "Check for infrastructure disruption"}
    ]
}
```

---

## Phase 6: REFLECTING

**Node:** `reflect_on_analysis(state, llm)`  
**Output:** `reflection_notes`, `needs_more_reflection`, may trigger additional `GATHERING`

### Prompt Composition

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"""
        {REFLECTION_PROMPT}

        ## Original Task
        {state["task"]}

        ## Current Hypotheses
        {json.dumps(state.get("hypotheses", []))}

        ## Key Insights So Far
        {json.dumps(state.get("key_insights", []))}

        ## Correlations Found
        {json.dumps(state.get("correlations", []))}

        ## Uncertainties Identified
        {json.dumps(state.get("uncertainties", []))}

        Critically reflect on this analysis.
    """),
]
```

### REFLECTION_PROMPT Role

Identifies biases, gaps, and alternative explanations:

```python
REFLECTION_PROMPT = """You are performing critical self-reflection...

## Why Self-Reflection Matters
1. Confirmation bias: Did we favor evidence that supports our preferred explanation?
2. Availability bias: Did we overweight recent or memorable findings?
3. Anchoring: Did early findings unduly influence later analysis?
4. Gaps: What did we NOT look for that we should have?

## Reflection Categories
1. Bias Check: Are our conclusions potentially biased?
2. Gap Analysis: What information are we missing?
3. Alternative Explanations: What other explanations haven't we considered?
4. Confidence Calibration: Are our confidence levels appropriate?
5. Consistency Check: Do our conclusions contradict each other?
..."""
```

### Routing Logic

```python
if needs_investigation and investigation_queries and reflection_iterations < 2:
    next_phase = GATHERING  # Go back to gather more data
elif any_critical_action_required:
    next_phase = ANALYZING  # Re-analyze with new perspective
else:
    next_phase = CORRELATING  # Proceed to correlation
```

---

## Phase 7: CORRELATING

**Node:** `correlate_findings(state, llm)`  
**Output:** `correlations`, transitions to `VERIFYING`

### Prompt (Inline)

This phase uses an inline prompt (not a separate constant):

```python
correlation_prompt = f"""
You are correlating intelligence from multiple sources to find connections.

## Findings by Source Type
{json.dumps(by_type)}  # Findings grouped by source_type

## Task Context
{state["task"]}

Look for:
1. **Temporal correlations**: Events happening around the same time
2. **Geospatial correlations**: Events in the same location from different sources
3. **Causal correlations**: One event potentially causing another
4. **Pattern correlations**: Similar patterns across different data types

For each correlation found, explain the connection and its implications.
"""
```

### Finding Grouping

```python
by_type = {}
for i, f in enumerate(findings):
    ftype = f.get("source_type", "unknown")
    if ftype not in by_type:
        by_type[ftype] = []
    by_type[ftype].append({"index": i, **f})
```

---

## Phase 8: VERIFYING

**Node:** `verify_conclusions(state, llm)`  
**Output:** `verification_results`, `verified_insights`, `verified_correlations`, transitions to `SYNTHESIZING`

### Prompt Composition

```python
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"""
        {VERIFICATION_PROMPT}

        ## Original Task
        {state["task"]}

        ## Findings (Evidence Base)
        {json.dumps(state.get("findings", [])[:30])}

        ## Hypotheses and Their Status
        {json.dumps(state.get("hypotheses", []))}

        ## Key Insights to Verify
        {json.dumps(state.get("key_insights", []))}

        ## Correlations to Verify
        {json.dumps(state.get("correlations", []))}

        Verify each conclusion against the evidence.
    """),
]
```

### VERIFICATION_PROMPT Role

Performs final consistency checks:

```python
VERIFICATION_PROMPT = """You are verifying conclusions before final synthesis.

## Verification Checks
1. Internal Consistency: Do conclusions contradict each other?
2. Evidence Support: Is each conclusion adequately supported?
3. Temporal Logic: Do the timelines make sense?
4. Geospatial Logic: Do the locations and distances make sense?
5. Causal Logic: Are causal claims reasonable?

## Verification Standards
- PASS: Well-supported, consistent, confidence appropriate
- ADJUST: Valid but confidence should be adjusted
- FLAG: Has issues that need addressing
- REJECT: Not supported or contradicted by evidence
..."""
```

### Routing Logic

```python
if overall.get("ready_for_synthesis", True):
    next_phase = SYNTHESIZING
else:
    next_phase = REFLECTING  # Go back for another reflection pass
```

---

## Phase 9: SYNTHESIZING

**Node:** `synthesize_report(state, llm)`  
**Output:** `executive_summary`, `detailed_report`, `recommendations`, transitions to `COMPLETE`

### Prompt Composition

```python
# Use verified data if available
insights_to_use = state.get("verified_insights") or state.get("key_insights", [])
correlations_to_use = state.get("verified_correlations") or state.get("correlations", [])

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=f"""
        {ENHANCED_SYNTHESIZER_PROMPT}

        ## Original Task
        {state["task"]}

        ## Research Plan
        {json.dumps(state.get("research_plan", {}))}

        ## Hypothesis Results
        {hypotheses_summary}

        ## Verified Key Insights
        {json.dumps(insights_to_use)}

        ## Verified Correlations
        {json.dumps(correlations_to_use)}

        ## Uncertainties
        {json.dumps(state.get("uncertainties", []))}

        ## Reasoning Process Summary
        - Reasoning depth: {state.get("reasoning_depth", 0)} steps
        - Reflection iterations: {state.get("reflection_iterations", 0)}
        - Verification results: {len(state.get("verification_results", []))} items

        Create a comprehensive intelligence report as JSON...
    """),
]
```

### ENHANCED_SYNTHESIZER_PROMPT Role

```python
ENHANCED_SYNTHESIZER_PROMPT = """You are synthesizing a final report from verified findings.

## CRITICAL: Report Only Verified Data
- Use verified_insights and verified_correlations as primary sources
- Note any conclusions that were flagged during verification
- Be transparent about confidence adjustments

## Multi-Step Synthesis Process
1. Structure the narrative: What's the most logical order?
2. Weigh the evidence: What's strongly supported vs. tentative?
3. Address uncertainties: What don't we know and why does it matter?
4. Draw conclusions: What can we confidently conclude?
5. Make recommendations: What actions or further research is suggested?
..."""
```

### Final Output Structure

```json
{
    "executive_summary": "Military activity in Kharkiv region increased significantly over the past 7 days, with 8 thermal anomalies detected correlating with news reports of strikes.",
    "detailed_report": "## Overview\n\nThis analysis examined...\n\n## Key Findings\n\n1. **Thermal Anomalies**...",
    "recommendations": [
        "Continue monitoring Kharkiv region with daily FIRMS queries",
        "Expand search to adjacent oblasts"
    ],
    "confidence_assessment": "Medium-High confidence. 3 of 4 hypotheses supported by evidence. Limited by GDELT API rate limits.",
    "methodology_note": "Analysis used 5-step multi-step reasoning with 2 reflection iterations and 12 verification checks."
}
```

---

## State Management

### Key State Fields

| Field | Type | Description |
|-------|------|-------------|
| `current_phase` | `str` | Current `ResearchPhase` value |
| `iteration` | `int` | Iteration counter (max 10 default) |
| `hypotheses` | `list[dict]` | Active hypotheses with confidence scores |
| `findings` | `list[dict]` | Collected intelligence findings |
| `reasoning_trace` | `list[dict]` | Step-by-step reasoning log |
| `chain_of_thought` | `list[str]` | LLM thinking blocks |
| `reflection_notes` | `list[dict]` | Self-critique notes |
| `verification_results` | `list[dict]` | Verification outcomes |
| `verified_insights` | `list[str]` | Insights that passed verification |

### Reasoning Step Structure

Each reasoning step logged to `reasoning_trace`:

```python
{
    "step_number": 3,
    "phase": "analyzing",
    "thought": "Analyzing patterns in collected evidence",
    "action": "Identified 5 insights",
    "observation": "Found 3 correlations, 2 uncertainties",
    "conclusion": "Analysis complete, proceeding to reflection",
    "confidence": 0.75,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Iteration Control

The `route_next_step` function enforces safety limits:

```python
def route_next_step(state):
    # Safety check
    if state["iteration"] >= state["max_iterations"]:
        logger.warning("Max iterations reached, forcing synthesis")
        return "synthesize"
    
    # Check for errors
    if state.get("error"):
        return "end"
    
    # Route based on current phase
    phase = state.get("current_phase")
    return phase_to_node_mapping[phase]
```

### Possible Loops

1. **GATHERING ⟷ ANALYZING**: Follow-up queries trigger new gathering
2. **REFLECTING → GATHERING**: Reflection identifies gaps requiring more data
3. **VERIFYING → REFLECTING**: Verification failures require re-analysis

---

## Output Files

The `OutputWriter` saves two files per run:

1. **`report.md`**: Final intelligence report in markdown
2. **`reasoning.log`**: JSON-structured reasoning trace

Example `reasoning.log` entry:

```json
{
    "timestamp": "2024-01-15T10:30:00Z",
    "phase": "analyzing",
    "step": 5,
    "action": "Chain of thought reasoning",
    "details": {
        "thinking": "First, I observe 15 news articles..."
    }
}
```

