#!/usr/bin/env python
"""
Simple verification test for dual-LLM node execution without MCP server.
Tests that nodes correctly route to appropriate LLMs.
"""

import asyncio
from src.agent.graph import get_dual_llms
from src.agent.nodes import PlanningNode, ReflectionNode, SynthesisNode
from src.agent.state import create_initial_state

async def test_node_execution():
    """Test individual node execution with dual LLMs."""
    
    print("🧪 Testing Dual-LLM Node Execution\n")
    print("=" * 60)
    
    # Initialize dual LLMs
    thinking_llm, structured_llm = get_dual_llms()
    
    # Create test state
    state = create_initial_state(
        task="What are the latest developments in AI?",
        context={},
        max_iterations=5
    )
    
    print("\n1️⃣ Testing Structured Node (PlanningNode)")
    print("-" * 60)
    planning_node = PlanningNode(thinking_llm, structured_llm)
    print(f"   Node Type: {planning_node.get_node_type()}")
    print(f"   Expected: structured (uses {structured_llm.__class__.__name__})")
    
    try:
        result = await planning_node.execute(state)
        print(f"   ✅ Planning executed successfully")
        if "research_plan" in result:
            plan = result["research_plan"]
            print(f"   📋 Generated plan with {len(plan.get('objectives', []))} objectives")
    except Exception as e:
        print(f"   ❌ Planning failed: {e}")
    
    print("\n2️⃣ Testing Thinking Node (ReflectionNode)")
    print("-" * 60)
    reflection_node = ReflectionNode(thinking_llm, structured_llm)
    print(f"   Node Type: {reflection_node.get_node_type()}")
    print(f"   Expected: thinking (uses both LLMs)")
    print(f"   - Step 1: {thinking_llm.__class__.__name__} for reasoning")
    print(f"   - Step 2: {structured_llm.__class__.__name__} for JSON formatting")
    
    # Add some mock data for reflection
    state_with_data = {**state, "key_insights": ["AI is advancing rapidly"]}
    
    try:
        result = await reflection_node.execute(state_with_data)
        print(f"   ✅ Reflection executed successfully")
        if "reflection_notes" in result:
            notes = result["reflection_notes"]
            print(f"   🔍 Generated {len(notes)} reflection notes")
    except Exception as e:
        print(f"   ❌ Reflection failed: {e}")
    
    print("\n3️⃣ Testing Thinking Markdown Node (SynthesisNode)")  
    print("-" * 60)
    synthesis_node = SynthesisNode(thinking_llm, structured_llm)
    print(f"   Node Type: {synthesis_node.get_node_type()}")
    print(f"   Expected: thinking_markdown (uses {thinking_llm.__class__.__name__})")
    
    # Add comprehensive mock data for synthesis
    state_for_synthesis = {
        **state,
        "findings": [
            {"content": "AI models are getting larger", "source": "test"},
            {"content": "New breakthroughs in efficiency", "source": "test"}
        ],
        "key_insights": ["AI is advancing", "Efficiency is key"],
        "correlations": [],
        "research_plan": {"objectives": ["Understand AI trends"]},
        "executed_queries": [],
    }
    
    try:
        result = await synthesis_node.execute(state_for_synthesis)
        print(f"   ✅ Synthesis executed successfully")
        if "markdown_report" in result:
            report = result["markdown_report"]
            print(f"   📄 Generated markdown report ({len(report)} chars)")
            print(f"\n   Preview (first 300 chars):")
            print("   " + "-" * 56)
            preview = report[:300].replace("\n", "\n   ")
            print(f"   {preview}")
            print("   " + "-" * 56)
        else:
            print(f"   ⚠️ No markdown_report in result. Keys: {list(result.keys())}")
    except Exception as e:
        print(f"   ❌ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Dual-LLM Node Execution Test Complete!\n")

if __name__ == "__main__":
    asyncio.run(test_node_execution())
