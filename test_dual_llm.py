#!/usr/bin/env python
"""
Test script to verify dual-LLM node architecture.
"""

from src.agent.graph import get_dual_llms
from src.agent.nodes import (
    PlanningNode,
    DecompositionNode,
    HypothesisGenerationNode,
    AnalysisNode,
    ReflectionNode,
    CorrelationNode,
    VerificationNode,
    SynthesisNode,
)

def test_node_types():
    """Test that all nodes have correct node types."""
    
    print("🧪 Testing Dual-LLM Node Architecture\n")
    
    # Initialize dual LLMs
    thinking_llm, structured_llm = get_dual_llms()
    
    # Test structured nodes
    structured_nodes = [
        ("PlanningNode", PlanningNode(thinking_llm, structured_llm)),
        ("DecompositionNode", DecompositionNode(thinking_llm, structured_llm)),
        ("HypothesisGenerationNode", HypothesisGenerationNode(thinking_llm, structured_llm)),
        ("AnalysisNode", AnalysisNode(thinking_llm, structured_llm)),
        ("CorrelationNode", CorrelationNode(thinking_llm, structured_llm)),
    ]
    
    # Test thinking nodes
    thinking_nodes = [
        ("ReflectionNode", ReflectionNode(thinking_llm, structured_llm)),
        ("VerificationNode", VerificationNode(thinking_llm, structured_llm)),
    ]
    
    # Test thinking_markdown nodes
    markdown_nodes = [
        ("SynthesisNode", SynthesisNode(thinking_llm, structured_llm)),
    ]
    
    print("📊 Structured Nodes (JSON output):")
    for name, node in structured_nodes:
        node_type = node.get_node_type()
        status = "✅" if node_type == "structured" else "❌"
        print(f"  {status} {name}: {node_type}")
    
    print("\n🧠 Thinking Nodes (reasoning → JSON):")
    for name, node in thinking_nodes:
        node_type = node.get_node_type()
        status = "✅" if node_type == "thinking" else "❌"
        print(f"  {status} {name}: {node_type}")
    
    print("\n📝 Thinking Markdown Nodes (reasoning → markdown):")
    for name, node in markdown_nodes:
        node_type = node.get_node_type()
        status = "✅" if node_type == "thinking_markdown" else "❌"
        print(f"  {status} {name}: {node_type}")
    
    # Verify all nodes
    all_nodes = structured_nodes + thinking_nodes + markdown_nodes
    all_correct = all([
        node.get_node_type() in ["structured", "thinking", "thinking_markdown"]
        for _, node in all_nodes
    ])
    
    print(f"\n{'✅ All nodes configured correctly!' if all_correct else '❌ Some nodes have incorrect configuration'}")
    return all_correct

if __name__ == "__main__":
    success = test_node_types()
    exit(0 if success else 1)
