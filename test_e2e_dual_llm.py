#!/usr/bin/env python
"""
End-to-end test for dual-LLM architecture.
Tests a simple query to verify both LLMs are being used correctly.
"""

import asyncio
from src.agent.graph import DeepResearchAgent

async def test_dual_llm_e2e():
    """Run a simple research task to test dual-LLM functionality."""
    
    print("🧪 End-to-End Dual-LLM Test\n")
    print("=" * 60)
    
    # Initialize agent (will use dual-LLM mode from .env)
    agent = DeepResearchAgent(
        mcp_server_url="http://127.0.0.1:8080/sse",
    )
    
    # Simple test query
    test_query = "What are the latest tech news today?"
    
    print(f"\n📝 Test Query: {test_query}\n")
    print("=" * 60)
    print("\nStarting research...\n")
    
    try:
        # Run research
        result = await agent.research(
            task=test_query,
            context={"max_iterations": 3},  # Keep it short for testing
        )
        
        print("\n" + "=" * 60)
        print("✅ Research Completed!\n")
        
        # Check if markdown report was generated
        if "markdown_report" in result:
            report = result["markdown_report"]
            print(f"📄 Markdown Report Generated: {len(report)} characters\n")
            print("Preview (first 500 chars):")
            print("-" * 60)
            print(report[:500])
            print("-" * 60)
        else:
            print("⚠️ No markdown_report found in result")
            print(f"Available keys: {list(result.keys())}")
        
        # Show output paths if available
        if "output_paths" in result:
            print(f"\n📁 Report saved to: {result['output_paths'].get('report')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_dual_llm_e2e())
    exit(0 if success else 1)
