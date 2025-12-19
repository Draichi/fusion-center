"""Add error handling wrapper to graph.py."""

# Read the file
with open('/Users/lucasdraichi/Workspace/fusion-center/src/agent/graph.py', 'r') as f:
    lines = f.readlines()

# Find where to insert the helper method (before _run_with_mcp)
insert_index = None
for i, line in enumerate(lines):
    if 'async def _run_with_mcp(' in line:
        insert_index = i
        break

if insert_index is None:
    print("ERROR: Could not find _run_with_mcp")
    exit(1)

# Add the helper method
helper_lines = [
    '    async def _execute_node_with_error_check(self, node, state, node_name):\n',
    '        """Helper to execute node and check for errors."""\n',
    '        try:\n',
    '            updates = await node.execute(state)\n',
    '            \n',
    '            # Check if node returned an error\n',
    '            if "error" in updates:\n',
    '                logger.error(f"❌ Error in {node_name} phase: {updates[\'error\']}")\n',
    '                # Force completion with error\n',
    '                return {**state, **updates, "current_phase": ResearchPhase.COMPLETE.value}, True\n',
    '            \n',
    '            # Get next phase\n',
    '            next_phase = node.get_next_phase(state, updates)\n',
    '            updates["current_phase"] = next_phase\n',
    '            \n',
    '            return {**state, **updates}, False\n',
    '            \n',
    '        except Exception as e:\n',
    '            logger.error(f"❌ Unhandled exception in {node_name}: {e}")\n',
    '            import traceback\n',
    '            logger.error(traceback.format_exc())\n',
    '            error_state = {\n',
    '                **state,\n',
    '                "error": f"{type(e).__name__}: {str(e)}",\n',
    '                "current_phase": ResearchPhase.COMPLETE.value,\n',
    '                "iteration": state["iteration"] + 1,\n',
    '            }\n',
    '            return error_state, True\n',
    '\n',
]

# Insert the helper
lines[insert_index:insert_index] = helper_lines

# Write back
with open('/Users/lucasdraichi/Workspace/fusion-center/src/agent/graph.py', 'w') as f:
    f.writelines(lines)

print("✅ Added _execute_node_with_error_check helper method")
