"""Update all node.execute() calls in _run_with_mcp to use error checking wrapper."""

# Read the file
with open('/Users/lucasdraichi/Workspace/fusion-center/src/agent/graph.py', 'r') as f:
    content = f.read()

# Replace pattern for each node execution
# Old: updates = await node.execute(state)\n updates["current_phase"] = node.get_next_phase(state, updates)\n state = {**state, **updates}
# New: state, has_error = await self._execute_node_with_error_check(node, state, "node_name")\n if has_error: break

replacements = [
    # Planning node
    (
        'updates = await planning_node.execute(state)\n                        updates["current_phase"] = planning_node.get_next_phase(state, updates)\n                        state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(planning_node, state, "planning")\n                        if has_error:\n                            break'
    ),
    # Decomposition node
    (
        'updates = await decomposition_node.execute(state)\n                        updates["current_phase"] = decomposition_node.get_next_phase(state, updates)\n                        state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(decomposition_node, state, "decomposition")\n                        if has_error:\n                            break'
    ),
    # Hypothesis node
    (
        'updates = await hypothesis_node.execute(state)\n                        updates["current_phase"] = hypothesis_node.get_next_phase(state, updates)\n                        state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(hypothesis_node, state, "hypothesis")\n                        if has_error:\n                            break'
    ),
    # Hypothesis update node
    (
        'updates = await hypothesis_update_node.execute(state)\n                            updates["current_phase"] = hypothesis_update_node.get_next_phase(state, updates)\n                            state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(hypothesis_update_node, state, "hypothesis_update")\n                            if has_error:\n                                break'
    ),
    # Analysis node
    (
        'updates = await analysis_node.execute(state)\n                    updates["current_phase"] = analysis_node.get_next_phase(state, updates)\n                    state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(analysis_node, state, "analysis")\n                    if has_error:\n                        break'
    ),
    # Reflection node
    (
        'updates = await reflection_node.execute(state)\n                    updates["current_phase"] = reflection_node.get_next_phase(state, updates)\n                    state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(reflection_node, state, "reflection")\n                    if has_error:\n                        break'
    ),
    # Correlation node
    (
        'updates = await correlation_node.execute(state)\n                    updates["current_phase"] = correlation_node.get_next_phase(state, updates)\n                    state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(correlation_node, state, "correlation")\n                    if has_error:\n                        break'
    ),
    # Verification node
    (
        'updates = await verification_node.execute(state)\n                    updates["current_phase"] = verification_node.get_next_phase(state, updates)\n                    state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(verification_node, state, "verification")\n                    if has_error:\n                        break'
    ),
    # Synthesis node
    (
        'updates = await synthesis_node.execute(state)\n                    updates["current_phase"] = synthesis_node.get_next_phase(state, updates)\n                    state = {**state, **updates}',
        'state, has_error = await self._execute_node_with_error_check(synthesis_node, state, "synthesis")\n                    if has_error:\n                        break'
    ),
]

# Apply replacements
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"✅ Replaced {old.split('=')[0].strip()}")
    else:
        print(f"⚠️  Pattern not found (might be different formatting): {old[:50]}...")

# Write back
with open('/Users/lucasdraichi/Workspace/fusion-center/src/agent/graph.py', 'w') as f:
    f.write(content)

print("\n✅ Updated all node.execute() calls to use error checking wrapper")
