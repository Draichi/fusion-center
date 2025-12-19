#!/usr/bin/env python
"""
Exemplo correto de uso do Dual-LLM Agent.
"""

import asyncio
from src.agent import DeepResearchAgent


async def main():
    # ✅ CORRETO: Sem passar provider/model/temperature
    # O agent vai ler automaticamente do .env
    agent = DeepResearchAgent()
    
    # Agora execute sua pesquisa
    result = await agent.research(
        task="Sua pergunta aqui",
        context={},
    )
    
    print(result.get("markdown_report", ""))


# ❌ ERRADO: Passando parâmetros de LLM
# agent = DeepResearchAgent(
#     provider="ollama",      # ❌ NÃO faça isso
#     model="qwen2.5:7b",     # ❌ NÃO faça isso  
#     temperature=0.7,        # ❌ NÃO faça isso
# )


if __name__ == "__main__":
    asyncio.run(main())
