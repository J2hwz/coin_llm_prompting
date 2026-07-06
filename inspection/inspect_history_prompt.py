from pathlib import Path
from coinenv.agents.llm_agent import LLMAgent

agent = LLMAgent(
    model_name="together_ai/openai/gpt-oss-20b",
    template_path=Path("src/coinenv/templates/grid_one_coin_control.j2"),
    track_history=True,
)

# Seed two history entries manually
agent.history = [
    ((2, 3), "UP", 0),
    ((3, 3), "RIGHT", 1),
]

prompt = agent.render_template(grid_state="<GRID>", history=agent.history)
print(prompt)