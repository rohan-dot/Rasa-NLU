Add a --research-model MODEL command-line argument to microagent_team.py. Behavior:
When set, only the research engineer role uses that model. Every other role (manager, the implementation/coding specialists, reviewer) keeps using --model.
When not set, it defaults to the value of --model, so existing behavior is unchanged.
Use the same LLM client code, base URL, endpoint (/v1/responses), and API key for both — only the model field in the request differs per role.
Keep everything else identical. Show me the full updated argparse section and the one place where the research role picks its model.
