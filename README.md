export LITELLM_BASE_URL=https://llai-proxy.llan.ll.mit.edu/v1
export LITELLM_API_KEY=sk-GhojGlZcfXdeDOG7syHKEQ
export AGENT_MODEL=claude-opus-4-8
export AGENT_TLS_VERIFY=false

pip install openai python-docx
# paste your four export lines (LITELLM_BASE_URL etc.)
python -m pipeline.run_pipeline status
