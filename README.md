curl -sk "$LITELLM_BASE_URL/responses" -H "Authorization: Bearer $LITELLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.6-sol","input":"Reply with the single word: ready","reasoning":{"effort":"high"}}'
