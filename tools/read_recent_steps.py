import json

log_file = r"C:\Users\chris\.gemini\antigravity-ide\brain\6151d60b-153f-4758-b3cf-e380fbe5f7a2\.system_generated\logs\transcript.jsonl"
with open(log_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for idx in range(1834, 1838):
    data = json.loads(lines[idx])
    print(f"=== STEP {data.get('step_index')} | {data.get('source')} | {data.get('type')} ===")
    print(str(data.get('content')))
