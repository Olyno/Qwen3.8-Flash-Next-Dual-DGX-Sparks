import json, sys, urllib.request
PORT=int(sys.argv[1]); OUT=sys.argv[2]
# v2: the v1 probe was confounded. max_tokens=48 on a thinking model means
# content=None rows were REASONING-BUDGET EXHAUSTION (not engine), and short
# content cuts made arithmetic answers look divergent ("3" vs "391" = trunc,
# not corruption). v2 gives 1024 total, records finish_reason + reasoning
# length separately, and uses prompts where a GEMM bit-flip changes a printed
# token (exact arithmetic, code echo, deterministic copy).
PROMPTS=[
 "What is 17*23? Answer with only the number.",
 "What is 97*89? Answer with only the number.",
 "What is 2^16? Answer with only the number.",
 "Complete the sequence with only the number: 2,3,5,7,11,13,",
 "Repeat exactly, nothing else: the quick brown fox jumps over the lazy dog",
 "def greet(name):\n    return",
 "print(sum(range(101)))  # output only the number",
 "The capital of Australia is (one word):",
 "Convert 0xFF to decimal, number only:",
 "13th letter of the alphabet, letter only:",
 "What is 100/7 rounded to 4 decimals, number only:",
 "Translate to French only: good morning",
 "Balance: 3+5*2 equals? number only:",
 "SHA1 of 'abc' starts with (first 8 hex):",
 "List numbers 1-15 separated by commas, nothing else.",
 "9*8*7 equals? number only:",
 "Binary of 42, digits only:",
 "Next leap year after 2024, year only:",
 "Complete the Python line: for i in range(3):",
 "How many letters in 'supercalifragilisticexpialidocious'? number only.",
 "Sort descending, comma-only: 7 3 9 1 5",
 "What is 12345 + 6789? number only:",
 "Repeat only: to be or not to be",
 "The word 'banana' reversed, word only:",
 "Fib(20) = ? number only:",
]
with open(OUT,"w") as f:
    for p in PROMPTS:
        body=json.dumps({"model":"qwen3.8-flash-next","messages":[{"role":"user","content":p}],"max_tokens":1024,"temperature":0,"logprobs":True,"top_logprobs":1}).encode()
        req=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
        try:
            d=json.load(urllib.request.urlopen(req,timeout=300))
        except Exception as e:
            f.write(json.dumps({"prompt":p,"err":repr(e)[:120]})+"\n"); continue
        ch=d["choices"][0]
        m=ch["message"]
        lp=(ch.get("logprobs") or {}).get("content") or [{}]
        f.write(json.dumps({"prompt":p,"text":m.get("content"),"reasoning_len":len(m.get("reasoning_content") or ""),"finish":ch.get("finish_reason"),"first_lp":lp[0].get("logprob"),"tok_lp_sum":round(sum(t.get("logprob") or 0 for t in (ch.get("logprobs") or {}).get("content") or []),4)})+"\n")
print("PROBE2-OK",OUT)
