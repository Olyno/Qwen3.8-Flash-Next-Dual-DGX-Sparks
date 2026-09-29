import json, sys, urllib.request
PORT=int(sys.argv[1]); OUT=sys.argv[2]
PROMPTS=["Explain quicksort in two sentences.","What is 17*23? Show only the number.","Write a haiku about rain.","def fib(n):","Translate to French: The library is closed.","List three prime numbers.","The capital of Japan is","2+2=","Summarize: ATP stores energy in cells.","What color is the sky?"]
with open(OUT,"w") as f:
    for p in PROMPTS:
        body=json.dumps({"model":"qwen3.8-flash-next","messages":[{"role":"user","content":p}],"max_tokens":48,"temperature":0,"logprobs":True,"top_logprobs":1}).encode()
        req=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
        d=json.load(urllib.request.urlopen(req,timeout=180))
        ch=d["choices"][0]
        lp=(ch.get("logprobs") or {}).get("content") or [{}]
        f.write(json.dumps({"prompt":p,"text":ch["message"]["content"],"lp":lp[0].get("logprob")})+"\n")
print("PROBE-OK",OUT)
