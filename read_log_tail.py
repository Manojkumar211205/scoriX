try:
    with open("d:/scoriX_agent/agent.log", "r") as f:
        lines = f.readlines()
        print("".join(lines[-20:]))
except Exception as e:
    print(e)
