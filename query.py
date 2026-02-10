from rag import ask

while True:
    q = input("Ask (type 'exit' to quit): ")
    if q.lower() == "exit":
        break

    try:
        answer = ask(q)
        print("\nAnswer:\n", answer, "\n")
    except Exception as e:
        print("Error:", e)
