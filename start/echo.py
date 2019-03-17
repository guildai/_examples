msg = "Hello Guild!"

print(msg)
with open("message.txt", "w") as out:
    out.write(msg + "\n")
