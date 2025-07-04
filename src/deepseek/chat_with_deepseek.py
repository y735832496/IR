from openai import OpenAI

client = OpenAI(api_key="sk-498e5c6d3af8478ca19cfa4f2a4047ed", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "redis的跳表是什么"},
    ],
    stream=False
)

print(response.choices[0].message.content)