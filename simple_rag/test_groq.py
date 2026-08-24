from groq import Groq
from simple_rag.config import settings

client = Groq(api_key=settings.groq_api_key)

models = client.models.list()

for model in models.data:
    print(model.id)