import requests
import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from datetime import datetime
class QueryEngine:
    def __init__(self, openAiClient, search_client, bing_access_key, custom_config_id, google_access_key, google_search_engine_id):
        self.search_client = search_client
        self.openAiClient = openAiClient
        self.bing_access_key = bing_access_key
        self.google_access_key = google_access_key
        self.google_search_engine_id = google_search_engine_id
        self.custom_config_id = custom_config_id

    def parse_google_results(self, results):
        parsed_results = []
        for item in results.get("items", []):
            parsed_results.append({
                "title": item['title'],
                "snippet": item['snippet'],
                "url": item['link']
            })
        return parsed_results
    
    def parse_bing_results(self, results):
        parsed_results = []
        for item in results.get('webPages', {}).get('value', []):
            parsed_results.append({
                "title": item['name'],
                "snippet": item['snippet'],
                "url": item['url']
            })
        return parsed_results

    def refine_query(self, messages):
        filteredMessages = list(filter(lambda obj: obj["role"] != 'system', messages))
        formattedToday = datetime.today().strftime('%d.%m.%Y')
        completion = self.openAiClient.chat.completions.create(
            model="gpt-4o", 
            temperature=0.1,
            top_p=0.95,
            max_tokens=800,

            messages = [
                {"role": "system", "content": f'''Analyze the user's latest message in the context of the conversation history. Identify the key information or action the user is requesting. Based on this, generate a clear and concise search query suitable for Bing to retrieve the relevant information. The query should include specific names, roles, locations, or any other details mentioned by the user, ensuring accuracy and relevance.  Replace any time expressions (e.g., "today," "this weekend") with the corresponding actual dates in the format dd.mm.yyyy, today is {formattedToday}. The weekend should be returned in the format dd-dd.mm.yyyy.  Return the plain query, no additional information. '''},  
                {"role": "user", "content": json.dumps(filteredMessages)}
        ]
        )

        return  completion.choices[0].message.content
    def search_google(self, query):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_access_key,
            "cx": self.google_search_engine_id,
            "q": query,
            "num": 5,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results =  response.json()
            return self.parse_google_results(results)
        elif response.status_code in {403}:
            raise RuntimeError(f"Chyba Google API. Bylo dosaženo limitu pro dnešní den. Zkuste to zítra.")
        elif response.status_code in {400}:
            raise RuntimeError(f"Google API Error: Received status code 400.")
        else:
            response.raise_for_status()
    
    def search_bing(self, query):
        endpoint = "https://api.bing.microsoft.com/v7.0/custom/search"

        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_access_key
        }
        params = {
            "q": query,
            "customconfig": self.custom_config_id
        }

        response = requests.get(endpoint, headers=headers, params=params)
        
        if response.status_code == 200:
            result =  response.json()
            return self.parse_bing_results(result)
        else:
            raise Exception(f"Bing API error: {response.status_code}")
    
    def generate_embedding(self, user_query):
        response = self.openAiClient.embeddings.create(
            input=user_query,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def search_index(self, user_query):
        query_vector = self.generate_embedding(user_query)
        search_results = self.search_client.search(
                                        search_text=user_query, 
                                        top=5,  
                                        vector_fields="contentVector",
                                        vector=query_vector,
                                        select = ['url', 'id', 'filepath', 'content']
                                        )
        result = []
        for doc in search_results:
            result.append({"content": doc["content"], "id": doc["id"] })
        return result
