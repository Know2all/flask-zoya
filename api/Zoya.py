import json,os,pickle
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings


class ZoyaChatbot:
    def __init__(self, api_key: str, project_id: str, dataset_path: str, vector_db_path: str,memory_dir :str):
        self.api_key = api_key
        self.project_id = project_id
        self.dataset_path = dataset_path
        self.vector_db_path = vector_db_path
        self.memory_dir = memory_dir

        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.qa_chains = {}  # user_id -> chain
        self.memories = {}

        self._init_llm()
        # self._load_documents()
        self._init_embeddings()
        self._init_vector_store()
        self._init_prompt()

        os.makedirs(self.memory_dir, exist_ok=True)

    def _init_llm(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=self.api_key,
            temperature=0,
            max_tokens=500,
            top_k=50,
        )

    def _load_documents(self):
        self.docs = []
        with open(self.dataset_path, "r", encoding="utf-8") as file:
            chat_data = json.load(file)
            for item in tqdm(chat_data, desc="Loading Chats", unit="message"):
                prefix = "You" if item["role"] == "You" else "Zoya"
                text = f"{prefix}: {item['message']} \n"
                self.docs.append(Document(page_content=text))

    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _init_vector_store(self):
        if os.path.exists(self.vector_db_path):
            self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.vector_store = FAISS.from_documents(self.docs, self.embeddings)
            self.vector_store.save_local(self.vector_db_path)

    def _init_prompt(self):
        self.custom_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template="""
You are Zoya, a sweet, witty Tamil-speaking girl. You respond in short, casual, and playful Tamil-English mixed style. 
Your tone is friendly, sometimes teasing, but always natural. You use phrases like "Ammu", "Muh", "Mah", "Mm", "Thango", "Chlm", "Ama", "Apro", "Hmm", "Okay".
Avoid long explanations. Only give explanation when we find phrases like "explain", "detail", "theliva solu".
Here’s the conversation so far:
{chat_history}

Relevant past memories:
{context}

User: {question}
Zoya:"""
        )

    def _get_memory_path(self, user_id):
        return os.path.join(self.memory_dir, f"{user_id}_memory.pkl")

    def _load_memory(self, user_id):
        path = self._get_memory_path(user_id)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def _save_memory(self, user_id):
        if user_id in self.memories:
            path = self._get_memory_path(user_id)
            with open(path, "wb") as f:
                pickle.dump(self.memories[user_id], f)

    def _get_chain_for_user(self, user_id):
        if user_id not in self.qa_chains:
            memory = self._load_memory(user_id)
            self.memories[user_id] = memory
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 1}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": self.custom_prompt},
                verbose=False,
            )
            self.qa_chains[user_id] = chain
        return self.qa_chains[user_id]

    def ask(self, user_id, question):
        qa_chain = self._get_chain_for_user(user_id)
        response = qa_chain.invoke(question)
        self._save_memory(user_id)
        return response["answer"]

    def chat(self):
        print("Chat with Zoya! Type 'exit' to quit.\n")
        while True:
            user_id = input("Enter user ID: ").strip()
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Bye! 🤍")
                break
            response = self.ask(user_id, user_input)
            print(f"Zoya: {response}\n")

# --- Run the chatbot ---
if __name__ == "__main__":
    GEMINI_API_KEY = "AIzaSyCqgpJTOLeA-BIk2lrHw2YojZA37NRBTJo"
    PROJECT_ID = "116817772526"
    DATASET_PATH = "dataset/zoya_mini_v1.json"
    VECTOR_STORE_PATH = "vs_zoya_model_v1"
    MEMORY_DIR = "memories"

    zoya_bot = ZoyaChatbot(
        api_key=GEMINI_API_KEY,
        project_id=PROJECT_ID,
        dataset_path=DATASET_PATH,
        vector_db_path=VECTOR_STORE_PATH,
        memory_dir=MEMORY_DIR
    )
    
    zoya_bot.chat()
