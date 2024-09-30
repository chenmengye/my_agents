from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from Mytools import *
from utils import get_chat_llm
from config import HP


class ChatBot:
    def __init__(self):
        self.model = get_chat_llm("gpt-4")
        self.MOOD = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEM_PL = HP.SystemPromptTemplate
        self.MOODS = HP.MOODS_DICT
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                   "system",
                   self.SYSTEM_PL.format(who_you_are=self.MOODS[self.MOOD]["roleSet"]),
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        
        tools = [search, get_info_from_local_db, generate_images]
        agent = create_openai_tools_agent(
            self.model,
            tools = tools,
            prompt=self.prompt,
        )

        self.memory = self.get_memory()
        memory = ConversationTokenBufferMemory(
            llm=self.model,
            human_prefix="用户",
            ai_prefix="大师",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            max_token_limit=1000,
            chat_memory=self.memory,
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
        )
    
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url=HP.REDIS_URL, session_id="session"
        )
        # print("chat_message_history:", chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(store_message) > 3:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEM_PL + "\n这是一段你和用户的对话记忆，对其进行总结摘要，尤其要注意记住用户的个人信息"
                    ),
                    ("user", "{input}"),
                ]
            )
            chain = prompt | self.model
            summary = chain.invoke({"input": store_message, "who_you_are": self.MOODS[self.MOOD]["roleSet"]})
            # print("summary:", summary)
            chat_message_history.clear()  # 清空历史记录
            chat_message_history.add_message(summary)
            # print("总结后：", chat_message_history.messages)
        return chat_message_history

    def run(self, query):
        self.mood_chain(query)
        result = self.agent_executor.invoke({"input": query, "chat_history": self.memory.messages})
        return result
    
    def mood_chain(self, query: str):
        prompt = HP.MoodPromptTemplate
        chain = ChatPromptTemplate.from_template(prompt) | self.model | StrOutputParser()
        result = chain.invoke({"query": query})
        self.MOOD = result
        print("情绪判断结果:", self.MOOD)
        return result


if __name__ == "__main__":
    chatbot = ChatBot()
    '''
    你好，我叫小陈
    先导智能装备股份有限公司当前的股价是多少
    一次注液机有哪些构成部件
    切刀是否是一次注液机的构成部件
    请帮我生成一张关于鲜花的图片，要求简约大方，体现生活的美好
    你还记得我叫什么名字吗
    '''
    result = chatbot.run(query="你工作做得很好")
    print(result)
