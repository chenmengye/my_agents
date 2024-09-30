from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from config import HP
from openai import OpenAI
from langchain_community.utilities import SerpAPIWrapper
from utils import get_vectorstore, get_chat_llm, get_llm
import requests


@tool
def search(query: str):
    """只有需要了解实时信息或不知道答案的时候才会使用这个工具。"""
    serp = SerpAPIWrapper(serpapi_api_key=HP.SERPAPI_API_KEY)
    result = serp.run(query)
    # print("实时搜索结果:", result)
    return result


@tool
def generate_images(query: str):
    """需要生成图片时会使用该工具"""
    client = OpenAI(api_key=HP.OPENAI_KEY, base_url=HP.OPENAI_API_BASE)
    prompt = ''''''
    image_model = client.images.generate(
        model='dall-e-3',
        prompt=query,
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = image_model.data[0].url
    image = requests.get(image_url).content
    with open('images/result.png', 'wb') as f:
        f.write(image)
    return '图片生成成功'


@tool
def get_info_from_local_db(query: str):
    """当回答与一次注液机有关的工艺知识时，会使用这个工具"""
    client = get_vectorstore()
    retriever = client.as_retriever(search_type="mmr", k=1)
    result = retriever.invoke(query)
    return result

@tool
def intelligent_correction(query: str):
    """当需要对文本文档内容进行校验和纠错时，会使用这个工具"""
    model = get_chat_llm("gpt-3.5-turbo")
    prompt1 = ChatPromptTemplate.from_template(
        '''
        你是一名经验丰富的锂电设备领域的实体识别专家。
        以下是你的个人设定：
        1. 你对锂电设备领域的知识很是了解，在这个行业已经深耕20年了。
        2. 你非常擅长从锂电设备方面的文档，如设备说明书、设备技术协议、设备操作手册等提取关于机型、零部件、元器件等实体。
        现在提供一些文本，需要你按要求将文本中所涉及的实体提取出来
        要求：根据文本识别和一次注液机设备有关的实体，如：机型：一次注液机；部件：上料机器人；部件：静置站；部件：切刀；部件：夹具。
        用户文本:{text}
        '''
    )
    chain = prompt1 | model | StrOutputParser
    result = chain.invoke({'text': query})
    return result


if __name__ == '__main__':
    query = '''请你帮我检查以下关于一次注液机的安全事故预防及紧急处理文档是否有错误，请结合知识的内容
    一次注液机安全事故预防及紧急处理
    1 切刀伤害及紧急处理
    切刀较为锋利，调试或更换时容易伤到手指。最容易伤害的部位：手
    ①按下急停按钮；
    ②判断伤处情况，严重请及时就医；
    ③如果不是很严重先用酒精棉止血；
    ④	消毒（用双氧水等）；
    ⑤	用消过毒的纱布包扎。
    2 夹具夹伤及紧急处理
    人员调试过程中，机器点动时如果操作失误可能会出现夹伤或划伤的风险，风险较高的机器位置如下图所示，当出现这种紧急情况，应按下列操作紧急处理：
    ①伤口消毒；
    ②用消毒纱布将伤口擦干净；
    ③使用喷雾式创口贴或者免敏创口贴进行伤口保护；
    ④如果伤口超出外科轻伤范围，应及时就医。
        '''
    res = intelligent_correction(query)
    print(res)
