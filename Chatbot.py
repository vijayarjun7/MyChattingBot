import os

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    groq_api = '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'
    model = 'llama3-8b-8192'

    groq_chat = ChatGroq(
        groq_api_key = groq_api,
        model_name = model
    )

    print("Hello! I'm your friendly Groq Chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation")

    system_prompt = 'You are a friendly conversational chatbot'
    conversatioanl_memory_length = 5

    memory = ConversationBufferWindowMemory(k=conversatioanl_memory_length, memory_key='chat_history', return_messages=True)

    while True:
        user_question = input('Ask a Question...')

        if user_question:

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=system_prompt
                    ),

                    MessagesPlaceholder(
                        variable_name='chat_history'
                    ),

                    HumanMessagePromptTemplate.from_template(
                        "{human_input}"
                    )
                ]
            )

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=False,
                memory=memory
            )

            response = conversation.predict(human_input=user_question)
            print(f'Chatbot: {response}')

if __name__ == '__main__':
    main()
