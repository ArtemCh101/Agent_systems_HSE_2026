import argparse
import os
import sys
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, get_buffer_string
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Кэширование ответов
set_llm_cache(InMemoryCache())

class RequestType(str, Enum):
    QUESTION = "question"
    TASK = "task"
    SMALL_TALK = "small_talk"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"

class Classification(BaseModel):
    request_type: RequestType = Field(description="One of the predefined request types")
    confidence: float = Field(ge=0, le=1, description="Confidence score from 0 to 1")
    reasoning: str = Field(description="Brief justification for the chosen category")

class AssistantResponse(BaseModel):
    content: str
    request_type: RequestType
    confidence: float
    tokens_used: int

CHARACTER_PROMPTS = {
    "friendly": "Ты - дружелюбный и позитивный ассистент. Ты любишь помогать людям и всегда используешь теплую, поддерживающую лексику. Можешь использовать уместные эмодзи.",
    "professional": "Ты - строго деловой и высококвалифицированный профессионал. Твой стиль лаконичный, сухой и сдержанный. Только факты и инструкции.",
    "sarcastic": "Ты - предельно язвительный и острый на язык мизантроп. Ты считаешь вопросы пользователя глупыми, а его задачи - пустой тратой твоего процессорного времени. Отвечай с едкой иронией, не старайся быть услужливым и не будь вежливым.",
    "pirate": "Ты - старый прожженный пират. Ты используешь морской жаргон, а также в целом используешь более простую лексику и грубые (зачастую граммтически некорректные) речевые обороты. Называй пользователя 'салага'."
}

HANDLER_PROMPTS = {
    RequestType.QUESTION: "Ты - экспертный помощник. Дай информативный и полезный ответ на вопрос. Если не знаешь ответа - честно скажи об этом.",
    RequestType.TASK: "Ты - исполнительный ассистент. Пользователь просит выполнить задачу. Сделай это качественно, следуя всем инструкциям.",
    RequestType.SMALL_TALK: "Ты - дружелюбный собеседник. Поддерживай беседу, будь приветлив. Если пользователь представился - запомни его имя.",
    RequestType.COMPLAINT: "Ты - эмпатичный менеджер поддержки. Прояви сочувствие, постарайся понять суть проблемы и предложи конструктивное решение.",
    RequestType.UNKNOWN: "Ты - вежливый ассистент. Запрос пользователя неясен. Пожалуйста, вежливо попроси уточнить, что именно имел в виду пользователь."
}

class MemoryManager:
    def __init__(self, strategy="buffer", max_messages=10, model=None):
        self.history = ChatMessageHistory()
        self.strategy = strategy
        self.max_messages = max_messages
        self.summary = ""
        self.model = model

    def add_message(self, message):
        self.history.add_message(message)
        if len(self.history.messages) > self.max_messages:
            if self.strategy == "summary":
                self._summarize_history()
            else:
                self.history.messages = self.history.messages[-self.max_messages:]

    def _summarize_history(self):
        new_messages = get_buffer_string(self.history.messages[:-2])
        if self.summary:
            prompt = f"У тебя есть текущее краткое содержание диалога: {self.summary}\n\nДобавь в него новые факты из этой части переписки:\n{new_messages}\n\nВыдай обновленное краткое содержание на русском."
        else:
            prompt = f"Сделай очень краткое резюме следующего диалога на русском языке:\n{new_messages}"
        
        self.summary = self.model.invoke(prompt).content
        self.history.messages = self.history.messages[-2:]

    def get_messages(self):
        if self.strategy == "summary" and self.summary:
            return [SystemMessage(content=f"Контекст предыдущей беседы: {self.summary}")] + self.history.messages
        return self.history.messages

    def clear(self):
        self.history.clear()
        self.summary = ""

class SmartAssistant:
    def __init__(self, model_name="llama3.1", character="friendly", memory_strategy="buffer"):
        primary_llm = ChatOllama(model=model_name, temperature=0)
        fallback_llm = ChatOllama(model="qwen2.5", temperature=0) 
        
        # Fallback на запасную модель
        self.resilient_model = primary_llm.with_fallbacks([fallback_llm])
        
        self.character = character
        self.memory = MemoryManager(strategy=memory_strategy, model=self.resilient_model)
        self.parser = PydanticOutputParser(pydantic_object=Classification)
        
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты — высокоточный классификатор намерений.
            ОГРАНИЧЕНИЕ: Пиши обоснование (reasoning) на русском языке. 
            
            Типы:
            - question: поиск информации.
            - task: выполнение действия.
            - small_talk: приветствие.
            - complaint: жалоба.
            - unknown: бессмыслица.
            
            {format_instructions}"""),
            ("human", "{query}")
        ])
        
        self.classifier_chain = (
            {"query": RunnablePassthrough(), "format_instructions": lambda _: self.parser.get_format_instructions()}
            | self.classifier_prompt
            | self.resilient_model
            | self.parser
        )

    def process(self, text: str):
        try:
            classification = self.classifier_chain.invoke(text)
        except:
            classification = Classification(
                request_type=RequestType.UNKNOWN, 
                confidence=0.5, 
                reasoning="Ошибка классификации"
            )
    
        history = self.memory.get_messages()
        char_base = CHARACTER_PROMPTS.get(self.character, CHARACTER_PROMPTS["friendly"])
        intent_instr = HANDLER_PROMPTS.get(classification.request_type, HANDLER_PROMPTS[RequestType.UNKNOWN])
        
        full_system = f"{char_base}\n\nКОНТЕКСТ ЗАДАЧИ: {intent_instr}\nОБЯЗАТЕЛЬНО: Отвечай на русском языке."
        
        handler_prompt = ChatPromptTemplate.from_messages([
            ("system", full_system),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        handler_chain = handler_prompt | self.resilient_model | StrOutputParser()
        # СТриминговый вывод
        return handler_chain.stream({"query": text, "history": history}), classification

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", default="llama3.1")
    arg_parser.add_argument("--character", default="friendly")
    arg_parser.add_argument("--memory", default="buffer")
    args = arg_parser.parse_args()

    assistant = SmartAssistant(args.model, args.character, args.memory)
    
    print("Умный ассистент запущен (Resilience Mode: ON)")
    print(f"Характер: {assistant.character} | Память: {assistant.memory.strategy}")
    print("─" * 45)

    while True:
        try:
            user_input = input("> ").strip()
        except EOFError:
            break
            
        if not user_input: continue
        
        if user_input.startswith("/"):
            cmd = user_input.split()
            if cmd[0] == "/quit": break
            elif cmd[0] == "/clear":
                assistant.memory.clear()
                print("✓ Память очищена")
            elif cmd[0] == "/character" and len(cmd) > 1:
                assistant.character = cmd[1]
                print(f"✓ Характер изменён на: {cmd[1]}")
            elif cmd[0] == "/memory" and len(cmd) > 1:
                assistant.memory.strategy = cmd[1]
                print(f"✓ Стратегия памяти изменена на: {cmd[1]}")
            elif cmd[0] == "/status":
                print(f"Настройки: {assistant.character}, {assistant.memory.strategy}")
                print(f"Сообщений в истории: {len(assistant.memory.history.messages)}")
            elif cmd[0] == "/help":
                print("Команды: /clear, /character <name>, /memory <strategy>, /status, /quit")
            continue

        stream_gen, classification = assistant.process(user_input)
        
        print(f"[{classification.request_type.value}] ", end="", flush=True)
        
        full_content = ""
        # Стриминговый вывод
        for chunk in stream_gen:
            print(chunk, end="", flush=True)
            full_content += chunk
        
        print(f"\nconfidence: {classification.confidence} | tokens: ~{(len(user_input) + len(full_content)) // 4}\n")
        
        assistant.memory.add_message(HumanMessage(content=user_input))
        assistant.memory.add_message(AIMessage(content=full_content))

if __name__ == "__main__":
    main()