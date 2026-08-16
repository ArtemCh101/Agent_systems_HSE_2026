# Smart Assistant CLI with Intent Routing & Memory Management

Интерактивный CLI-ассистент на базе LangChain и Ollama, реализующий классификацию намерений пользователя, динамический роутинг, адаптивную память и механизмы отказоустойчивости.

## Архитектурные особенности

* **Intent Classification & Routing**: Вызовы классифицируются с помощью Pydantic-схем на 5 категорий (Question, Task, Small Talk, Complaint, Unknown), после чего маршрутизируются в соответствующие хэндлеры.
* **Resilience Mode**: Автоматический fallback с основной модели (Llama 3.1) на резервную (Qwen 2.5) при сбоях API или локального инференса.
* **Smart Memory Manager**: Поддержка скользящего окна сообщений (Buffer) и динамического сжатия контекста (LLM Summarization) при превышении лимитов.
* **Persona Customization**: Динамическая смена характера и стиля ответов (Friendly, Professional, Sarcastic, Pirate) без потери контекста.
* **Performance & UX**: Стриминговый вывод ответов, кэширование повторных запросов (InMemoryCache) и интерфейс системных команд.

## Установка и запуск

```bash
git clone [https://github.com/ArtemCh101/Agent_systems_HSE_2026.git](https://github.com/ArtemCh101/Agent_systems_HSE_2026.git)

cd Agent_systems_HSE_2026

pip install -r requirements.txt

ollama pull llama3.1

ollama pull qwen2.5

# Запуск CLI-приложения с конфигурацией по умолчанию
python app.py --model llama3.1 --character friendly --memory buffer
```
## Системные команды CLI

* `/character <name>`: сменить характер (friendly, professional, sarcastic, pirate)
* `/memory <strategy>`: сменить стратегию памяти (buffer, summary)
* `/clear`: очистить историю диалога
* `/status`: показать текущие настройки и размер истории
* `/quit`: завершить работу
