import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = os.getenv("GEN_API_MODEL", "claude-sonnet-4-6")
BASE_URL = os.getenv("GEN_API_BASE_URL", "https://proxy.gen-api.ru/v1")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_TOPIC = (
    "Искусственный интеллект в образовании: "
    "возможности, риски и перспективы"
)


class TaskBrief(BaseModel):
    """Задание и требования, полученные от пользователя на этапе 0."""

    topic: str = Field(description="Тема или задача статьи")
    audience: str = Field(description="Целевая аудитория")
    goal: str = Field(description="Цель статьи и ожидаемый результат для читателя")
    style: str = Field(description="Стиль и тон изложения")
    length: str = Field(description="Требуемый объём")
    must_include: List[str] = Field(description="Что обязательно раскрыть")
    must_avoid: List[str] = Field(description="Что нельзя использовать")


def ask(text: str, default: str = "") -> str:
    """Один вопрос в консоли: пустой ввод = значение по умолчанию."""
    suffix = f" [по умолчанию: {default}]" if default else ""
    try:
        answer = input(f"{text}{suffix}\n> ").strip()
    except EOFError:
        print(f"{text} (ввод недоступен, беру по умолчанию: {default})")
        return default
    return answer or default


def ask_list(text: str) -> List[str]:
    """Список: по одному пункту в строке, пустая строка завершает ввод."""
    print(f"{text}\n(по одному пункту в строке, пустая строка — закончить)")
    items: List[str] = []
    while True:
        try:
            line = input("- ").strip()
        except EOFError:
            break
        if not line:
            break
        items.append(line)
    return items


def collect_brief(args: argparse.Namespace) -> TaskBrief:
    """ЭТАП 0 — интерактивный ввод задания.

    Всё можно задать аргументами командной строки (тогда вопросы не задаются),
    либо ввести в консоли. Пустой ввод = значение по умолчанию.
    """
    complete_from_args = bool(args.topic and args.audience and args.goal
                              and args.style and args.length)
    interactive = (
        sys.stdin.isatty()
        and not complete_from_args
        and not args.yes
    )

    if interactive:
        print("\n" + "=" * 70)
        print("ЭТАП 0 — ПОСТАНОВКА ЗАДАНИЯ")
        print("=" * 70)
        print("Отвечайте на вопросы или нажимайте Enter для значения по умолчанию.\n")

    topic = args.topic or (
        ask("Тема или задача статьи", DEFAULT_TOPIC) if interactive else DEFAULT_TOPIC
    )

    brief = TaskBrief(
        topic=topic,
        audience=args.audience or (
            ask("Для кого пишем?", "широкая образованная аудитория")
            if interactive else "широкая образованная аудитория"),
        goal=args.goal or (
            ask("Какая цель статьи?",
                "разобраться в теме и понять, что делать на практике")
            if interactive else
            "разобраться в теме и понять, что делать на практике"),
        style=args.style or (
            ask("Стиль и тон?",
                "информационно-аналитический, нейтральный, без жаргона")
            if interactive else
            "информационно-аналитический, нейтральный, без жаргона"),
        length=args.length or (
            ask("Объём?", "1500–2500 слов") if interactive else "1500–2500 слов"),
        must_include=args.must if args.must is not None else (
            ask_list("Что обязательно раскрыть?") if interactive else []),
        must_avoid=args.avoid if args.avoid is not None else (
            ask_list("Что нельзя использовать?") if interactive else []),
    )

    if interactive:
        print("\nПроверьте задание:")
        print(brief.model_dump_json(indent=2, ensure_ascii=False))
        try:
            if input("\nЗапускаем цепочку? [Y/n] ").strip().lower().startswith("н"):
                raise SystemExit("Запуск отменён. Измените параметры и повторите.")
        except EOFError:
            pass

    return brief



class TaskAnalysis(BaseModel):
    goal: str = Field(description="Цель статьи")
    audience: str = Field(description="Целевая аудитория")
    style: str = Field(description="Рекомендуемый стиль")
    key_questions: List[str] = Field(description="Ключевые вопросы")
    length_requirements: str = Field(description="Требования к объёму")
    quality_criteria: List[str] = Field(description="Критерии качества")


class ArticlePlan(BaseModel):
    title: str = Field(description="Рабочий заголовок")
    sections: List[str] = Field(description="Разделы статьи")
    theses: List[str] = Field(description="Основные тезисы")
    logical_sequence: str = Field(description="Логика последовательности")


class Review(BaseModel):
    score: int = Field(ge=0, le=10, description="Оценка от 0 до 10")
    strengths: List[str] = Field(description="Сильные стороны")
    weaknesses: List[str] = Field(description="Недостатки")
    factual_risks: List[str] = Field(description="Факты, требующие проверки")
    recommendations: List[str] = Field(description="Рекомендации")
    approved: bool = Field(description="Готова ли статья")


llm = ChatOpenAI(
    model=MODEL,
    temperature=0.3,
    base_url=BASE_URL,
    api_key=os.getenv("GEN_API_KEY"),
)

# 1. Анализ задания
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты редактор и аналитик контента.
Проанализируй задание на будущую статью. Не пиши статью.
Определи цель, аудиторию, стиль, ключевые вопросы,
требования к объёму и критерии качества.
Все ограничения из задания обязательны и должны попасть в критерии качества."""),
    ("human", "Задание:\n{brief}")
])
analysis_chain = analysis_prompt | llm.with_structured_output(TaskAnalysis)

# 2. План
plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты опытный редактор.
На основании задания и анализа разработай логичный план статьи.
Не пиши полный текст. Сформируй заголовок, разделы,
тезисы и логику последовательности.
Разделы должны покрывать обязательные темы из задания."""),
    ("human", """Задание:
{brief}

Анализ:
{analysis}""")
])
plan_chain = plan_prompt | llm.with_structured_output(ArticlePlan)

# 3. Написание
writing_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты профессиональный автор информационно-аналитических статей.
Напиши полноценную статью по заданию, анализу и плану.
Соблюдай аудиторию, стиль и объём из задания.
Обязательно раскрой указанные темы и не используй то, что заказчик запретил.
Не выдумывай статистику, исследования или цитаты.
Факты, требующие подтверждения, формулируй осторожно."""),
    ("human", """Задание:
{brief}

Анализ:
{analysis}

План:
{plan}

Напиши готовую статью.""")
])
writing_chain = writing_prompt | llm

# 4. Ревью
review_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты независимый редактор.
Проведи критическое ревью готовой статьи.
Сначала проверь соответствие заданию заказчика: цель, аудитория, объём,
стиль, раскрытие обязательных тем и наличие запрещённых приёмов.
Затем проверь полноту, логику, структуру, повторы
и потенциально неподтверждённые факты.
Не переписывай статью целиком."""),
    ("human", """Задание заказчика:
{brief}

Требования после анализа:
{analysis}

Статья:
{article}""")
])
review_chain = review_prompt | llm.with_structured_output(Review)


def show_stage(title, value):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if isinstance(value, BaseModel):
        print(value.model_dump_json(indent=2, ensure_ascii=False))
    else:
        print(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Многоэтапная цепочка подготовки статьи (LangChain + GenAPI)"
    )
    parser.add_argument("--topic", help="Тема статьи")
    parser.add_argument("--audience", help="Целевая аудитория")
    parser.add_argument("--goal", help="Цель статьи")
    parser.add_argument("--style", help="Стиль изложения")
    parser.add_argument("--length", help="Требуемый объём")
    parser.add_argument("--must", nargs="+", help="Обязательные к раскрытию темы")
    parser.add_argument("--avoid", nargs="+", help="Что нельзя использовать")
    parser.add_argument(
        "--yes", action="store_true",
        help="Без интерактивных вопросов (всё берётся из аргументов или дефолтов)",
    )
    return parser.parse_args()


def main():
    if not os.getenv("GEN_API_KEY"):
        raise SystemExit(
            "Не задан GEN_API_KEY. Создайте файл .env на основе "
            ".env.example и укажите ключ из личного кабинета GenAPI."
        )

    args = parse_args()
    brief = collect_brief(args)
    brief_json = brief.model_dump_json(ensure_ascii=False)

    print("\nAI ARTICLE WRITER — LANGCHAIN")
    print(f"Провайдер: GenAPI ({BASE_URL})")
    print(f"Модель: {MODEL}")
    print(f"Тема: {brief.topic}")

    analysis = analysis_chain.invoke({"brief": brief_json})
    show_stage("ЭТАП 1 — АНАЛИЗ ЗАДАНИЯ", analysis)

    plan = plan_chain.invoke({
        "brief": brief_json,
        "analysis": analysis.model_dump_json(ensure_ascii=False)
    })
    show_stage("ЭТАП 2 — РАЗРАБОТКА ПЛАНА", plan)

    article_response = writing_chain.invoke({
        "brief": brief_json,
        "analysis": analysis.model_dump_json(ensure_ascii=False),
        "plan": plan.model_dump_json(ensure_ascii=False)
    })
    article = article_response.content
    show_stage("ЭТАП 3 — НАПИСАНИЕ СТАТЬИ", article)

    review = review_chain.invoke({
        "brief": brief_json,
        "analysis": analysis.model_dump_json(ensure_ascii=False),
        "article": article
    })
    show_stage("ЭТАП 4 — РЕДАКТОРСКОЕ РЕВЬЮ", review)

    result = {
        "brief": brief.model_dump(mode="json"),
        "model": MODEL,
        "analysis": analysis.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "article": article,
        "review": review.model_dump(mode="json")
    }

    (OUTPUT_DIR / "chain_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    def bullets(items):
        return "\n".join(f"- {x}" for x in items) if items else "- не задано"

    markdown = f"""# {plan.title}

## Задание

- **Тема:** {brief.topic}
- **Аудитория:** {brief.audience}
- **Цель:** {brief.goal}
- **Стиль:** {brief.style}
- **Объём:** {brief.length}

**Обязательно раскрыть:**
{bullets(brief.must_include)}

**Избегаем:**
{bullets(brief.must_avoid)}

---

## Статья

{article}

---

# Редакторское заключение

**Оценка: {review.score}/10**

## Сильные стороны
{bullets(review.strengths)}

## Недостатки
{bullets(review.weaknesses)}

## Фактические риски
{bullets(review.factual_risks)}

## Рекомендации
{bullets(review.recommendations)}

**Готовность к публикации:** {"да" if review.approved else "нет"}
"""
    (OUTPUT_DIR / "article.md").write_text(markdown, encoding="utf-8")

    print("\nГотово. Результаты сохранены в папке output/.")


if __name__ == "__main__":
    main()
