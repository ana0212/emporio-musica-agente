from types import SimpleNamespace

import pytest

from src import agent


def make_response(content="", tool_calls=None):
    """Create a lightweight Ollama-like response for tests."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(message=message)


def make_tool_call(name, arguments):
    """Create a lightweight Ollama tool-call object."""
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(function=function)


def test_empty_message_returns_prompt():
    result = agent.ask_agent("   ")
    assert result == "Por favor, digite sua dúvida para que eu possa ajudar."


def test_simple_greeting_returns_menu_without_calling_model(monkeypatch):
    def unexpected_chat_call(*args, **kwargs):
        pytest.fail("O modelo não deveria ser chamado para uma saudação simples.")

    monkeypatch.setattr(agent, "chat", unexpected_chat_call)
    result = agent.ask_agent("Olá, tudo bem?")

    assert "Sou o Kiko" in result
    assert "Consulta de produtos" in result
    assert "Trocas e devoluções" in result


def test_closing_intent_returns_closing_message_without_calling_model(monkeypatch):
    history = [{
        "role": "assistant",
        "content": "Ficou claro? Posso ajudar com mais alguma informação?",
    }]

    def unexpected_chat_call(*args, **kwargs):
        pytest.fail("O modelo não deveria ser chamado para encerrar a conversa.")

    monkeypatch.setattr(agent, "chat", unexpected_chat_call)
    result = agent.ask_agent("Não, obrigada.", history=history)

    assert "Agradecemos seu contato" in result
    assert "estaremos à disposição" in result


def test_policy_question_retrieves_context_before_generating_answer(monkeypatch):
    retrieved = [{
        "text": "Trocas por preferência podem ser solicitadas em até 7 dias.",
        "page": 4,
        "distance": 0.2,
    }]
    retrieval_calls = []

    def fake_search_store_policies(query):
        retrieval_calls.append(query)
        return retrieved

    def fake_chat(**kwargs):
        messages = kwargs["messages"]
        assert any(
            "Trechos do manual:" in item.get("content", "")
            for item in messages
            if item.get("role") == "system"
        )
        return make_response(
            "As trocas por preferência podem ser solicitadas em até 7 dias."
        )

    monkeypatch.setattr(agent, "search_store_policies", fake_search_store_policies)
    monkeypatch.setattr(agent, "chat", fake_chat)

    result = agent.ask_agent("Posso trocar um produto por preferência?")

    assert len(retrieval_calls) == 1
    assert "trocar" in retrieval_calls[0].casefold()
    assert "7 dias" in result
    assert "Ficou claro?" in result


def test_product_search_executes_registered_tool_and_returns_answer(monkeypatch):
    tool_calls = [make_tool_call(
        "search_products",
        {
            "query": "violão",
            "category_id": None,
            "in_stock_only": True,
        },
    )]
    tool_results = []

    def fake_search_products(**kwargs):
        tool_results.append(kwargs)
        return [{"product_id": 10, "name": "Violão de teste", "stock": 2}]

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_products", fake_search_products)

    responses = iter([
        make_response(content="", tool_calls=tool_calls),
        make_response(content="Encontrei um violão disponível."),
    ])
    monkeypatch.setattr(agent, "chat", lambda **kwargs: next(responses))

    result = agent.ask_agent("Quais violões estão disponíveis?")

    assert len(tool_results) == 1
    assert tool_results[0]["query"] == "violão"
    assert tool_results[0]["in_stock_only"] is True
    assert "Encontrei um violão disponível." in result
    assert "Ficou claro?" in result


def test_empty_model_response_returns_fallback_message(monkeypatch):
    monkeypatch.setattr(agent, "chat", lambda **kwargs: make_response(content=""))

    result = agent.ask_agent("Qual é a melhor opção de instrumento?")

    assert "Não consegui formular uma resposta" in result
