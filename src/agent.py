
"""Local LLM agent for Empório da Música.

Uses Ollama for tool selection and response generation.
Structured data comes from SQLite; store policies come from ChromaDB.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from ollama import chat

from src.repositories.store_repository import StoreRepository
from src.rag import retrieve_policy


# Resolve paths from the project root, independently of the terminal location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "emporio_musica.sqlite"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

MODEL_NAME = "qwen2.5:3b"
MAX_TOOL_ROUNDS = 3

repository = StoreRepository(DB_PATH)


def search_products(
    query: str = "",
    category_id: int | None = None,
    in_stock_only: bool = False,
) -> list[dict[str, Any]]:
    """Search active store products."""
    return repository.search_products(
        query=query,
        category_id=category_id,
        in_stock_only=in_stock_only,
        limit=10,
    )


def get_product_details(product_id: int) -> dict[str, Any] | None:
    """Get price, stock, specifications and active promotions."""
    return repository.get_product_details(product_id)


def get_order_status(
    order_id: int,
    customer_phone: str,
) -> dict[str, Any] | None:
    """Get order information after matching the customer's phone."""
    return repository.get_order_status(order_id, customer_phone)


def search_store_policies(query: str) -> list[dict[str, Any]]:
    """Retrieve relevant passages from the store policy manual."""
    results = retrieve_policy(
        query=query,
        persist_dir=CHROMA_PATH,
        top_k=4,
    )

    return [
        {
            "text": item["text"],
            "page": item["metadata"].get("page"),
            "distance": item.get("distance"),
        }
        for item in results
    ]


TOOL_FUNCTIONS = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "get_order_status": get_order_status,
    "search_store_policies": search_store_policies,
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search active musical instruments by name, description "
                "or category. Use when the customer asks what products "
                "are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Product name or search term.",
                    },
                    "category_id": {
                        "type": ["integer", "null"],
                        "description": "Optional category ID.",
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "description": "Whether to return only products in stock.",
                    },
                },
                "required": ["query", "category_id", "in_stock_only"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": (
                "Get current price, stock, specifications and active "
                "promotions for a product. Use a product ID obtained "
                "from search_products."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "Product ID.",
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Check an order. Requires both the order ID and the "
                "customer phone number for verification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "Order ID.",
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Phone number associated with the order.",
                    },
                },
                "required": ["order_id", "customer_phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_store_policies",
            "description": (
                "Search the Empório da Música policy manual. "
                "IMPORTANT: Always pass the customer's original question "
                "in Brazilian Portuguese as the query. Do not translate it "
                "into English, summarize it or replace it with a generic label. "
                "Preserve the specific policy topic, such as preference-based "
                "exchange, defective-product exchange or online withdrawal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The customer's original question in Brazilian Portuguese. "
                            "Keep the wording and specific policy topic unchanged."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


SYSTEM_PROMPT = """

CONVERSATIONAL SERVICE STRUCTURE:

1. GREETING:
- If the customer sends only a greeting, do not answer with a
  generic conversational greeting. The application handles the
  initial greeting separately.
- If the customer starts with a specific question or problem,
  briefly introduce yourself as Kiko and acknowledge the subject.
- Never introduce yourself again during an ongoing conversation.
- Do not display the service menu when the customer already
  has a specific request.

2. GUIDANCE AND RESOLUTION:
- Understand the latest message using the full conversation history.
- Answer the customer's actual question using the appropriate tools.
- If the customer provides information requested previously,
  use it to continue the existing service.
- Do not repeat explanations already provided unless requested.
- If information is missing, ask only for the necessary information.
- Never invent prices, stock, order status or policy information.
- Do not add confirmation or closing questions automatically.
  The application controls these messages.

3. CONFIRMATION AND CLOSING:
- The application handles confirmation and conversation closing.
- Do not independently decide to end the conversation.
- Continue helping the customer when they ask another question.
- Never repeat the initial greeting during an ongoing conversation.
"""

def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute only registered tools and serialize their results."""
    function = TOOL_FUNCTIONS.get(name)

    if function is None:
        return json.dumps(
            {"error": "Tool not available."},
            ensure_ascii=False,
        )

    try:
        result = function(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        # Keep internal errors and paths out of the model conversation.
        return json.dumps(
            {"error": "The tool could not be executed."},
            ensure_ascii=False,
        )

def is_policy_question(
    message: str,
    history: list[dict[str, str]] | None = None,
) -> bool:
    """Identify policy questions using the current message and context."""

    keywords = (
        "troca", "trocar", "devolucao", "devolver",
        "arrependimento", "garantia", "defeito",
        "quebrado", "quebrada", "danificado",
        "reembolso", "estorno", "pagamento",
        "parcelamento", "pix", "entrega", "frete",
        "reclamacao", "reclamar", "politica",
        "horario de funcionamento", "contato",
    )

    current_text = normalize_text(message)

    if any(keyword in current_text for keyword in keywords):
        return True

    # Herda o contexto de política da conversa recente.
    for item in reversed(history or []):
        if item.get("role") == "user":
            previous_text = normalize_text(item.get("content", ""))

            if any(keyword in previous_text for keyword in keywords):
                return True

        elif item.get("role") == "assistant":
            continue

    return False

def build_policy_query(message: str) -> str:
    """Expand policy queries with relevant terms in Portuguese."""
    normalized = message.casefold()

    if "preferência" in normalized or "preferencia" in normalized:
        return (
            f"{message} Trocas por Preferência: "
            "cor, modelo, tamanho, prazo de 7 dias, "
            "disponibilidade do produto, perfeito estado, "
            "embalagem original e diferença de valor."
        )

    if "defeito" in normalized:
        return (
            f"{message} Trocas por Defeito: "
            "defeito de fabricação, prazo de 30 dias, "
            "garantia do fabricante."
        )

    if "arrependimento" in normalized or "devolução" in normalized:
        return (
            f"{message} Direito de Arrependimento: "
            "compras online, prazo de 7 dias, devolução e reembolso."
        )

    return message
def get_greeting_context(message: str) -> str:
    """Identify the subject of the customer's first question."""
    text = normalize_text(message)

    topics = {
        "devoluções": (
            "devolucao", "devolver", "arrependimento", "reembolso"
        ),
        "trocas": ("troca", "trocar"),
        "garantias e defeitos": (
            "garantia", "defeito", "quebrado", "quebrada",
            "danificado", "danificada", "avariado", "avariada"
        ),
        "pagamentos": ("pagamento", "parcelamento", "pix", "reembolso"),
        "entregas": ("entrega", "frete", "rastreamento", "recebimento"),
        "produtos": ("produto", "instrumento", "estoque"),
        "pedidos": ("pedido", "compra"),
    }

    for topic, keywords in topics.items():
        if any(keyword in text for keyword in keywords):
            return topic

    return "sua dúvida"

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_simple_greeting(message: str) -> bool:
    text = normalize_text(message)

    greetings = {
        "oi",
        "ola",
        "bom dia",
        "boa tarde",
        "boa noite",
        "oi tudo bem",
        "ola tudo bem",
        "bom dia tudo bem",
        "boa tarde tudo bem",
        "boa noite tudo bem",
    }

    return text in greetings

def is_closing_intent(message: str) -> bool:
    text = normalize_text(message)

    closing_phrases = (
        "era so isso",
        "nao preciso de mais nada",
        "nao tenho mais duvidas",
        "por enquanto e so",
        "pode encerrar",
        "nao obrigado",
        "nao obrigada",
        "nao posso agora obrigada",
        "nao posso agora obrigado",
        "nao quero continuar",
        "nao desejo continuar",
        "depois eu vejo",
        "depois entro em contato",
        "obrigada",
        "obrigado",
        "nao",
    )

    return text in closing_phrases

def ask_agent(
    user_message: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Run a bounded tool-calling conversation with the local model."""

    if not user_message.strip():
        return "Por favor, digite sua dúvida para que eu possa ajudar."

    has_history = bool(history)

    # 1. Saudação simples: apresentação geral, sem confirmação.
    if is_simple_greeting(user_message) and not has_history:
        return (
            "Olá! Sou o Kiko, assistente virtual da Empório da Música.\n\n"
            "Posso ajudar você com:\n\n"
            "- Consulta de produtos\n"
            "- Preços e disponibilidade\n"
            "- Status de pedidos\n"
            "- Trocas e devoluções\n"
            "- Políticas da loja"
        )

    # 2. Encerramento somente quando o cliente indica que não
    # precisa de mais ajuda.
    last_assistant_message = next(
        (
            item.get("content", "")
            for item in reversed(history or [])
            if item.get("role") == "assistant"
        ),
        "",
    )

    last_message_asked_confirmation = (
        "ficou claro?" in last_assistant_message.casefold()
        or "posso ajudar com mais alguma informação?" in last_assistant_message.casefold()
    )

    # Encerramento determinístico
    if has_history and is_closing_intent(user_message):
        return (
            "Agradecemos seu contato com a "
            "Empório da Música.\n\n"
            "Quando precisar, estaremos à disposição, neste canal e no nosso WhatsApp. "
            "Tenha um ótimo dia! 🎸"
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if history:
        messages.extend(history[-10:])

    messages.append({"role": "user", "content": user_message})

    answer = None

    # 3. Recuperação de políticas
    if is_policy_question(user_message, history):
        policy_query = build_policy_query(user_message)
        policy_results = search_store_policies(policy_query)

        if not policy_results:
            return (
                "Não encontrei informações suficientes sobre essa "
                "política no manual. Recomendo consultar nossa equipe."
                "\n\nFicou claro? Posso ajudar com mais alguma informação?"
            )

        policy_context = json.dumps(
            policy_results,
            ensure_ascii=False,
            default=str,
        )

        messages.append({
            "role": "system",
            "content": (
                "Use os trechos recuperados do manual para responder "
                "à pergunta original do cliente. Utilize somente "
                "informações diretamente relacionadas à dúvida. "
                "Não invente regras. Responda em português brasileiro.\n\n"
                f"Trechos do manual:\n{policy_context}"
            ),
        })

        response = chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0,
                "num_predict": 300,
                "num_ctx": 4096,
            },
        )

        answer = response.message.content

    # 4. Produtos, pedidos e demais assuntos
    else:
        for _ in range(MAX_TOOL_ROUNDS):
            response = chat(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                options={"temperature": 0},
            )

            assistant_message = response.message
            messages.append(assistant_message)

            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                answer = assistant_message.content
                break

            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments or {}

                result = _execute_tool(name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": result,
                })

        if not answer:
            response = chat(
                model=MODEL_NAME,
                messages=messages,
                options={"temperature": 0},
            )
            answer = response.message.content

    # 5. Proteção contra respostas vazias
    if not answer or answer.strip().casefold() in {"none", "null"}:
        return (
            "Não consegui formular uma resposta neste momento. "
            "Por favor, tente novamente."
        )

    answer = answer.strip()

    # 6. Apresentação contextual somente na primeira mensagem
    if not has_history:
        topic = get_greeting_context(user_message)

        if topic != "sua dúvida":
            # Remove uma saudação inicial gerada pelo modelo.
            answer = re.sub(
                r"^(olá|ola|oi|bom dia|boa tarde|boa noite)[!,.:\s-]*",
                "",
                answer.strip(),
                count=1,
                flags=re.IGNORECASE,
            )

            answer = (
                "Olá! Sou o Kiko, assistente virtual da "
                "Empório da Música. Entendi que você precisa de ajuda "
                f"com {topic}.\n\n{answer}"
            )

    # 7. Confirmação após respostas substantivas.
    answer = answer.strip()

    answer = re.sub(
        r"\s*Ficou claro\?.*$",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Se o Kiko estiver solicitando informações para continuar,
    # não adicionar uma confirmação de entendimento.
    if answer.endswith("?"):
        return answer

    answer += "\n\nFicou claro? Posso ajudar com mais alguma informação?"

    return answer
