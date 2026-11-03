
import streamlit as st

from src.agent import ask_agent


# --------------------------------------------------
# Configuração da página
# --------------------------------------------------

st.set_page_config(
    page_title="Empório da Música",
    page_icon="🎸",
    layout="wide",
)


# --------------------------------------------------
# Estilo visual
# --------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f5f0;
        }

        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #252525;
            padding-top: 20px;
        }

        .subtitle {
            text-align: center;
            color: #666666;
            font-size: 1.1rem;
            margin-bottom: 35px;
        }

        .info-card {
            background-color: white;
            padding: 22px;
            border-radius: 12px;
            border: 1px solid #e8e4dc;
            height: 100%;
        }

        .info-card h3 {
            color: #333333;
            margin-top: 0;
        }

        .footer {
            text-align: center;
            color: #777777;
            font-size: 0.85rem;
            padding: 20px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Estado da aplicação
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []


# --------------------------------------------------
# Navegação
# --------------------------------------------------

def open_chat():
    st.session_state.page = "chat"


def back_to_home():
    st.session_state.page = "home"


# ==================================================
# PÁGINA INICIAL
# ==================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🎸 Empório da Música</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Sua paixão pela música começa aqui.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("## Sobre a Empório da Música")

    st.write(
        """
        A Empório da Música é uma loja especializada em instrumentos
        musicais, fundada em 2008 em Campo Grande, Mato Grosso do Sul.

        Com mais de 15 anos de atuação no mercado, somos referência
        regional na venda de instrumentos para músicos iniciantes,
        intermediários e profissionais.

        Nosso catálogo abrange mais de 300 instrumentos de marcas
        nacionais e internacionais, incluindo guitarras, baixos,
        violões, baterias, teclados, instrumentos de sopro,
        cordas orquestrais e ukuleles.

        Trabalhamos exclusivamente com instrumentos musicais.
        Não comercializamos acessórios como cordas, palhetas,
        cabos, cases, pedais ou amplificadores.
        """
    )

    st.divider()

    st.markdown("## Nossa missão")

    st.write(
        """
        Democratizar o acesso a instrumentos musicais de qualidade
        na região Centro-Oeste, oferecendo atendimento especializado
        e preços competitivos, contribuindo para o desenvolvimento
        cultural e musical da comunidade.
        """
    )

    st.divider()

    st.markdown("## Entre em contato")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>📍 Endereço</h3>
                Rua 14 de Maio, 3200 — Centro<br>
                Campo Grande – MS<br>
                CEP 79202-333
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>📞 Contato</h3>
                (67) 3341-4444<br>
                contato@emporiodamusica.com.br
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("## Podemos ajudar você?")

    st.write(
        """
        Conheça o Kiko, nosso assistente virtual de atendimento!
        Ele pode ajudar com consultas sobre instrumentos, preços,
        disponibilidade, pedidos e políticas da loja.
        """
    )

    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.button(
            "💬 Falar com Kiko",
            on_click=open_chat,
            use_container_width=True,
            type="primary",
        )

    st.markdown(
        """
        <div class="footer">
            Empório da Música | Atendimento virtual
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# PÁGINA DE ATENDIMENTO
# ==================================================

else:

    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("💬 Atendimento com Kiko")
        st.caption("Assistente virtual da Empório da Música")

    with col2:
        st.button(
            "← Voltar à loja",
            on_click=back_to_home,
            use_container_width=True,
        )

    st.divider()

    with st.sidebar:
        st.title("🎸 Kiko")

        st.write(
            """
            Olá! Sou o assistente virtual da Empório da Música.

            Posso ajudar você com:
            - Consulta de produtos
            - Preços e disponibilidade
            - Status de pedidos
            - Trocas e devoluções
            - Políticas da loja
            """
        )

        st.divider()

        if st.button("Nova conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_history = []
            st.rerun()

    # Histórico da conversa
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada do cliente
    if prompt := st.chat_input("Digite sua mensagem..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Kiko está preparando sua resposta..."):

                try:
                    response = ask_agent(
                        user_message=prompt,
                        history=st.session_state.agent_history,
                    )

                    st.markdown(response)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    )

                    st.session_state.agent_history.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ])

                except Exception:
                    st.error(
                        "Ocorreu um erro ao processar sua mensagem. "
                        "Por favor, tente novamente."
                    )
