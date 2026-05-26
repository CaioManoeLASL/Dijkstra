import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dijkstra – Logística Fortaleza",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 2. CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background: #0d1117; }

h1, h2, h3 { color: #f0a500 !important; }

.metric-box {
    background: linear-gradient(135deg, #1a1f2e, #21293a);
    border: 1px solid #f0a50040;
    border-left: 4px solid #f0a500;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}
.metric-box .label { color: #8892a4; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
.metric-box .value { color: #f0a500; font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

.route-step {
    background: #161b27;
    border: 1px solid #2a3142;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 4px 0;
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.route-step .arrow { color: #f0a500; font-size: 16px; }

.explain-box {
    background: #0e1420;
    border: 1px solid #2a3142;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 10px 0;
    color: #8892a4;
    line-height: 1.7;
}
.explain-box strong { color: #79c0ff; }
.explain-box code {
    background: #1a2035;
    color: #f0a500;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}

.section-title {
    color: #f0a500;
    font-size: 18px;
    font-weight: 700;
    border-bottom: 1px solid #f0a50030;
    padding-bottom: 8px;
    margin-bottom: 12px;
}

.tag {
    display: inline-block;
    background: #f0a50020;
    color: #f0a500;
    border: 1px solid #f0a50060;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
}

.highlight { color: #56d364; font-weight: 600; }
.danger    { color: #f85149; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. MODELAGEM DO GRAFO – BAIRROS DE FORTALEZA
# ─────────────────────────────────────────────
# Pesos = estimativa de tempo em minutos (horário comercial)
# Fonte: estimativas baseadas em distâncias reais e trânsito típico

ARESTAS = [
    # (origem, destino, peso_minutos)
    ("Centro",        "Benfica",        8),
    ("Centro",        "Aldeota",        12),
    ("Centro",        "Fátima",         10),
    ("Centro",        "Barra do Ceará", 15),
    ("Centro",        "Parangaba",      18),
    ("Benfica",       "Fátima",         7),
    ("Benfica",       "Parangaba",      14),
    ("Benfica",       "Messejana",      25),
    ("Aldeota",       "Meireles",       8),
    ("Aldeota",       "Dionísio Torres",9),
    ("Aldeota",       "Fátima",         11),
    ("Aldeota",       "Cocó",           14),
    ("Meireles",      "Varjota",        6),
    ("Meireles",      "Mucuripe",       10),
    ("Meireles",      "Papicu",         9),
    ("Varjota",       "Dionísio Torres",7),
    ("Varjota",       "Papicu",         8),
    ("Papicu",        "Cocó",           7),
    ("Papicu",        "Cidade 2000",    10),
    ("Mucuripe",      "Cidade 2000",    8),
    ("Cocó",          "Cidade 2000",    6),
    ("Cocó",          "Maraponga",      20),
    ("Cidade 2000",   "Messejana",      18),
    ("Dionísio Torres","Aldeota",       9),
    ("Fátima",        "Parangaba",      10),
    ("Fátima",        "Mondubim",       16),
    ("Parangaba",     "Mondubim",       12),
    ("Parangaba",     "Maraponga",      14),
    ("Parangaba",     "Messejana",      22),
    ("Mondubim",      "Maraponga",      10),
    ("Mondubim",      "Barra do Ceará", 20),
    ("Maraponga",     "Messejana",      15),
    ("Messejana",     "Cidade 2000",    18),
    ("Barra do Ceará","Centro",         15),
]

BAIRROS = sorted({b for par in ARESTAS for b in par[:2]})

# Posições geográficas aproximadas (lon, lat normalizadas) para layout do grafo
POSICOES = {
    "Barra do Ceará": (0.05, 0.85),
    "Centro":         (0.30, 0.70),
    "Benfica":        (0.40, 0.65),
    "Fátima":         (0.38, 0.52),
    "Parangaba":      (0.30, 0.40),
    "Mondubim":       (0.18, 0.30),
    "Maraponga":      (0.35, 0.22),
    "Messejana":      (0.65, 0.18),
    "Aldeota":        (0.55, 0.62),
    "Dionísio Torres":(0.62, 0.55),
    "Meireles":       (0.65, 0.72),
    "Varjota":        (0.70, 0.65),
    "Papicu":         (0.78, 0.60),
    "Cocó":           (0.72, 0.48),
    "Cidade 2000":    (0.85, 0.50),
    "Mucuripe":       (0.80, 0.75),
}


def construir_grafo(arestas):
    """
    Constrói lista de adjacência (dict) a partir das arestas.
    Grafo não-direcionado: cada aresta vira dois sentidos.
    """
    grafo = defaultdict(list)
    for origem, destino, peso in arestas:
        grafo[origem].append((destino, peso))
        grafo[destino].append((origem, peso))   # bidirecional
    return grafo


# ─────────────────────────────────────────────
# 4. ALGORITMO DE DIJKSTRA
# ─────────────────────────────────────────────
def dijkstra(grafo, origem, destino):
    """
    Implementação do algoritmo de Dijkstra com fila de prioridade (min-heap).

    Retorna:
        caminho  : lista de vértices do menor caminho
        custo    : custo total (minutos)
        iteracoes: log passo-a-passo para fins didáticos
    """
    # dist[v] = menor distância conhecida até v
    dist = {v: float("inf") for v in grafo}
    dist[origem] = 0

    # anterior[v] = vértice que veio antes de v no caminho ótimo
    anterior = {v: None for v in grafo}

    # Fila de prioridade: (custo_acumulado, vertice)
    heap = [(0, origem)]

    visitados = set()
    iteracoes = []   # log didático

    while heap:
        custo_atual, u = heapq.heappop(heap)

        if u in visitados:
            continue
        visitados.add(u)

        iteracoes.append({
            "vertice":   u,
            "custo":     custo_atual,
            "visitados": list(visitados),
        })

        if u == destino:
            break

        for vizinho, peso in grafo[u]:
            if vizinho in visitados:
                continue
            nova_dist = dist[u] + peso
            if nova_dist < dist[vizinho]:
                dist[vizinho] = nova_dist
                anterior[vizinho] = u
                heapq.heappush(heap, (nova_dist, vizinho))

    # Reconstruir caminho
    caminho = []
    v = destino
    while v is not None:
        caminho.append(v)
        v = anterior[v]
    caminho.reverse()

    custo_total = dist[destino]
    return caminho, custo_total, iteracoes


# ─────────────────────────────────────────────
# 5. VISUALIZAÇÃO DO GRAFO (matplotlib)
# ─────────────────────────────────────────────
def desenhar_grafo(grafo_nx, caminho, origem, destino, pos):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Conjuntos para colorir
    arestas_caminho = set()
    if len(caminho) > 1:
        for i in range(len(caminho) - 1):
            arestas_caminho.add((caminho[i], caminho[i + 1]))
            arestas_caminho.add((caminho[i + 1], caminho[i]))

    # Arestas normais
    arestas_normais = [(u, v) for u, v in grafo_nx.edges()
                       if (u, v) not in arestas_caminho]
    nx.draw_networkx_edges(
        grafo_nx, pos,
        edgelist=arestas_normais,
        edge_color="#2a3142",
        width=1.5,
        ax=ax,
        alpha=0.8,
    )

    # Arestas do caminho ótimo
    if arestas_caminho:
        arestas_dest = [(u, v) for u, v in grafo_nx.edges() if (u, v) in arestas_caminho]
        nx.draw_networkx_edges(
            grafo_nx, pos,
            edgelist=arestas_dest,
            edge_color="#f0a500",
            width=4,
            ax=ax,
            alpha=1.0,
        )

    # Cores dos nós
    cores_nos = []
    for n in grafo_nx.nodes():
        if n == origem:
            cores_nos.append("#56d364")
        elif n == destino:
            cores_nos.append("#f85149")
        elif n in caminho:
            cores_nos.append("#f0a500")
        else:
            cores_nos.append("#21293a")

    nx.draw_networkx_nodes(
        grafo_nx, pos,
        node_color=cores_nos,
        node_size=600,
        ax=ax,
    )

    # Labels dos nós (abreviados)
    labels = {n: n.replace(" ", "\n") for n in grafo_nx.nodes()}
    nx.draw_networkx_labels(
        grafo_nx, pos,
        labels=labels,
        font_color="#e6edf3",
        font_size=7,
        font_family="monospace",
        ax=ax,
    )

    # Pesos das arestas
    pesos = nx.get_edge_attributes(grafo_nx, "weight")
    nx.draw_networkx_edge_labels(
        grafo_nx, pos,
        edge_labels=pesos,
        font_color="#8892a4",
        font_size=7,
        ax=ax,
        bbox=dict(boxstyle="round,pad=0.15", fc="#0d1117", ec="none", alpha=0.8),
    )

    # Legenda
    legenda = [
        mpatches.Patch(color="#56d364", label="Origem (CD)"),
        mpatches.Patch(color="#f85149", label="Destino"),
        mpatches.Patch(color="#f0a500", label="Caminho ótimo"),
        mpatches.Patch(color="#21293a", label="Outros bairros"),
    ]
    ax.legend(handles=legenda, loc="lower left",
              facecolor="#161b27", edgecolor="#2a3142",
              labelcolor="#c9d1d9", fontsize=9)

    ax.set_title("Grafo de Entregas – Fortaleza/CE",
                 color="#f0a500", fontsize=14, pad=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 6. INTERFACE STREAMLIT
# ─────────────────────────────────────────────
grafo_adj = construir_grafo(ARESTAS)

# NetworkX para visualização
G_nx = nx.Graph()
for origem_a, destino_a, peso in ARESTAS:
    G_nx.add_edge(origem_a, destino_a, weight=peso)

# ── Cabeçalho ──────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:2.2rem; margin-bottom:4px;'>
🚚 Roteamento de Entregas – Fortaleza
</h1>
<p style='text-align:center; color:#8892a4; font-size:15px; margin-top:0;'>
Algoritmo de Dijkstra aplicado à logística urbana
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuração da Entrega")

    origem_sel = st.selectbox(
        "📦 Centro de Distribuição (origem)",
        BAIRROS,
        index=BAIRROS.index("Centro"),
    )
    destino_sel = st.selectbox(
        "📍 Endereço do Cliente (destino)",
        BAIRROS,
        index=BAIRROS.index("Messejana"),
    )

    calcular = st.button("🔍 Calcular Rota Ótima", use_container_width=True, type="primary")

    st.divider()
    st.markdown("### 📊 Informações do Grafo")
    st.markdown(f"""
<div class='explain-box'>
<strong>Vértices:</strong> {len(BAIRROS)} bairros<br>
<strong>Arestas:</strong> {len(ARESTAS)} conexões<br>
<strong>Pesos:</strong> tempo em minutos<br>
<strong>Tipo:</strong> não-direcionado
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🗺️ Bairros Modelados")
    for b in BAIRROS:
        st.markdown(f"<span class='tag'>{b}</span>", unsafe_allow_html=True)


# ── Abas principais ────────────────────────────
aba1, aba2, aba3, aba4 = st.tabs([
    "🗺️ Grafo & Rota",
    "📋 Passo a Passo",
    "💻 Código Explicado",
    "📚 Teoria",
])

# ── ABA 1: Grafo & Resultado ───────────────────
with aba1:
    if origem_sel == destino_sel:
        st.warning("⚠️ Origem e destino são iguais. Selecione pontos diferentes.")
    else:
        caminho_res, custo_res, iters = dijkstra(grafo_adj, origem_sel, destino_sel)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
<div class='metric-box'>
  <div class='label'>⏱ Tempo Estimado</div>
  <div class='value'>{custo_res} min</div>
</div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
<div class='metric-box'>
  <div class='label'>📍 Paradas na Rota</div>
  <div class='value'>{len(caminho_res)}</div>
</div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
<div class='metric-box'>
  <div class='label'>🔄 Vértices Visitados</div>
  <div class='value'>{len(iters)}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("#### 🛣️ Rota Encontrada")
        for i, bairro in enumerate(caminho_res):
            icone = "🟢" if bairro == origem_sel else ("🔴" if bairro == destino_sel else "🟡")
            if i < len(caminho_res) - 1:
                # calcula custo da aresta
                peso_trecho = next(
                    (p for viz, p in grafo_adj[bairro] if viz == caminho_res[i + 1]), "?")
                st.markdown(f"""
<div class='route-step'>
  {icone} <strong>{bairro}</strong>
  <span class='arrow'>──── {peso_trecho} min ────▶</span>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class='route-step'>
  {icone} <strong>{bairro}</strong> &nbsp;
  <span class='highlight'> ✓ DESTINO</span>
</div>""", unsafe_allow_html=True)

        st.markdown("#### 🗺️ Visualização do Grafo")
        fig = desenhar_grafo(G_nx, caminho_res, origem_sel, destino_sel, POSICOES)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── ABA 2: Passo a passo ───────────────────────
with aba2:
    if origem_sel == destino_sel:
        st.warning("⚠️ Selecione origem e destino diferentes.")
    else:
        caminho_res2, custo_res2, iters2 = dijkstra(grafo_adj, origem_sel, destino_sel)

        st.markdown("### 🔬 Execução Iteração por Iteração")
        st.markdown("""
<div class='explain-box'>
A tabela abaixo mostra cada vez que o algoritmo <strong>extrai o vértice de menor custo</strong>
da fila de prioridade e o marca como <strong>visitado</strong>.
</div>
""", unsafe_allow_html=True)

        for i, it in enumerate(iters2):
            cor = "#56d364" if it["vertice"] == origem_sel else (
                  "#f85149" if it["vertice"] == destino_sel else "#f0a500")
            visitados_str = ", ".join(it["visitados"])
            st.markdown(f"""
<div class='route-step' style='border-left:3px solid {cor}; margin:6px 0;'>
  <span style='color:#8892a4; min-width:30px;'>#{i+1}</span>
  <span style='color:{cor}; font-weight:700; min-width:160px;'>{it["vertice"]}</span>
  <span style='color:#8892a4; font-size:12px;'>custo acumulado: </span>
  <span style='color:#79c0ff; font-weight:700; min-width:60px;'>{it["custo"]} min</span>
  <span style='color:#444d56; font-size:11px;'>visitados: {visitados_str}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class='metric-box' style='margin-top:20px;'>
  <div class='label'>✅ Resultado Final</div>
  <div class='value'>{' → '.join(caminho_res2)} &nbsp;|&nbsp; {custo_res2} min</div>
</div>
""", unsafe_allow_html=True)


# ── ABA 3: Código comentado ────────────────────
with aba3:
    st.markdown("### 💻 Estruturas de Dados")
    st.markdown("""
<div class='explain-box'>
<strong>Lista de adjacência</strong> — escolhemos um <code>defaultdict(list)</code> do Python.
Cada chave é um bairro; o valor é uma lista de tuplas <code>(vizinho, peso)</code>.
É eficiente para grafos esparsos (poucos vizinhos por vértice), exatamente o nosso caso.
</div>
""", unsafe_allow_html=True)

    st.code("""
# Construção do grafo (lista de adjacência)
from collections import defaultdict

def construir_grafo(arestas):
    grafo = defaultdict(list)          # dict de listas
    for origem, destino, peso in arestas:
        grafo[origem].append((destino, peso))
        grafo[destino].append((origem, peso))  # grafo não-direcionado
    return grafo
""", language="python")

    st.markdown("### 🔑 Algoritmo de Dijkstra – Linha a Linha")
    st.markdown("""
<div class='explain-box'>
Usamos um <strong>min-heap</strong> (<code>heapq</code>) como fila de prioridade.
O heap sempre mantém o vértice de <em>menor custo acumulado</em> no topo — isso garante
que quando um vértice é extraído, seu caminho já é ótimo.
</div>
""", unsafe_allow_html=True)

    st.code("""
import heapq

def dijkstra(grafo, origem, destino):
    # Inicializa todas as distâncias como infinito
    dist     = {v: float("inf") for v in grafo}
    dist[origem] = 0                          # origem custa 0

    anterior = {v: None for v in grafo}       # para reconstruir caminho
    heap     = [(0, origem)]                  # fila de prioridade (custo, vértice)
    visitados = set()

    while heap:
        custo_atual, u = heapq.heappop(heap)  # extrai o de menor custo

        if u in visitados:                    # já processado? ignora
            continue
        visitados.add(u)

        if u == destino:                      # chegou! para cedo
            break

        for vizinho, peso in grafo[u]:
            if vizinho in visitados:
                continue
            nova_dist = dist[u] + peso        # tenta "relaxar" a aresta
            if nova_dist < dist[vizinho]:
                dist[vizinho]     = nova_dist
                anterior[vizinho] = u
                heapq.heappush(heap, (nova_dist, vizinho))

    # Reconstrói o caminho do destino até a origem (reverso)
    caminho, v = [], destino
    while v is not None:
        caminho.append(v)
        v = anterior[v]
    caminho.reverse()

    return caminho, dist[destino]
""", language="python")

    st.markdown("### 📦 Modelagem das Arestas")
    st.code("""
ARESTAS = [
    # (bairro_a, bairro_b, tempo_minutos)
    ("Centro",   "Benfica",  8),
    ("Centro",   "Aldeota", 12),
    ("Aldeota",  "Meireles", 8),
    ("Meireles", "Papicu",   9),
    # ... (16 bairros, 34 conexões no total)
]
""", language="python")


# ── ABA 4: Teoria ──────────────────────────────
with aba4:
    st.markdown("### 📚 Por que Dijkstra?")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
<div class='explain-box'>
<strong>Complexidade de tempo</strong><br>
Com min-heap (heapq): <code>O((V + E) log V)</code><br>
<ul>
  <li><strong>V</strong> = número de vértices (bairros)</li>
  <li><strong>E</strong> = número de arestas (ruas)</li>
</ul>
Para nosso grafo: V=16, E=34 → muito rápido!
</div>
""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
<div class='explain-box'>
<strong>Pré-requisitos</strong><br>
✅ Arestas com peso <strong>não-negativo</strong><br>
✅ Grafo direcionado ou não-direcionado<br>
❌ NÃO funciona com pesos negativos<br>
→ Para isso existe o algoritmo de <em>Bellman-Ford</em>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🔄 Invariante do Algoritmo")
    st.markdown("""
<div class='explain-box'>
A cada iteração, o algoritmo mantém a seguinte garantia:<br><br>
<strong>"Quando um vértice é removido da fila de prioridade, a distância registrada para
ele é definitivamente a menor possível."</strong><br><br>
Isso funciona porque os pesos são não-negativos: nunca vai aparecer um caminho mais
curto para um vértice já visitado. O heap garante que sempre processamos o vizinho
mais promissor primeiro (<em>greedy</em>).
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🗃️ Comparação com Outros Algoritmos")
    dados_comp = {
        "Algoritmo":     ["Dijkstra", "Bellman-Ford", "Floyd-Warshall", "A*"],
        "Complexidade":  ["O((V+E)logV)", "O(V·E)", "O(V³)", "O(E logV)"],
        "Pesos neg.":    ["❌", "✅", "✅", "❌"],
        "Todos pares":   ["❌", "❌", "✅", "❌"],
        "Heurística":    ["❌", "❌", "❌", "✅"],
    }
    import pandas as pd
    df = pd.DataFrame(dados_comp)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 🏙️ Relação com o Cenário Real")
    st.markdown("""
<div class='explain-box'>
Neste projeto modelamos Fortaleza de forma simplificada, mas a abordagem reflete
sistemas reais:<br><br>
• <strong>Google Maps / Waze</strong> usam variantes de Dijkstra com heurísticas (A*)
e grafos com milhões de vértices.<br>
• Os <strong>pesos</strong> reais variam em tempo real (trânsito, obras, acidentes).<br>
• Sistemas de logística como iFood e Amazon Logistics combinam Dijkstra com
algoritmos de <em>roteirização de veículos (VRP)</em> para múltiplas entregas.
</div>
""", unsafe_allow_html=True)