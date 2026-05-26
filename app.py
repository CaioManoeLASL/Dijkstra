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
# 5. VISUALIZAÇÃO DO GRAFO (matplotlib – premium)
# ─────────────────────────────────────────────
def _cor_por_peso(peso, pmin=6, pmax=25):
    """Interpola de verde (#56d364) a vermelho (#f85149) conforme o peso."""
    t = np.clip((peso - pmin) / (pmax - pmin), 0, 1)
    r = int(0x56 + t * (0xf8 - 0x56))
    g = int(0xd3 + t * (0x51 - 0xd3))
    b = int(0x64 + t * (0x49 - 0x64))
    return f"#{r:02x}{g:02x}{b:02x}"


def _glow(ax, x, y, radius, cor, camadas=5):
    """Desenha halo de brilho em torno de um ponto."""
    for i in range(camadas, 0, -1):
        alpha = 0.06 * (camadas - i + 1)
        r = radius * (1 + i * 0.55)
        circle = plt.Circle((x, y), r, color=cor, alpha=alpha, zorder=3)
        ax.add_patch(circle)


def desenhar_grafo(grafo_nx, caminho, origem, destino, pos):
    # ── Figura ────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")

    # limites do canvas
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.12, max(xs) + 0.12)
    ax.set_ylim(min(ys) - 0.12, max(ys) + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Grade de fundo suave ──────────────────
    for gx in np.linspace(min(xs) - 0.12, max(xs) + 0.12, 18):
        ax.axvline(gx, color="#1a2035", linewidth=0.4, zorder=0)
    for gy in np.linspace(min(ys) - 0.12, max(ys) + 0.12, 12):
        ax.axhline(gy, color="#1a2035", linewidth=0.4, zorder=0)

    # ── Conjuntos do caminho ──────────────────
    caminho_set   = set(caminho)
    arestas_rota  = set()
    if len(caminho) > 1:
        for i in range(len(caminho) - 1):
            arestas_rota.add((caminho[i], caminho[i + 1]))
            arestas_rota.add((caminho[i + 1], caminho[i]))

    pesos_dict = nx.get_edge_attributes(grafo_nx, "weight")

    # ── 1. Arestas fora da rota ───────────────
    for u, v in grafo_nx.edges():
        if (u, v) in arestas_rota:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        peso = pesos_dict.get((u, v), pesos_dict.get((v, u), 10))
        cor = _cor_por_peso(peso)
        ax.plot([x0, x1], [y0, y1], color=cor, linewidth=1.2,
                alpha=0.30, solid_capstyle="round", zorder=1)

        # rótulo de peso pequeno
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, str(peso),
                fontsize=6, color="#3d4a5c", ha="center", va="center",
                fontfamily="monospace", zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", fc="#080c14", ec="none", alpha=0.75))

    # ── 2. Glow difuso na rota (várias camadas) ──
    if arestas_rota:
        for u, v in grafo_nx.edges():
            if (u, v) not in arestas_rota:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            for largura, alfa in [(18, 0.04), (12, 0.07), (7, 0.12), (4, 0.25)]:
                ax.plot([x0, x1], [y0, y1], color="#f0a500",
                        linewidth=largura, alpha=alfa,
                        solid_capstyle="round", zorder=3)

        # ── 3. Linha da rota (núcleo nítido) ──────
        for u, v in grafo_nx.edges():
            if (u, v) not in arestas_rota:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            peso = pesos_dict.get((u, v), pesos_dict.get((v, u), 10))
            ax.plot([x0, x1], [y0, y1], color="#f0a500",
                    linewidth=2.8, alpha=1.0,
                    solid_capstyle="round", zorder=4)
            # rótulo de peso na rota em destaque
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"{peso} min",
                    fontsize=7.5, color="#f0a500", ha="center", va="center",
                    fontfamily="monospace", fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#0d0f1a",
                              ec="#f0a50060", linewidth=0.8, alpha=0.95))

    # ── 4. Nós ────────────────────────────────
    NODE_R = 0.022   # raio visual (em unidades de dados)

    for n in grafo_nx.nodes():
        x, y = pos[n]
        em_rota = n in caminho_set

        if n == origem:
            cor_fill  = "#56d364"
            cor_anel  = "#9be9a8"
            cor_glow  = "#56d364"
            zord      = 9
        elif n == destino:
            cor_fill  = "#f85149"
            cor_anel  = "#ff8183"
            cor_glow  = "#f85149"
            zord      = 9
        elif em_rota:
            cor_fill  = "#f0a500"
            cor_anel  = "#ffd166"
            cor_glow  = "#f0a500"
            zord      = 8
        else:
            cor_fill  = "#1a2235"
            cor_anel  = "#2d3a50"
            cor_glow  = "#3d4f6e"
            zord      = 6

        # glow
        if em_rota or n in (origem, destino):
            _glow(ax, x, y, NODE_R, cor_glow, camadas=6)

        # sombra do nó
        sombra = plt.Circle((x + 0.004, y - 0.004), NODE_R,
                             color="#000000", alpha=0.45, zorder=zord - 1)
        ax.add_patch(sombra)

        # anel externo
        anel = plt.Circle((x, y), NODE_R * 1.22, color=cor_anel,
                          alpha=0.55 if em_rota or n in (origem, destino) else 0.25,
                          zorder=zord)
        ax.add_patch(anel)

        # círculo principal
        circulo = plt.Circle((x, y), NODE_R, color=cor_fill,
                             alpha=1.0, zorder=zord + 1)
        ax.add_patch(circulo)

    # ── 5. Rótulos dos nós ───────────────────
    # Direção de offset para evitar sobreposição com arestas
    OFFSETS = {
        "Barra do Ceará": (-0.065, +0.028),
        "Centro":         (-0.058,  0.000),
        "Benfica":        (+0.000, +0.030),
        "Fátima":         (+0.058,  0.000),
        "Parangaba":      (-0.060,  0.000),
        "Mondubim":       (-0.058,  0.000),
        "Maraponga":      (+0.000, -0.030),
        "Messejana":      (+0.000, -0.030),
        "Aldeota":        (+0.000, +0.030),
        "Dionísio Torres":(+0.000, -0.032),
        "Meireles":       (+0.000, +0.030),
        "Varjota":        (+0.055, +0.000),
        "Papicu":         (+0.055, +0.000),
        "Cocó":           (+0.050, +0.000),
        "Cidade 2000":    (+0.068, +0.000),
        "Mucuripe":       (+0.055, +0.000),
    }

    for n in grafo_nx.nodes():
        x, y = pos[n]
        ox, oy = OFFSETS.get(n, (0.0, 0.032))
        em_rota = n in caminho_set
        is_endpoint = n in (origem, destino)

        cor_txt  = "#ffffff" if is_endpoint else ("#ffd166" if em_rota else "#7a8899")
        fontsize = 8.5 if is_endpoint else (8.0 if em_rota else 7.0)
        fw       = "bold" if (em_rota or is_endpoint) else "normal"

        ax.text(x + ox, y + oy, n,
                fontsize=fontsize, color=cor_txt, ha="center", va="center",
                fontfamily="monospace", fontweight=fw, zorder=12,
                bbox=dict(boxstyle="round,pad=0.22",
                          fc="#080c14" if not is_endpoint else "#0d1117",
                          ec=("#f0a500" if em_rota and not is_endpoint
                              else ("#56d364" if n == origem
                              else ("#f85149" if n == destino else "none"))),
                          linewidth=0.9, alpha=0.92))

    # ── 6. Ícones de origem/destino ──────────
    ox_orig, oy_orig = pos[origem]
    ox_dest, oy_dest = pos[destino]
    ax.text(ox_orig, oy_orig, "▲",
            fontsize=9, color="#080c14", ha="center", va="center",
            fontweight="bold", zorder=13)
    ax.text(ox_dest, oy_dest, "★",
            fontsize=9, color="#080c14", ha="center", va="center",
            fontweight="bold", zorder=13)

    # ── 7. Legenda ────────────────────────────
    leg_items = [
        mpatches.Patch(facecolor="#56d364", edgecolor="#9be9a8", label="▲ Origem (CD)"),
        mpatches.Patch(facecolor="#f85149", edgecolor="#ff8183", label="★ Destino"),
        mpatches.Patch(facecolor="#f0a500", edgecolor="#ffd166", label="  Rota ótima"),
        mpatches.Patch(facecolor="#1a2235", edgecolor="#2d3a50", label="  Outros bairros"),
    ]
    leg = ax.legend(
        handles=leg_items,
        loc="lower left",
        facecolor="#0d1117",
        edgecolor="#2a3142",
        labelcolor="#c9d1d9",
        fontsize=9,
        framealpha=0.95,
        borderpad=1.0,
        handlelength=1.4,
    )
    leg.get_frame().set_linewidth(0.8)

    # ── 8. Barra de escala de cor (pesos) ────
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("peso", ["#56d364", "#f0a500", "#f85149"])
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=6, vmax=25))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.025, pad=0.01, aspect=30,
                        location="bottom")
    cbar.set_label("Tempo da aresta (min)", color="#8892a4", fontsize=8)
    cbar.ax.xaxis.set_tick_params(color="#8892a4", labelsize=7, labelcolor="#8892a4")
    cbar.outline.set_edgecolor("#2a3142")

    # ── 9. Título ─────────────────────────────
    ax.set_title(
        "Grafo de Entregas — Fortaleza / CE",
        color="#f0a500", fontsize=15, pad=16,
        fontweight="bold", fontfamily="monospace",
    )

    plt.tight_layout(pad=1.5)
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
aba1, aba2 = st.tabs([
    "🗺️ Grafo & Rota",
    "📋 Passo a Passo",
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