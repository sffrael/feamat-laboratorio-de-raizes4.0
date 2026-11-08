import ast
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FEAMAT | Laboratório de Raízes", page_icon="📘",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:2rem;padding-bottom:3rem}
.feamat-hero{background:linear-gradient(115deg,#15346f 0%,#2457c5 65%,#3978e8 100%);
color:white;padding:2rem 2.3rem;border-radius:22px;margin-bottom:1.5rem;
box-shadow:0 12px 28px rgba(25,65,140,.16)}
.feamat-hero .eyebrow{font-size:.78rem;letter-spacing:.14em;font-weight:800;opacity:.85}
.feamat-hero h1{color:white;font-size:2.7rem;margin:.6rem 0}
.feamat-hero p{color:#eaf1ff;font-size:1.05rem;margin:0}
.info-panel{border-left:5px solid #3978e8;border-radius:10px;padding:1rem 1.1rem;
background:rgba(55,120,232,.09)}
div[data-testid="stMetric"]{border:1px solid rgba(120,140,170,.25);border-radius:12px;
padding:.8rem 1rem;background:rgba(120,140,170,.06)}
</style>
<div class="feamat-hero">
 <div class="eyebrow">LABORATÓRIO VIRTUAL DE CÁLCULO NUMÉRICO</div>
 <h1>FEAMAT</h1>
 <p>Laboratório de Raízes · Por trás dos resultados: como os algoritmos numéricos encontram raízes de funções</p>
</div>
""", unsafe_allow_html=True)

PRESETS = {
    "x² − 2": {"expr":"x^2 - 2","a":1.0,"b":2.0,"x0":1.5,"desc":"Exemplo clássico: aproximação de √2."},
    "x³ − x − 2": {"expr":"x^3 - x - 2","a":1.0,"b":2.0,"x0":1.5,"desc":"Equação cúbica com uma raiz no intervalo [1, 2]."},
    "cos(x) − x": {"expr":"cos(x) - x","a":0.0,"b":1.0,"x0":0.5,"desc":"Equação transcendental com raiz entre 0 e 1."},
    "eˣ − 3": {"expr":"exp(x) - 3","a":0.0,"b":2.0,"x0":1.0,"desc":"Função exponencial; raiz igual a ln(3)."},
    "ln(x) − 2": {"expr":"log(x) - 2","a":7.0,"b":8.0,"x0":7.5,"desc":"Função logarítmica, definida para x positivo."},
    "√x − 3": {"expr":"sqrt(x) - 3","a":8.0,"b":10.0,"x0":9.0,"desc":"Função raiz quadrada; raiz igual a 9."},
    "Função personalizada": {"expr":"x^2 - 2","a":1.0,"b":2.0,"x0":1.5,"desc":"Digite uma função e configure os parâmetros."}
}

FUNCS = {"sin":math.sin,"sen":math.sin,"cos":math.cos,"tan":math.tan,"tg":math.tan,
         "exp":math.exp,"sqrt":math.sqrt,"log":math.log,"ln":math.log,"abs":abs}
CONSTS = {"pi":math.pi,"e":math.e}

def eval_ast(node, x):
    if isinstance(node, ast.Expression): return eval_ast(node.body, x)
    if isinstance(node, ast.Constant) and isinstance(node.value,(int,float)): return float(node.value)
    if isinstance(node, ast.Name):
        if node.id=="x": return float(x)
        if node.id in CONSTS: return CONSTS[node.id]
        raise ValueError(f"Símbolo não permitido: {node.id}")
    if isinstance(node, ast.UnaryOp):
        v=eval_ast(node.operand,x)
        if isinstance(node.op,ast.UAdd): return v
        if isinstance(node.op,ast.USub): return -v
    if isinstance(node, ast.BinOp):
        a,b=eval_ast(node.left,x),eval_ast(node.right,x)
        if isinstance(node.op,ast.Add): return a+b
        if isinstance(node.op,ast.Sub): return a-b
        if isinstance(node.op,ast.Mult): return a*b
        if isinstance(node.op,ast.Div): return a/b
        if isinstance(node.op,ast.Pow): return a**b
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
        if node.func.id not in FUNCS or len(node.args)!=1:
            raise ValueError("Função não permitida.")
        return FUNCS[node.func.id](eval_ast(node.args[0],x))
    raise ValueError("Expressão inválida. Use x, operações e funções permitidas.")

def compile_f(expr):
    tree=ast.parse(expr.replace("^","**").replace("π","pi"),mode="eval")
    allowed=(ast.Expression,ast.Constant,ast.Name,ast.UnaryOp,ast.UAdd,ast.USub,
             ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Call,ast.Load)
    for n in ast.walk(tree):
        if not isinstance(n,allowed): raise ValueError("A expressão contém operação não permitida.")
        if isinstance(n,ast.Name) and n.id not in ({"x"}|set(CONSTS)|set(FUNCS)):
            raise ValueError(f"Símbolo não reconhecido: {n.id}")
    def f(x):
        y=float(eval_ast(tree,x))
        if not math.isfinite(y): raise ValueError("A função produziu valor não finito.")
        return y
    return f

def derivative(f,x):
    h=1e-6*max(1.0,abs(x))
    return (f(x+h)-f(x-h))/(2*h)

def sign(v):
    return "Positivo (+)" if v>0 else "Negativo (−)" if v<0 else "Zero (0)"
def symbol(v): return "+" if v>0 else "−" if v<0 else "0"

def bisection(f,a,b,tol,maxit):
    fa,fb=f(a),f(b)
    if fa==0: return a,0,0.0,[],"O extremo a já é raiz."
    if fb==0: return b,0,0.0,[],"O extremo b já é raiz."
    if fa*fb>0: raise ValueError("f(a) e f(b) têm o mesmo sinal. Escolha outro intervalo.")
    rows=[]
    for i in range(1,maxit+1):
        m=(a+b)/2; fm=f(m)
        width=(b-a)/2
        if fa*fm<0:
            na,nb=a,m
            decision="[a, m] mantido: f(a) e f(m) têm sinais opostos."
        elif fm*fb<0:
            na,nb=m,b
            decision="[m, b] mantido: f(m) e f(b) têm sinais opostos."
        else:
            na,nb=m,m
            decision="O ponto médio é uma raiz."
        rows.append({"Iteração":i,"a":a,"f(a)":fa,"Sinal f(a)":symbol(fa),
                     "b":b,"f(b)":fb,"Sinal f(b)":symbol(fb),
                     "m = (a+b)/2":m,"f(m)":fm,"Sinal f(m)":symbol(fm),
                     "Erro estimado (metade do intervalo)":width,
                     "Intervalo mantido":f"[{na:.10g}, {nb:.10g}]","Decisão":decision})
        if abs(fm)<=tol or width<=tol: break
        a,b=na,nb; fa,fb=f(a),f(b)
    root=rows[-1]["m = (a+b)/2"]
    return root,len(rows),abs(f(root)),rows,"Critério de parada atingido."

def newton(f,x0,tol,maxit):
    rows=[]; x=x0
    for i in range(1,maxit+1):
        fx=f(x); d=derivative(f,x)
        if abs(d)<1e-14: raise ValueError(f"Derivada numericamente nula em x={x:.10g}.")
        xn=x-fx/d; err=abs(xn-x)
        rows.append({"Iteração":i,"xₙ":x,"f(xₙ)":fx,"f'(xₙ) aproximada":d,
                     "xₙ₊₁":xn,"Erro |xₙ₊₁ − xₙ|":err})
        x=xn
        if err<=tol or abs(f(x))<=tol: break
    return x,len(rows),abs(f(x)),rows,"Critério de parada atingido."

def preset_callback():
    p=PRESETS[st.session_state.preset]
    st.session_state.expr=p["expr"]
    st.session_state.a=p["a"]; st.session_state.b=p["b"]; st.session_state.x0=p["x0"]

if "preset" not in st.session_state: st.session_state.preset="x² − 2"
if "expr" not in st.session_state:
    q=PRESETS[st.session_state.preset]
    st.session_state.expr=q["expr"]; st.session_state.a=q["a"]; st.session_state.b=q["b"]; st.session_state.x0=q["x0"]

st.sidebar.title("Experimento")
st.sidebar.selectbox("Funções predefinidas",list(PRESETS),key="preset",on_change=preset_callback)
st.sidebar.caption(PRESETS[st.session_state.preset]["desc"])
st.sidebar.text_input("Função f(x)",key="expr",help="Use ^ ou ** para potência.")
st.sidebar.caption("Funções: sin, sen, cos, tan, tg, exp, sqrt, log, ln, abs. Constantes: pi, e.")
st.sidebar.subheader("Parâmetros numéricos")
c1,c2=st.sidebar.columns(2)
with c1: st.number_input("Extremo a",key="a",step=.1,format="%.8f")
with c2: st.number_input("Extremo b",key="b",step=.1,format="%.8f")
st.sidebar.number_input("Valor inicial x₀ (Newton)",key="x0",step=.1,format="%.8f")
tol=st.sidebar.select_slider("Precisão (tolerância)",options=[1e-2,1e-3,1e-4,1e-5,1e-6,1e-8],value=1e-6,format_func=lambda v:f"{v:g}")
maxit=st.sidebar.number_input("Número máximo de iterações",5,200,50,5)
run=st.sidebar.button("▶ Executar comparação",type="primary",use_container_width=True)
if run:
    st.session_state["experiment_executed"] = True

col1,col2=st.columns([1.35,1])
with col1:
    st.subheader("Investigue como os métodos encontram uma raiz")
    st.write("Acompanhe as aproximações, compare os sinais e observe a evolução do erro em cada método.")
with col2:
    st.markdown("<div class='info-panel'><b>Como começar</b><br>1. Escolha uma função.<br>2. Confira os parâmetros.<br>3. Execute a comparação e explore os gráficos.</div>",unsafe_allow_html=True)

if st.session_state.get("experiment_executed", False):
    try:
        f=compile_f(st.session_state.expr)
        a=float(st.session_state.a); b=float(st.session_state.b); x0=float(st.session_state.x0)
        if a>=b: st.error("O extremo a precisa ser menor que b."); st.stop()
        fa,fb=f(a),f(b)
        st.markdown("---")
        st.subheader("1. Análise inicial dos sinais — Bisseção")
        c1,c2,c3=st.columns(3)
        c1.metric("f(a)",f"{fa:.10g}",sign(fa))
        c2.metric("f(b)",f"{fb:.10g}",sign(fb))
        if fa*fb<0:
            c3.metric("Comparação","Sinais opostos","Intervalo válido")
            st.success("Há mudança de sinal. Para uma função contínua, existe pelo menos uma raiz em [a,b].")
        elif fa==0 or fb==0:
            c3.metric("Comparação","Extremo é raiz","Raiz encontrada")
        else:
            c3.metric("Comparação","Mesmo sinal","Intervalo inválido")
            st.warning("A Bisseção não pode começar: f(a) e f(b) têm o mesmo sinal.")

        bis=None; new=None
        if fa*fb<=0:
            bis=bisection(f,a,b,tol,int(maxit))
            rb,ib,eb,rowsb,msgb=bis
            st.subheader("2. Método da Bisseção — tabela de iterações")
            st.caption("A tabela mostra f(a), f(b), f(m), os respectivos sinais e a decisão de intervalo em cada passo.")
            dfb=pd.DataFrame(rowsb)
            st.dataframe(dfb,use_container_width=True,hide_index=True)
            if rowsb:
                st.info(rowsb[-1]["Decisão"])
        st.subheader("3. Método de Newton-Raphson — tabela de iterações")
        try:
            new=newton(f,x0,tol,int(maxit))
            rn,in_,en,rowsn,msgn=new
            dfn=pd.DataFrame(rowsn)
            st.dataframe(dfn,use_container_width=True,hide_index=True)
        except Exception as e:
            rowsn=[]; rn=in_=en=None
            st.error(f"Newton-Raphson: {e}")

        st.markdown("---")
        st.subheader("4. Gráfico interativo da função")
        st.caption("Passe o cursor sobre a curva para consultar coordenadas. Use zoom, arraste e os controles do gráfico para explorar.")
        # Amplitude do gráfico ajustada aos dados e aos resultados.
        candidates=[a,b,x0]
        if bis: candidates.append(rb)
        if new and rn is not None: candidates.append(rn)
        xmin=min(candidates)-1; xmax=max(candidates)+1
        if xmax-xmin<2: xmin-=1; xmax+=1
        xx=np.linspace(xmin,xmax,650)
        yy=[]
        for xv in xx:
            try:
                y=f(float(xv)); yy.append(y if abs(y)<1e7 else None)
            except Exception: yy.append(None)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=xx,y=yy,mode="lines",name="f(x)",line=dict(width=3)))
        fig.add_hline(y=0,line_dash="dash",line_color="gray")
        if bis and rowsb:
            chosen_b=st.slider("Iteração da Bisseção para visualizar",1,len(rowsb),len(rowsb),key="bisection_graph_step")
            br=rowsb[chosen_b-1]
            m=br["m = (a+b)/2"]
            fig.add_trace(go.Scatter(x=[br["a"],br["b"]],y=[br["f(a)"],br["f(b)"]],
                mode="markers+text",text=["a","b"],textposition="top center",name="Extremos atuais",
                marker=dict(size=11,symbol="circle")))
            fig.add_trace(go.Scatter(x=[m],y=[f(m)],mode="markers+text",text=["m"],textposition="top center",
                name="Ponto médio",marker=dict(size=13,symbol="diamond")))
            fig.add_vrect(x0=br["a"],x1=br["b"],fillcolor="royalblue",opacity=.08,line_width=0)
        if new and rowsn:
            chosen_n=st.slider("Iteração de Newton-Raphson para visualizar",1,len(rowsn),len(rowsn),key="newton_graph_step")
            nr=rowsn[chosen_n-1]
            xn=nr["xₙ"]; yn=f(xn); slope=nr["f'(xₙ) aproximada"]
            tangent_x=np.linspace(xn-0.8,xn+0.8,60)
            tangent_y=yn+slope*(tangent_x-xn)
            fig.add_trace(go.Scatter(x=tangent_x,y=tangent_y,mode="lines",name="Reta tangente",
                                     line=dict(dash="dot",width=2)))
            fig.add_trace(go.Scatter(x=[xn,nr["xₙ₊₁"]],y=[yn,0],mode="markers+text",
                text=["xₙ","xₙ₊₁"],textposition="top center",name="Newton-Raphson",
                marker=dict(size=11,symbol="x")))
        fig.update_layout(height=520,xaxis_title="x",yaxis_title="f(x)",hovermode="x unified",
                          legend=dict(orientation="h",y=1.08),margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"displaylogo":False})

        st.subheader("5. Progressão do erro")
        st.caption("A escala vertical é logarítmica para facilitar a observação da redução do erro ao longo das iterações.")
        errfig=go.Figure()
        if bis and rowsb:
            errfig.add_trace(go.Scatter(
                x=[r["Iteração"] for r in rowsb],
                y=[max(r["Erro estimado (metade do intervalo)"],1e-16) for r in rowsb],
                mode="lines+markers",name="Bisseção — metade do intervalo",
                hovertemplate="Iteração %{x}<br>Erro estimado: %{y:.3e}<extra></extra>"))
        if new and rowsn:
            errfig.add_trace(go.Scatter(
                x=[r["Iteração"] for r in rowsn],
                y=[max(r["Erro |xₙ₊₁ − xₙ|"],1e-16) for r in rowsn],
                mode="lines+markers",name="Newton-Raphson — diferença entre aproximações",
                hovertemplate="Iteração %{x}<br>Erro estimado: %{y:.3e}<extra></extra>"))
        errfig.update_layout(height=430,xaxis_title="Iteração",yaxis_title="Erro estimado",
                             yaxis_type="log",hovermode="x unified",
                             legend=dict(orientation="h",y=1.12),margin=dict(l=20,r=20,t=45,b=20))
        st.plotly_chart(errfig,use_container_width=True,config={"displaylogo":False})

        st.subheader("6. Comparação dos resultados")
        cm1,cm2=st.columns(2)
        if bis:
            cm1.metric("Raiz — Bisseção",f"{rb:.10g}",f"{ib} iterações")
            cm1.caption(f"Resíduo |f(x)| = {eb:.3e}")
        else:
            cm1.warning("Bisseção não executada: intervalo sem mudança de sinal.")
        if new and rn is not None:
            cm2.metric("Raiz — Newton-Raphson",f"{rn:.10g}",f"{in_} iterações")
            cm2.caption(f"Resíduo |f(x)| = {en:.3e}")

        st.subheader("7. Exportar tabelas")
        ec1,ec2=st.columns(2)
        if bis and rowsb:
            ec1.download_button("⬇ Baixar tabela da Bisseção (CSV)",
                dfb.to_csv(index=False).encode("utf-8-sig"),"feamat_bissecao.csv","text/csv")
        if new and rowsn:
            ec2.download_button("⬇ Baixar tabela de Newton (CSV)",
                dfn.to_csv(index=False).encode("utf-8-sig"),"feamat_newton.csv","text/csv")
    except Exception as e:
        st.error(f"Não foi possível executar o experimento: {e}")
else:
    st.markdown("---")
    st.subheader("Comece por um exemplo")
    st.write("Selecione uma função predefinida na barra lateral. O FEAMAT preencherá automaticamente a expressão, os extremos a e b e o valor inicial x₀.")
    st.markdown("""
    **O que será investigado?**
    - Como os sinais de f(a) e f(b) indicam se o intervalo pode ser usado na Bisseção.
    - Como o ponto médio determina o novo intervalo.
    - Como Newton-Raphson utiliza a reta tangente para produzir novas aproximações.
    - Como o erro evolui ao longo das iterações dos dois métodos.
    """)
    st.info("Dica: comece com f(x) = x² − 2, no intervalo [1, 2], e valor inicial x₀ = 1,5.")
