import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Global Football Analyzer Pro", page_icon="⚽", layout="wide")

# ============================================================
# GLOBAL FOOTBALL ANALYZER PRO
# ------------------------------------------------------------
# Educational/statistical analysis only.
# No bet placement, no guaranteed profit, no "99% winning" claim.
# ============================================================

REQUIRED = ["HomeTeam","AwayTeam","FTHG","FTAG"]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def poisson(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def normalize(v):
    v=np.asarray(v,dtype=float)
    return v/v.sum()

def market_probs(odds):
    inv=np.array([1/o for o in odds],float)
    return normalize(inv)

def dc_adjust(i,j,lh,la,rho):
    # Dixon-Coles low-score correction.
    if i==0 and j==0: return 1-rho*lh*la
    if i==0 and j==1: return 1+rho*lh
    if i==1 and j==0: return 1+rho*la
    if i==1 and j==1: return 1-rho
    return 1.0

def score_matrix(lh,la,max_goals=10,rho=-0.05):
    m=np.zeros((max_goals+1,max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            m[i,j]=poisson(i,lh)*poisson(j,la)*dc_adjust(i,j,lh,la,rho)
    return m/m.sum()

def markets_from_matrix(m):
    n=m.shape[0]
    home=np.tril(m,-1).sum()
    draw=np.trace(m)
    away=np.triu(m,1).sum()
    out={"1":home,"X":draw,"2":away}
    for line in [0.5,1.5,2.5,3.5,4.5,5.5]:
        under=sum(m[i,j] for i in range(n) for j in range(n) if i+j<line)
        out[f"Under {line}"]=under
        out[f"Over {line}"]=1-under
    btts=sum(m[i,j] for i in range(1,n) for j in range(1,n))
    out["BTTS Yes"]=btts
    out["BTTS No"]=1-btts
    out["1X"]=home+draw
    out["X2"]=draw+away
    out["12"]=home+away
    return out

def poisson_expected(df, home, away, half_life=180, rho=-0.05):
    d=df.copy()
    d["Date"]=pd.to_datetime(d["Date"],errors="coerce") if "Date" in d else pd.NaT
    d["FTHG"]=pd.to_numeric(d["FTHG"],errors="coerce")
    d["FTAG"]=pd.to_numeric(d["FTAG"],errors="coerce")
    d=d.dropna(subset=["FTHG","FTAG"])
    now=d["Date"].max() if d["Date"].notna().any() else None
    if now is not None:
        age=(now-d["Date"]).dt.days.clip(lower=0)
        w=np.power(0.5,age/half_life).fillna(1)
    else:
        w=pd.Series(np.ones(len(d)),index=d.index)

    league_h=np.average(d.FTHG,weights=w)
    league_a=np.average(d.FTAG,weights=w)
    hs=d[d.HomeTeam.eq(home)].tail(20)
    aw=d[d.AwayTeam.eq(away)].tail(20)
    if len(hs)<3 or len(aw)<3:
        return league_h,league_a,{"warning":"Historique insuffisant pour au moins une équipe."}

    wh=w.loc[hs.index]; wa=w.loc[aw.index]
    ha=np.average(hs.FTHG,weights=wh)/max(league_h,.1)
    hd=np.average(hs.FTAG,weights=wh)/max(league_a,.1)
    aa=np.average(aw.FTAG,weights=wa)/max(league_a,.1)
    ad=np.average(aw.FTHG,weights=wa)/max(league_h,.1)

    lh=max(.05,league_h*ha*ad)
    la=max(.05,league_a*aa*hd)

    # Optional xG blend, if columns exist.
    if {"xG","xGA"}.issubset(d.columns):
        for c in ["xG","xGA"]:
            d[c]=pd.to_numeric(d[c],errors="coerce")
        hx=hs.xG.dropna().tail(10).mean()
        hxa=hs.xGA.dropna().tail(10).mean()
        ax=aw.xG.dropna().tail(10).mean()
        axa=aw.xGA.dropna().tail(10).mean()
        vals=[hx,hxa,ax,axa]
        if all(np.isfinite(v) for v in vals):
            lh=.55*lh+.45*(hx+axa)/2
            la=.55*la+.45*(ax+hxa)/2
    return lh,la,{"warning":None}

def elo_ratings(df, k=20, home_adv=60):
    ratings={}
    d=df.copy()
    if "Date" in d:
        d["Date"]=pd.to_datetime(d["Date"],errors="coerce")
        d=d.sort_values("Date")
    for _,r in d.iterrows():
        h,a=str(r.HomeTeam),str(r.AwayTeam)
        rh=ratings.get(h,1500); ra=ratings.get(a,1500)
        expected=1/(1+10**(-(rh+home_adv-ra)/400))
        if r.FTR=="H": result=1
        elif r.FTR=="D": result=.5
        else: result=0
        change=k*(result-expected)
        ratings[h]=rh+change
        ratings[a]=ra-change
    return ratings

def elo_prob(rh,ra,home_adv=60):
    p=1/(1+10**(-(rh+home_adv-ra)/400))
    # Convert binary Elo edge into a rough home/draw/away prior.
    draw=.24
    home=(1-draw)*p
    away=(1-draw)*(1-p)
    return normalize([home,draw,away])

def brier_multiclass(probs, outcomes):
    probs=np.asarray(probs,float); y=np.asarray(outcomes,int)
    return np.mean(np.sum((probs-np.eye(probs.shape[1])[y])**2,axis=1))

def logloss(probs,outcomes,eps=1e-12):
    probs=np.clip(np.asarray(probs,float),eps,1)
    return float(-np.mean(np.log(probs[np.arange(len(outcomes)),outcomes])))

def calibration_table(pred,actual,bins=10):
    p=np.asarray(pred); y=np.asarray(actual)
    edges=np.linspace(0,1,bins+1)
    rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if mask.sum():
            rows.append({"Bin":f"{lo:.1f}-{hi:.1f}","N":int(mask.sum()),
                         "Predicted":p[mask].mean(),"Observed":y[mask].mean()})
    return pd.DataFrame(rows)

st.title("⚽ Global Football Analyzer PRO")
st.caption("Moteur de probabilités, comparaison au marché, analyse des marchés dérivés et backtesting. Aucune garantie de gain et aucune mise automatique.")

with st.sidebar:
    st.header("⚙️ Paramètres")
    half=st.slider("Demi-vie des matchs (jours)",30,365,180)
    rho=st.slider("Correction Dixon-Coles", -0.20,0.20,-0.05,0.01)
    maxg=st.slider("Maximum de buts simulés",6,14,10)
    blend_market=st.slider("Poids du marché",0.0,1.0,0.50,0.05)
    st.markdown("**Principe:** le marché peut contenir une information agrégée importante. Une étude récente trouve que la calibration aux prix du marché peut fortement améliorer la prévision in-play. citeturn0academia16")

file=st.file_uploader("📥 Charger un CSV historique",type=["csv"])
if file:
    df=pd.read_csv(file)
else:
    df=pd.DataFrame(columns=REQUIRED+["Date","FTR","xG","xGA","B365H","B365D","B365A"])

st.markdown("### Format minimal")
st.code("HomeTeam,AwayTeam,FTHG,FTAG,FTR,Date,xG,xGA,B365H,B365D,B365A")

if len(df):
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: st.error(f"Colonnes manquantes : {missing}")
    else:
        st.success(f"{len(df):,} matchs chargés.")
        st.dataframe(df.head(10),use_container_width=True)

st.header("1️⃣ Analyse d'un match")
c1,c2,c3=st.columns(3)
with c1: home=st.text_input("Domicile")
with c2: away=st.text_input("Extérieur")
with c3:
    odds_h=st.number_input("Cote 1",1.01,1000.0,2.0,.01)
    odds_d=st.number_input("Cote X",1.01,1000.0,3.3,.01)
    odds_a=st.number_input("Cote 2",1.01,1000.0,3.2,.01)

if st.button("🔬 Lancer l'analyse",type="primary") and len(df) and home and away:
    if not {"HomeTeam","AwayTeam","FTHG","FTAG"}.issubset(df.columns):
        st.error("CSV incomplet.")
    else:
        lh,la,meta=poisson_expected(df,home,away,half,rho)
        mat=score_matrix(lh,la,maxg,rho)
        mk=markets_from_matrix(mat)
        mp=market_probs([odds_h,odds_d,odds_a])
        p_model=np.array([mk["1"],mk["X"],mk["2"]])
        p_final=(1-blend_market)*p_model+blend_market*mp

        m1,m2,m3,m4=st.columns(4)
        m1.metric("λ domicile",f"{lh:.2f}")
        m2.metric("λ extérieur",f"{la:.2f}")
        m3.metric("P(1)",f"{p_final[0]*100:.1f}%")
        m4.metric("P(X)",f"{p_final[1]*100:.1f}%")

        table=pd.DataFrame({
            "Issue":["1","X","2"],
            "Modèle":[*p_model],
            "Marché":[*mp],
            "Probabilité finale":[*p_final],
            "Cote":[odds_h,odds_d,odds_a],
            "EV théorique":[p_final[0]*odds_h-1,p_final[1]*odds_d-1,p_final[2]*odds_a-1]
        })
        st.subheader("Probabilités")
        st.dataframe(table.style.format({"Modèle":"{:.2%}","Marché":"{:.2%}","Probabilité finale":"{:.2%}","EV théorique":"{:.2%}"}),use_container_width=True)

        st.subheader("Marchés dérivés")
        derived=pd.DataFrame(sorted(mk.items(),key=lambda x:x[1],reverse=True),columns=["Marché","Probabilité"])
        st.dataframe(derived.style.format({"Probabilité":"{:.2%}"}),use_container_width=True)

        st.subheader("🎯 Scores exacts les plus probables")
        scores=[]
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                scores.append((f"{i}-{j}",mat[i,j]))
        st.dataframe(pd.DataFrame(sorted(scores,key=lambda x:x[1],reverse=True)[:12],columns=["Score","Probabilité"]).style.format({"Probabilité":"{:.2%}"}),use_container_width=True)

        st.subheader("🧮 Analyse de risque d'un combiné")
        st.info("Les probabilités de plusieurs événements se multiplient seulement sous une hypothèse d'indépendance approximative. Les corrélations peuvent rendre ce calcul trop optimiste.")
        selected=st.multiselect("Sélections à combiner",list(mk.keys()),default=["1"] if "1" in mk else [])
        if selected:
            p=np.prod([mk[x] for x in selected])
            st.metric("Probabilité conjointe approximative",f"{p*100:.4f}%")

st.header("2️⃣ Classement des opportunités")
st.write("Pour chaque marché disponible dans ton historique de cotes, le moteur peut comparer probabilité estimée et probabilité implicite. Cela ne signifie pas qu'un écart est forcément un 'value bet' : il peut provenir d'un mauvais modèle ou d'une donnée périmée.")

if len(df) and {"B365H","B365D","B365A","FTR"}.issubset(df.columns):
    st.info("Le module de classement complet nécessite un backtest temporel pour éviter de mesurer le modèle sur les mêmes matchs qui ont servi à l'entraîner.")

st.header("3️⃣ Backtesting hors échantillon")
st.write("Le backtest doit respecter l'ordre du temps : passé → entraînement → futur → évaluation. Ne mélange jamais les matchs futurs dans les variables du passé.")
if len(df) and {"FTR","HomeTeam","AwayTeam","FTHG","FTAG"}.issubset(df.columns):
    d=df.copy()
    if "Date" in d:
        d["Date"]=pd.to_datetime(d["Date"],errors="coerce")
        d=d.sort_values("Date")
    split=st.slider("Pourcentage entraînement",50,90,75)
    cut=max(1,int(len(d)*split/100))
    train=d.iloc[:cut].copy(); test=d.iloc[cut:].copy()
    if len(test):
        preds=[]; actual=[]
        for _,r in test.iterrows():
            hist=train[train.index<r.name]
            if len(hist)<30: continue
            try:
                lh,la,_=poisson_expected(hist,str(r.HomeTeam),str(r.AwayTeam),half,rho)
                mat=score_matrix(lh,la,maxg,rho)
                p=np.array([np.tril(mat,-1).sum(),np.trace(mat),np.triu(mat,1).sum()])
                preds.append(p)
                actual.append(0 if r.FTR=="H" else 1 if r.FTR=="D" else 2)
            except Exception:
                pass
        if preds:
            preds=np.array(preds); actual=np.array(actual)
            st.metric("Matchs évalués",len(actual))
            st.metric("Brier score",f"{brier_multiclass(preds,actual):.4f}")
            st.metric("Log loss",f"{logloss(preds,actual):.4f}")
            cal=calibration_table(preds.max(axis=1),np.argmax(preds,axis=1)==actual)
            if len(cal): st.dataframe(cal,use_container_width=True)

st.header("4️⃣ Elo + ensemble")
st.write("L'Elo donne une mesure indépendante de la forme brute. L'ensemble Poisson + Elo + marché peut être calibré sur validation historique, plutôt que de choisir arbitrairement un modèle.")
if len(df) and {"FTR","HomeTeam","AwayTeam"}.issubset(df.columns):
    ratings=elo_ratings(df)
    if home and away and home in ratings and away in ratings:
        ep=elo_prob(ratings[home],ratings[away])
        st.write(f"Elo {home}: **{ratings[home]:.0f}** — Elo {away}: **{ratings[away]:.0f}**")
        st.dataframe(pd.DataFrame({"Issue":["1","X","2"],"Elo":ep}),use_container_width=True)

st.header("5️⃣ Contrôles anti-erreurs")
checks=[
("Pas de fuite temporelle","Utiliser uniquement les données connues avant le coup d'envoi."),
("Calibration","Une probabilité annoncée à 70% doit se produire proche de 70% sur un grand échantillon."),
("Marché","Comparer au prix disponible au même instant ; ne pas mélanger opening et closing odds."),
("Échantillon","Évaluer sur une période future non utilisée pour entraîner."),
("Corrélation","Éviter de multiplier aveuglément des sélections liées au même match."),
("Données","Vérifier blessures, compositions, changements de coach et qualité de la source."),
("Incertitude","Présenter des intervalles ou scénarios plutôt qu'un chiffre unique trop précis.")
]
st.dataframe(pd.DataFrame(checks,columns=["Contrôle","Pourquoi"]),use_container_width=True)

st.header("6️⃣ Architecture production")
st.code("""
DATA LAYER
  fixtures + results + odds + xG + lineups + injuries + weather
       ↓
FEATURE STORE
  Elo / forme / home advantage / xG rolling / rest days / strength
       ↓
MODELS
  Dixon-Coles Poisson
  Elo
  Gradient Boosting / Logistic Regression
  calibration isotonic / Platt
       ↓
ENSEMBLE
  weights learned only on validation data
       ↓
MARKET ENGINE
  remove overround → compare model vs market
       ↓
RISK ENGINE
  correlation + uncertainty + drawdown + max exposure
       ↓
REPORT
  probabilities + confidence interval + reasons + data timestamp
""")

st.warning("⚠️ Aucun modèle sérieux ne peut garantir 99% de réussite sur une cote cumulée de 1000. Une cote de 1000 implique une probabilité de marché brute de seulement 0,1% avant marge. Le but ici est de mesurer l'incertitude et la calibration, pas de promettre un gain.")
