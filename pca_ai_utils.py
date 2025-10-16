"""
pca_ai_utils.py
===============
Modulo per analisi e interpretazione PCA basata su teoria consolidata
Focus su interpretazione geometrica di loadings e scores

Author: ChemometricSolutions Team
Version: 3.0.0 - Pure Interpretation Focus
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# CORE PCA INTERPRETATION BASED ON GEOMETRIC PRINCIPLES
# =============================================================================

def interpret_pca_geometry(loadings: pd.DataFrame, scores: pd.DataFrame, 
                          pc_x: int = 1, pc_y: int = 2, 
                          loading_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Interpretazione PCA basata su principi geometrici consolidati.
    
    TEORIA FONDAMENTALE:
    - Loadings: coefficienti delle combinazioni lineari (direzioni nello spazio variabili)
    - Scores: coordinate campioni nel nuovo spazio PC
    - Distanza dall'origine nei loadings = importanza variabile
    - Vicinanza negli scores = similarità tra campioni
    
    Parameters:
    -----------
    loadings : pd.DataFrame
        Matrice loadings (variabili × componenti)
    scores : pd.DataFrame  
        Matrice scores (campioni × componenti)
    pc_x, pc_y : int
        Componenti principali da analizzare
    loading_threshold : float
        Soglia per considerare un loading significativo
        
    Returns:
    --------
    Dict con interpretazione completa
    """
    
    pc_x_col = f'PC{pc_x}'
    pc_y_col = f'PC{pc_y}'
    
    # ===========================================================
    # PARTE 1: ANALISI LOADINGS (Relazioni tra Variabili)
    # ===========================================================
    
    loadings_interpretation = analyze_loadings_geometry(
        loadings, pc_x_col, pc_y_col, loading_threshold
    )
    
    # ===========================================================
    # PARTE 2: ANALISI SCORES (Distribuzione Campioni)
    # ===========================================================
    
    scores_interpretation = analyze_scores_geometry(
        scores, pc_x_col, pc_y_col
    )
    
    # ===========================================================
    # PARTE 3: INTERPRETAZIONE INTEGRATA
    # ===========================================================
    
    integrated = integrate_loadings_scores_interpretation(
        loadings_interpretation, scores_interpretation, pc_x, pc_y
    )
    
    return {
        'loadings_interpretation': loadings_interpretation,
        'scores_interpretation': scores_interpretation,
        'integrated_interpretation': integrated,
        'pc_x': pc_x,
        'pc_y': pc_y,
        'success': True
    }

def analyze_loadings_geometry(loadings: pd.DataFrame, pc_x: str, pc_y: str, 
                             threshold: float = 0.3) -> Dict[str, Any]:
    """
    Analizza geometria dei loadings secondo teoria PCA.
    
    INTERPRETAZIONE GEOMETRICA:
    - Distanza origine = |loading| = importanza
    - Angolo tra variabili = correlazione
    - Stesso quadrante = correlate positive
    - Quadranti opposti = correlate negative  
    - Ortogonali = non correlate
    """
    
    load_x = loadings[pc_x]
    load_y = loadings[pc_y]
    
    # Calcola distanze dall'origine (importanza assoluta)
    distances = np.sqrt(load_x**2 + load_y**2)
    
    # Ordina variabili per importanza
    importance_ranking = distances.sort_values(ascending=False)
    
    # Identifica variabili significative
    significant_vars = distances[distances >= threshold]
    
    # Analizza correlazioni tra variabili (basato su angoli)
    correlations = analyze_variable_correlations(load_x, load_y, distances, threshold)
    
    # Analizza distribuzione nei quadranti
    quadrants = classify_loading_quadrants(load_x, load_y, distances, threshold)
    
    # Identifica pattern dominanti
    patterns = identify_loading_patterns(load_x, load_y, distances, threshold)
    
    return {
        'importance_ranking': importance_ranking.to_dict(),
        'n_significant': len(significant_vars),
        'correlations': correlations,
        'quadrants': quadrants,
        'patterns': patterns,
        'interpretation_text': generate_loadings_interpretation_text(
            importance_ranking, correlations, quadrants, patterns, pc_x, pc_y
        )
    }

def analyze_variable_correlations(load_x: pd.Series, load_y: pd.Series, 
                                 distances: pd.Series, threshold: float) -> Dict[str, List]:
    """
    Identifica correlazioni tra variabili basate su geometria loadings.
    
    Usa prodotto scalare e angoli per determinare correlazioni.
    """
    
    # Solo variabili significative
    significant = distances >= threshold
    sig_vars = distances[significant].index
    
    correlations = {
        'strongly_correlated': [],      # angolo < 30°
        'moderately_correlated': [],    # 30° < angolo < 60°
        'uncorrelated': [],             # 60° < angolo < 120°
        'anticorrelated': []            # angolo > 150°
    }
    
    # Analizza coppie di variabili
    for i, var1 in enumerate(sig_vars):
        for var2 in sig_vars[i+1:]:
            # Calcola angolo tra vettori loading
            dot_product = load_x[var1]*load_x[var2] + load_y[var1]*load_y[var2]
            norm_product = distances[var1] * distances[var2]
            
            if norm_product > 0:
                cos_angle = dot_product / norm_product
                # Clamp per evitare errori numerici
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = np.arccos(cos_angle) * 180 / np.pi
                
                if angle_deg < 30:
                    correlations['strongly_correlated'].append({
                        'var1': var1, 'var2': var2, 'angle': angle_deg
                    })
                elif angle_deg < 60:
                    correlations['moderately_correlated'].append({
                        'var1': var1, 'var2': var2, 'angle': angle_deg
                    })
                elif angle_deg < 120:
                    correlations['uncorrelated'].append({
                        'var1': var1, 'var2': var2, 'angle': angle_deg
                    })
                elif angle_deg > 150:
                    correlations['anticorrelated'].append({
                        'var1': var1, 'var2': var2, 'angle': angle_deg
                    })
    
    return correlations

def classify_loading_quadrants(load_x: pd.Series, load_y: pd.Series, 
                              distances: pd.Series, threshold: float) -> Dict:
    """
    Classifica variabili per quadrante nel loading plot.
    """
    
    significant = distances >= threshold
    
    quadrants = {
        'Q1': [],  # ++
        'Q2': [],  # -+
        'Q3': [],  # --
        'Q4': []   # +-
    }
    
    for var in distances[significant].index:
        x_val = load_x[var]
        y_val = load_y[var]
        importance = distances[var]
        
        if x_val >= 0 and y_val >= 0:
            quadrants['Q1'].append({'var': var, 'x': x_val, 'y': y_val, 'importance': importance})
        elif x_val < 0 and y_val >= 0:
            quadrants['Q2'].append({'var': var, 'x': x_val, 'y': y_val, 'importance': importance})
        elif x_val < 0 and y_val < 0:
            quadrants['Q3'].append({'var': var, 'x': x_val, 'y': y_val, 'importance': importance})
        else:
            quadrants['Q4'].append({'var': var, 'x': x_val, 'y': y_val, 'importance': importance})
    
    # Ordina per importanza dentro ogni quadrante
    for q in quadrants:
        quadrants[q].sort(key=lambda x: x['importance'], reverse=True)
    
    return quadrants

def identify_loading_patterns(load_x: pd.Series, load_y: pd.Series, 
                             distances: pd.Series, threshold: float) -> Dict:
    """
    Identifica pattern specifici nei loadings.
    """
    
    significant = distances >= threshold
    sig_x = load_x[significant]
    sig_y = load_y[significant]
    sig_dist = distances[significant]
    
    patterns = {
        'pc_x_dominant': [],  # Variabili che contribuiscono principalmente a PC_X
        'pc_y_dominant': [],  # Variabili che contribuiscono principalmente a PC_Y
        'balanced': [],       # Variabili con contributi bilanciati
        'clusters': []        # Gruppi di variabili vicine
    }
    
    # Identifica dominanza
    for var in sig_dist.index:
        x_contrib = abs(load_x[var])
        y_contrib = abs(load_y[var])
        ratio = x_contrib / (y_contrib + 1e-10)
        
        if ratio > 2:
            patterns['pc_x_dominant'].append(var)
        elif ratio < 0.5:
            patterns['pc_y_dominant'].append(var)
        else:
            patterns['balanced'].append(var)
    
    # Identifica clusters di variabili (semplificato)
    # Variabili sono in cluster se distanza < 0.1 nel loading plot
    cluster_threshold = 0.1
    processed = set()
    
    for var1 in sig_dist.index:
        if var1 in processed:
            continue
            
        cluster = [var1]
        processed.add(var1)
        
        for var2 in sig_dist.index:
            if var2 in processed:
                continue
                
            dist = np.sqrt((load_x[var1]-load_x[var2])**2 + (load_y[var1]-load_y[var2])**2)
            if dist < cluster_threshold:
                cluster.append(var2)
                processed.add(var2)
        
        if len(cluster) > 1:
            patterns['clusters'].append(cluster)
    
    return patterns

def analyze_scores_geometry(scores: pd.DataFrame, pc_x: str, pc_y: str) -> Dict:
    """
    Analizza geometria degli scores secondo teoria PCA.
    
    INTERPRETAZIONE:
    - Campioni vicini = caratteristiche simili
    - Campioni distanti = caratteristiche diverse
    - Outliers = campioni atipici
    - Clusters = gruppi con caratteristiche comuni
    """
    
    score_x = scores[pc_x]
    score_y = scores[pc_y]
    
    # Calcola distanze dall'origine (atipicità)
    distances = np.sqrt(score_x**2 + score_y**2)
    
    # Identifica outliers (Hotelling T²-like)
    outlier_threshold = distances.mean() + 2*distances.std()
    outliers = distances[distances > outlier_threshold]
    
    # Analizza distribuzione spaziale
    distribution = analyze_score_distribution(score_x, score_y)
    
    # Identifica clusters
    clusters = identify_score_clusters(score_x, score_y, distances)
    
    # Analizza trend/pattern
    patterns = identify_score_patterns(score_x, score_y, scores.index)
    
    return {
        'n_samples': len(scores),
        'outliers': outliers.to_dict(),
        'n_outliers': len(outliers),
        'distribution': distribution,
        'clusters': clusters,
        'patterns': patterns,
        'interpretation_text': generate_scores_interpretation_text(
            len(scores), outliers, distribution, clusters, patterns, pc_x, pc_y
        )
    }

def analyze_score_distribution(score_x: pd.Series, score_y: pd.Series) -> Dict:
    """
    Analizza come i campioni sono distribuiti nello spazio PC.
    """
    
    # Statistiche base
    stats = {
        'x_range': [float(score_x.min()), float(score_x.max())],
        'y_range': [float(score_y.min()), float(score_y.max())],
        'x_spread': float(score_x.std()),
        'y_spread': float(score_y.std()),
        'centroid': [float(score_x.mean()), float(score_y.mean())]
    }
    
    # Analisi quadranti
    quadrant_counts = {
        'Q1': sum((score_x >= 0) & (score_y >= 0)),
        'Q2': sum((score_x < 0) & (score_y >= 0)),
        'Q3': sum((score_x < 0) & (score_y < 0)),
        'Q4': sum((score_x >= 0) & (score_y < 0))
    }
    
    # Simmetria/asimmetria
    asymmetry = {
        'x_skewness': float(score_x.skew()),
        'y_skewness': float(score_y.skew()),
        'x_symmetry': 'symmetric' if abs(score_x.skew()) < 0.5 else ('right-skewed' if score_x.skew() > 0 else 'left-skewed'),
        'y_symmetry': 'symmetric' if abs(score_y.skew()) < 0.5 else ('right-skewed' if score_y.skew() > 0 else 'left-skewed')
    }
    
    return {
        'statistics': stats,
        'quadrant_distribution': quadrant_counts,
        'asymmetry': asymmetry
    }

def identify_score_clusters(score_x: pd.Series, score_y: pd.Series, 
                           distances: pd.Series) -> List[Dict]:
    """
    Identifica gruppi naturali di campioni (clustering semplificato).
    """
    
    # Metodo semplice: divide lo spazio in regioni e conta densità
    n_bins = 3
    x_bins = pd.cut(score_x, bins=n_bins, labels=['low', 'medium', 'high'])
    y_bins = pd.cut(score_y, bins=n_bins, labels=['low', 'medium', 'high'])
    
    clusters = []
    
    for x_label in ['low', 'medium', 'high']:
        for y_label in ['low', 'medium', 'high']:
            mask = (x_bins == x_label) & (y_bins == y_label)
            if mask.sum() >= 3:  # Almeno 3 campioni per considerarlo cluster
                cluster_samples = score_x[mask].index.tolist()
                cluster_center = [float(score_x[mask].mean()), float(score_y[mask].mean())]
                
                clusters.append({
                    'region': f"{x_label}_{y_label}",
                    'n_samples': len(cluster_samples),
                    'samples': cluster_samples[:10],  # primi 10 per brevità
                    'center': cluster_center,
                    'density': len(cluster_samples) / len(score_x) * 100  # percentuale
                })
    
    # Ordina per densità
    clusters.sort(key=lambda x: x['density'], reverse=True)
    
    return clusters

def identify_score_patterns(score_x: pd.Series, score_y: pd.Series, 
                           sample_names: pd.Index) -> Dict:
    """
    Identifica pattern specifici nella distribuzione degli scores.
    """
    
    patterns = {
        'temporal_trend': False,
        'batch_effect': False,
        'gradient': None,
        'special_samples': []
    }
    
    # Check per trend temporale (se indici sono numerici o date)
    try:
        if pd.api.types.is_numeric_dtype(sample_names):
            corr_x = np.corrcoef(sample_names, score_x)[0, 1]
            corr_y = np.corrcoef(sample_names, score_y)[0, 1]
            
            if abs(corr_x) > 0.5 or abs(corr_y) > 0.5:
                patterns['temporal_trend'] = True
                patterns['gradient'] = 'x-axis' if abs(corr_x) > abs(corr_y) else 'y-axis'
    except:
        pass
    
    # Identifica campioni speciali (agli estremi)
    extremes = {
        'max_x': score_x.idxmax(),
        'min_x': score_x.idxmin(),
        'max_y': score_y.idxmax(),
        'min_y': score_y.idxmin()
    }
    
    for label, sample in extremes.items():
        patterns['special_samples'].append({
            'sample': sample,
            'type': label,
            'x_score': float(score_x[sample]),
            'y_score': float(score_y[sample])
        })
    
    return patterns

def integrate_loadings_scores_interpretation(loadings_interp: Dict, scores_interp: Dict,
                                           pc_x: int, pc_y: int) -> Dict[str, str]:
    """
    Integra interpretazioni di loadings e scores per insight completi.
    """
    
    insights = {}
    
    # Insight principale
    insights['main_finding'] = generate_main_finding(loadings_interp, scores_interp, pc_x, pc_y)
    
    # Collegamento loadings-scores
    insights['loading_score_link'] = generate_loading_score_link(loadings_interp, scores_interp)
    
    # Interpretazione outliers
    insights['outlier_explanation'] = generate_outlier_explanation(loadings_interp, scores_interp)
    
    # Raccomandazioni
    insights['recommendations'] = generate_recommendations(loadings_interp, scores_interp)
    
    return insights

# =============================================================================
# TEXT GENERATION FUNCTIONS
# =============================================================================

def generate_loadings_interpretation_text(importance: pd.Series, correlations: Dict,
                                         quadrants: Dict, patterns: Dict,
                                         pc_x: str, pc_y: str) -> str:
    """
    Genera testo interpretativo per i loadings.
    """
    
    text = f"### Interpretazione Loadings {pc_x} vs {pc_y}\n\n"
    
    # Variabili più importanti
    top_vars = list(importance.head(5).index)
    text += f"**Variabili più influenti:**\n"
    for i, var in enumerate(top_vars, 1):
        text += f"{i}. {var} (importanza: {importance[var]:.3f})\n"
    
    text += "\n**Relazioni tra variabili:**\n"
    
    # Correlazioni forti
    if correlations['strongly_correlated']:
        text += "- **Fortemente correlate**: "
        pairs = [f"{c['var1']}-{c['var2']}" for c in correlations['strongly_correlated'][:3]]
        text += ", ".join(pairs) + "\n"
    
    # Anticorrelazioni
    if correlations['anticorrelated']:
        text += "- **Anticorrelate**: "
        pairs = [f"{c['var1']} vs {c['var2']}" for c in correlations['anticorrelated'][:3]]
        text += ", ".join(pairs) + "\n"
    
    # Pattern dominanti
    text += f"\n**Pattern identificati:**\n"
    if patterns['pc_x_dominant']:
        text += f"- Variabili che definiscono principalmente {pc_x}: {', '.join(patterns['pc_x_dominant'][:3])}\n"
    if patterns['pc_y_dominant']:
        text += f"- Variabili che definiscono principalmente {pc_y}: {', '.join(patterns['pc_y_dominant'][:3])}\n"
    
    # Clusters di variabili
    if patterns['clusters']:
        text += f"- Identificati {len(patterns['clusters'])} gruppi di variabili correlate\n"
    
    return text

def generate_scores_interpretation_text(n_samples: int, outliers: pd.Series,
                                       distribution: Dict, clusters: List,
                                       patterns: Dict, pc_x: str, pc_y: str) -> str:
    """
    Genera testo interpretativo per gli scores.
    """
    
    text = f"### Interpretazione Scores {pc_x} vs {pc_y}\n\n"
    
    text += f"**Distribuzione campioni ({n_samples} totali):**\n"
    
    # Outliers
    if len(outliers) > 0:
        text += f"- **Outliers identificati**: {len(outliers)} campioni atipici\n"
        top_outliers = list(outliers.head(3).index)
        text += f"  Più estremi: {', '.join(map(str, top_outliers))}\n"
    
    # Clusters
    if clusters:
        text += f"- **Raggruppamenti naturali**: {len(clusters)} regioni ad alta densità\n"
        main_cluster = clusters[0]
        text += f"  Cluster principale: {main_cluster['n_samples']} campioni ({main_cluster['density']:.1f}%)\n"
    
    # Distribuzione quadranti
    quad_dist = distribution['quadrant_distribution']
    dominant_quad = max(quad_dist, key=quad_dist.get)
    text += f"- **Distribuzione spaziale**: maggior concentrazione in {dominant_quad} ({quad_dist[dominant_quad]} campioni)\n"
    
    # Pattern speciali
    if patterns['temporal_trend']:
        text += f"- **Trend temporale** rilevato lungo {patterns['gradient']}\n"
    
    # Asimmetria
    asym = distribution['asymmetry']
    if asym['x_symmetry'] != 'symmetric' or asym['y_symmetry'] != 'symmetric':
        text += f"- **Asimmetria**: {pc_x} {asym['x_symmetry']}, {pc_y} {asym['y_symmetry']}\n"
    
    return text

def generate_main_finding(loadings_interp: Dict, scores_interp: Dict, 
                        pc_x: int, pc_y: int) -> str:
    """
    Genera il finding principale dall'analisi integrata.
    """
    
    n_sig_vars = loadings_interp['n_significant']
    n_outliers = scores_interp['n_outliers']
    n_samples = scores_interp['n_samples']
    
    # Prendi le top 3 variabili
    top_vars = list(loadings_interp['importance_ranking'].keys())[:3]
    
    text = f"""
PC{pc_x} e PC{pc_y} rivelano la struttura principale del dataset attraverso {n_sig_vars} variabili significative.
Le variabili chiave ({', '.join(top_vars)}) definiscono le direzioni di massima varianza.
La distribuzione dei {n_samples} campioni mostra {n_outliers} potenziali outliers e 
{len(scores_interp['clusters'])} raggruppamenti naturali, suggerendo eterogeneità nel dataset.
"""
    
    return text

def generate_loading_score_link(loadings_interp: Dict, scores_interp: Dict) -> str:
    """
    Collega interpretazione loadings-scores.
    """
    
    text = """
Il collegamento loadings-scores permette di capire PERCHÉ i campioni si distribuiscono come osservato:
- Campioni con scores positivi hanno valori elevati nelle variabili con loadings positivi
- Raggruppamenti di campioni corrispondono a profili simili nelle variabili correlate
- Outliers sono caratterizzati da valori estremi nelle variabili più distanti dall'origine
"""
    
    # Aggiungi specifiche se ci sono pattern
    if loadings_interp['patterns']['clusters']:
        text += f"\nI {len(loadings_interp['patterns']['clusters'])} gruppi di variabili correlate "
        text += "suggeriscono ridondanza informativa che spiega i cluster negli scores."
    
    return text

def generate_outlier_explanation(loadings_interp: Dict, scores_interp: Dict) -> str:
    """
    Spiega gli outliers basandosi sui loadings.
    """
    
    if scores_interp['n_outliers'] == 0:
        return "Nessun outlier significativo identificato."
    
    text = f"""
I {scores_interp['n_outliers']} outliers identificati sono probabilmente dovuti a:
- Valori estremi nelle variabili ad alto loading
- Combinazioni inusuali di variabili normalmente correlate
- Possibili errori di misura o condizioni sperimentali anomale

Per investigare, esaminare le variabili top ({', '.join(list(loadings_interp['importance_ranking'].keys())[:3])}) 
nei campioni outlier.
"""
    
    return text

def generate_recommendations(loadings_interp: Dict, scores_interp: Dict) -> str:
    """
    Genera raccomandazioni basate sull'analisi.
    """

    recommendations = []

    # Basate su numero variabili significative
    if loadings_interp['n_significant'] > 10:
        recommendations.append("- Considerare feature selection per ridurre ridondanza")

    # Basate su outliers
    if scores_interp['n_outliers'] > scores_interp['n_samples'] * 0.1:
        recommendations.append("- Investigare gli outliers: possibili problemi di qualità dati")

    # Basate su correlazioni
    if len(loadings_interp['correlations']['strongly_correlated']) > 5:
        recommendations.append("- Molte variabili correlate: valutare se alcune sono ridondanti")

    # Basate su clusters
    if len(scores_interp['clusters']) > 1:
        recommendations.append("- Multipli raggruppamenti: considerare analisi separata per gruppo")

    # Sempre utili
    recommendations.extend([
        "- Creare biplot per visualizzazione integrata loadings-scores",
        "- Validare interpretazione con conoscenza del dominio",
        "- Considerare componenti aggiuntive se varianza spiegata < 70%"
    ])

    return "\n".join(recommendations)

def analyze_metavariable_grouping(scores: pd.DataFrame, metavar_data: pd.Series,
                                  metavar_name: str, pc_x: int, pc_y: int) -> str:
    """
    Analyzes sample groupings based on a metavariable and their spatial distribution in PCA.

    Parameters:
    -----------
    scores : pd.DataFrame
        Matrice degli scores
    metavar_data : pd.Series
        Metavariable data aligned with scores
    metavar_name : str
        Name of the metavariable
    pc_x, pc_y : int
        Components being analyzed

    Returns:
    --------
    str : Interpretation text for metavariable groupings
    """

    pc_x_col = f'PC{pc_x}'
    pc_y_col = f'PC{pc_y}'

    score_x = scores[pc_x_col]
    score_y = scores[pc_y_col]

    # Group samples by metavariable value
    unique_groups = metavar_data.dropna().unique()

    text = f"""
### 🏷️ Sample Grouping Analysis: {metavar_name}

**Groups identified:** {len(unique_groups)}

"""

    # Analyze each group
    group_stats = []
    for group_val in unique_groups:
        mask = (metavar_data == group_val)
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        group_x = score_x[mask]
        group_y = score_y[mask]

        # Calculate centroid
        centroid_x = float(group_x.mean())
        centroid_y = float(group_y.mean())

        # Calculate spread (how tight the group is)
        spread_x = float(group_x.std())
        spread_y = float(group_y.std())
        avg_spread = (spread_x + spread_y) / 2

        # Calculate distance from origin
        dist_from_origin = np.sqrt(centroid_x**2 + centroid_y**2)

        group_stats.append({
            'group': str(group_val),
            'n_samples': n_samples,
            'centroid': (centroid_x, centroid_y),
            'spread': avg_spread,
            'distance': dist_from_origin
        })

    # Sort by number of samples
    group_stats.sort(key=lambda x: x['n_samples'], reverse=True)

    # Report each group
    for stats in group_stats:
        text += f"**{stats['group']}** ({stats['n_samples']} samples)\n"
        text += f"  - Centroid: PC{pc_x}={stats['centroid'][0]:.3f}, PC{pc_y}={stats['centroid'][1]:.3f}\n"
        text += f"  - Spatial spread: {stats['spread']:.3f} (lower = more compact)\n"
        text += f"  - Distance from origin: {stats['distance']:.3f}\n\n"

    # Analyze separation between groups
    if len(group_stats) > 1:
        text += "**Spatial Separation Analysis:**\n"

        # Calculate distances between centroids
        for i, group1 in enumerate(group_stats):
            for group2 in group_stats[i+1:]:
                dx = group1['centroid'][0] - group2['centroid'][0]
                dy = group1['centroid'][1] - group2['centroid'][1]
                dist = np.sqrt(dx**2 + dy**2)

                # Assess separation quality
                avg_group_spread = (group1['spread'] + group2['spread']) / 2
                separation_ratio = dist / (avg_group_spread + 0.001)  # avoid division by zero

                if separation_ratio > 2:
                    quality = "Well separated"
                elif separation_ratio > 1:
                    quality = "Moderately separated"
                else:
                    quality = "Overlapping"

                text += f"- {group1['group']} vs {group2['group']}: distance={dist:.3f} ({quality})\n"

        text += "\n**Interpretation:**\n"
        if any(abs(group1['centroid'][0] - group2['centroid'][0]) > (group1['spread'] + group2['spread'])
               for i, group1 in enumerate(group_stats) for group2 in group_stats[i+1:]):
            text += "- Samples with the same metavariable code cluster together in the PCA space\n"
            text += "- This suggests the metavariable explains some of the variance captured by the PCs\n"
            text += "- Groups with similar coordinates have similar multivariate characteristics\n"
        else:
            text += "- Sample groups show overlap in the PCA space\n"
            text += "- The metavariable may not be a primary driver of variance in these components\n"
            text += "- Consider examining additional PCs or other factors\n"

    return text

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_pca_complete(loadings: pd.DataFrame, scores: pd.DataFrame,
                        pc_x: int = 1, pc_y: int = 2,
                        threshold: float = 0.3, data_type: str = "Generic",
                        metavar_data: pd.Series = None,
                        metavar_name: str = None) -> str:
    """
    Funzione principale per analisi completa PCA.

    Parameters:
    -----------
    loadings : pd.DataFrame
        Matrice dei loadings
    scores : pd.DataFrame
        Matrice degli scores
    pc_x, pc_y : int
        Componenti da analizzare
    threshold : float
        Soglia per loadings significativi
    data_type : str
        Tipo di dati per contestualizzazione
    metavar_data : pd.Series, optional
        Metavariable for sample grouping analysis
    metavar_name : str, optional
        Name of the metavariable

    Returns:
    --------
    str : Interpretazione testuale completa
    """
    
    try:
        # Esegui analisi geometrica
        result = interpret_pca_geometry(loadings, scores, pc_x, pc_y, threshold)
        
        if not result['success']:
            return "Errore nell'analisi PCA"
        
        # Genera report testuale
        report = f"""
# INTERPRETAZIONE PCA - PC{pc_x} vs PC{pc_y}
## Tipo di dati: {data_type}

---

## 📊 ANALISI LOADINGS (Relazioni tra Variabili)

{result['loadings_interpretation']['interpretation_text']}

### Interpretazione geometrica:
- **Distanza dall'origine** = importanza della variabile
- **Variabili vicine** = correlate positivamente  
- **Variabili opposte** = anticorrelate
- **Variabili ortogonali** = non correlate

---

## 📈 ANALISI SCORES (Distribuzione Campioni)

{result['scores_interpretation']['interpretation_text']}

### Interpretazione geometrica:
- **Campioni vicini** = caratteristiche simili
- **Campioni distanti** = caratteristiche diverse
- **Outliers** = campioni con caratteristiche atipiche
- **Clusters** = gruppi con profili comuni

---

## 🔗 INTERPRETAZIONE INTEGRATA

### Finding Principale:
{result['integrated_interpretation']['main_finding']}

### Collegamento Loadings-Scores:
{result['integrated_interpretation']['loading_score_link']}

### Spiegazione Outliers:
{result['integrated_interpretation']['outlier_explanation']}

---

## 💡 RACCOMANDAZIONI

{result['integrated_interpretation']['recommendations']}

---
"""

        # Add metavariable grouping analysis if provided
        if metavar_data is not None and metavar_name is not None:
            try:
                metavar_analysis = analyze_metavariable_grouping(
                    scores, metavar_data, metavar_name, pc_x, pc_y
                )
                report += f"\n{metavar_analysis}\n---\n"
            except Exception as e:
                logger.warning(f"Could not perform metavariable analysis: {str(e)}")
                report += f"\n⚠️ Note: Metavariable analysis could not be completed.\n\n---\n"

        report += """
## 📝 NOTE METODOLOGICHE

Questa interpretazione si basa sui principi geometrici della PCA:
- I loadings definiscono le nuove direzioni (PC) come combinazioni lineari delle variabili originali
- Gli scores sono le proiezioni dei campioni su queste nuove direzioni
- L'interpretazione congiunta rivela la struttura multivariata dei dati

Varianza spiegata e numero di componenti da considerare dipendono dal contesto applicativo.
"""
        
        # Aggiungi interpretazione specifica per tipo di dati
        if data_type.lower() in ['spettroscopia', 'nir', 'spectroscopy']:
            report += """

### 🔬 Note per Dati Spettroscopici:
- Loadings identificano lunghezze d'onda discriminanti
- Pattern nei loadings possono corrispondere a bande di assorbimento
- Correlazioni tra lunghezze d'onda vicine sono attese
"""
        elif 'process' in data_type.lower() or 'processo' in data_type.lower():
            report += """

### ⚙️ Note per Dati di Processo:
- Variabili dominanti indicano parametri critici di processo
- Clusters negli scores possono indicare diverse condizioni operative
- Outliers possono segnalare deviazioni di processo
"""
        
        return report
        
    except Exception as e:
        logger.error(f"Errore nell'analisi PCA: {str(e)}")
        return f"Errore nell'analisi: {str(e)}"

# =============================================================================
# SIMPLIFIED INTERFACE (NO AI NEEDED)
# =============================================================================

def quick_pca_interpretation(loadings_csv: str, scores_csv: str,
                           pc_x: int = 1, pc_y: int = 2) -> Tuple[bool, str]:
    """
    Interpretazione rapida PCA da dati CSV.
    
    Parameters:
    -----------
    loadings_csv : str
        CSV dei loadings
    scores_csv : str
        CSV degli scores
    pc_x, pc_y : int
        Componenti da analizzare
        
    Returns:
    --------
    Tuple[bool, str] : (successo, interpretazione)
    """
    
    try:
        from io import StringIO
        
        # Parse CSV
        loadings = pd.read_csv(StringIO(loadings_csv))
        scores = pd.read_csv(StringIO(scores_csv))
        
        # Set index se necessario
        if 'Variable' in loadings.columns:
            loadings = loadings.set_index('Variable')
        elif not loadings.index.name:
            loadings = loadings.set_index(loadings.columns[0])
            
        if not scores.index.name and scores.columns[0] not in [f'PC{i}' for i in range(1, 10)]:
            scores = scores.set_index(scores.columns[0])
        
        # Analizza
        interpretation = analyze_pca_complete(loadings, scores, pc_x, pc_y)
        
        return True, interpretation
        
    except Exception as e:
        return False, f"Errore: {str(e)}"

# =============================================================================
# LEGACY AI FUNCTIONS (KEPT FOR COMPATIBILITY)
# =============================================================================

def robust_ai_analysis(*args, **kwargs):
    """Mantenuta per compatibilità - reindirizza all'analisi locale."""
    logger.info("AI analysis redirected to local interpretation")
    if len(args) >= 4:
        return quick_pca_interpretation(args[0], args[1] if len(args) > 1 else "")
    return False, "Parametri insufficienti"

def run_loadings_analysis(loadings, pc_number=1, threshold=0.3, **kwargs):
    """Mantenuta per compatibilità - analizza solo loadings."""
    try:
        # Crea scores dummy per compatibilità
        dummy_scores = pd.DataFrame(
            np.random.randn(10, loadings.shape[1]),
            columns=loadings.columns
        )
        
        result = interpret_pca_geometry(loadings, dummy_scores, pc_number, pc_number+1, threshold)
        
        return {
            'success': result['success'],
            'result': result['loadings_interpretation']['interpretation_text']
        }
    except Exception as e:
        return {'success': False, 'result': str(e)}

# Mantieni le altre funzioni legacy per compatibilità...
analyze_loadings_local = lambda *args, **kwargs: {"error": "Use interpret_pca_geometry instead"}
run_integrated_pca_analysis = analyze_pca_complete
validate_pca_data = lambda l, s: (True, "Valid") if not l.empty and not s.empty else (False, "Empty data")

if __name__ == "__main__":
    print("PCA Interpretation Module v3.0")
    print("Pure geometric interpretation - no AI required")
    print("Based on established PCA theory and visual interpretation principles")