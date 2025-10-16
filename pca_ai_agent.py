"""
pca_claude_utils.py
==================
Estensione per usare Claude API nell'analisi PCA loadings
Alternativa/Complemento a OpenAI per interpretazione chemiometrica

Author: ChemometricSolutions Team  
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import json
import streamlit as st

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# CLAUDE API INTEGRATION
# =============================================================================

def create_claude_client(api_key: str, model: str = "claude-3-5-sonnet-20241022"):
    """
    Crea client per Claude API usando Datapizza-AI.
    
    Parameters:
    -----------
    api_key : str
        Anthropic API key
    model : str  
        Modello Claude da usare
        
    Returns:
    --------
    Client Claude o None se errore
    """
    try:
        from datapizza.clients.anthropic import AnthropicClient
        
        client = AnthropicClient(
            api_key=api_key,
            model=model
        )
        
        logger.info(f"Claude client created with model: {model}")
        return client
        
    except ImportError:
        logger.error("datapizza-ai-clients-anthropic not installed")
        return None
    except Exception as e:
        logger.error(f"Error creating Claude client: {str(e)}")
        return None

def analyze_loadings_with_claude(loadings_csv: str, pc_number: int, threshold: float, 
                                data_type: str, api_key: str) -> Tuple[bool, str]:
    """
    Analizza loadings PCA usando Claude API.
    
    Parameters:
    -----------
    loadings_csv : str
        Dati loadings in CSV
    pc_number : int
        Componente da analizzare
    threshold : float
        Soglia significatività
    data_type : str
        Tipo di dati
    api_key : str
        Anthropic API key
        
    Returns:
    --------
    Tuple[bool, str] : (success, analysis_result)
    """
    
    try:
        # Crea client Claude
        client = create_claude_client(api_key)
        if not client:
            return False, "❌ Impossibile creare client Claude. Verifica installazione datapizza-ai-clients-anthropic"
        
        # Prepara dati per analisi
        from io import StringIO
        df = pd.read_csv(StringIO(loadings_csv))
        
        # Prepara DataFrame
        if 'Variable' in df.columns:
            df_indexed = df.set_index('Variable')
        else:
            df_indexed = df.set_index(df.columns[0])
        
        # Analisi locale per preparare dati
        from pca_ai_utils import analyze_loadings_local
        stats = analyze_loadings_local(df_indexed, pc_number, threshold)
        
        if "error" in stats:
            return False, f"Errore preparazione dati: {stats['error']}"
        
        # Costruisci prompt per Claude
        prompt = create_claude_prompt(stats, pc_number, data_type, threshold)
        
        # Chiamata a Claude
        logger.info("Sending request to Claude API...")
        response = client.invoke(prompt)
        
        if hasattr(response, 'text'):
            result = response.text
        else:
            result = str(response)
        
        logger.info("Claude analysis completed successfully")
        return True, f"🧠 **Analisi Claude Expert:**\n\n{result}"
        
    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        return False, f"Errore Claude API: {str(e)}"

def create_claude_prompt(stats: Dict[str, Any], pc_number: int, data_type: str, threshold: float) -> str:
    """
    Crea prompt specializzato per Claude basato sui dati PCA.
    
    Parameters:
    -----------
    stats : Dict
        Statistiche loadings da analyze_loadings_local
    pc_number : int
        Numero componente
    data_type : str
        Tipo di dati
    threshold : float
        Soglia usata
        
    Returns:
    --------
    str : Prompt formattato per Claude
    """
    
    # Formatta top variabili per il prompt
    top_vars_text = "\n".join([
        f"- {var}: {val:.3f} ({'positivo' if val > 0 else 'negativo'})"
        for var, val in list(stats['top_variables'].items())[:10]
    ])
    
    prompt = f"""
Sei un esperto chemiometrista di fama mondiale. Analizza questi risultati di Principal Component Analysis (PCA) per dati di tipo: "{data_type}".

DATI ANALISI LOADINGS - COMPONENTE PRINCIPALE {pc_number}:

📊 STATISTICHE:
- Variabili totali nel dataset: {stats['total_variables']}
- Variabili significative (|loading| > {threshold}): {stats['significant_variables']}
- Loading massimo (valore assoluto): {stats['max_loading']:.3f}
- Loading medio: {stats['mean_loading']:.3f}
- Deviazione standard loadings: {stats['std_loading']:.3f}

📈 PATTERN LOADINGS:
- Variabili con loading positivo alto (>{threshold}): {stats['positive_high']}
- Variabili con loading negativo alto (<-{threshold}): {stats['negative_high']}

🎯 TOP VARIABILI INFLUENTI:
{top_vars_text}

COMPITO:
Fornisci un'interpretazione esperta e pratica di questi risultati. La tua analisi deve essere:

1. **SCIENTIFICAMENTE RIGOROSA**: Basata su principi chemiometrici consolidati
2. **CONTESTUALMENTE RILEVANTE**: Specifica per il tipo di dati "{data_type}"
3. **PRATICAMENTE UTILE**: Con suggerimenti actionable per l'analista
4. **CHIARA E STRUTTURATA**: Usa emoji e sezioni per organizzare la risposta

STRUTTURA RISPOSTA:
📊 **Interpretazione Statistica**: Cosa significano questi numeri
📈 **Pattern Identificati**: Cosa rivelano i loadings su questo componente  
🔬 **Significato Scientifico**: Interpretazione nel contesto "{data_type}"
💡 **Insights Pratici**: Cosa fare con queste informazioni
🚀 **Raccomandazioni**: Prossimi passi per approfondire l'analisi

Usa il tuo expertise per fornire insights che vanno oltre i numeri, spiegando il "perché" dietro i pattern osservati.
"""

    return prompt

# =============================================================================
# SISTEMA MULTI-API (CLAUDE + OPENAI)
# =============================================================================

def create_multi_api_analysis_system():
    """
    Crea sistema che supporta sia Claude che OpenAI con fallback intelligente.
    """
    
    class MultiAPIAnalyzer:
        def __init__(self):
            self.claude_available = False
            self.openai_available = False
            self._check_availability()
        
        def _check_availability(self):
            """Controlla quali API sono disponibili"""
            try:
                from datapizza.clients.anthropic import AnthropicClient
                self.claude_available = True
            except ImportError:
                pass
            
            try:
                from datapizza.clients.openai import OpenAIClient  
                self.openai_available = True
            except ImportError:
                pass
        
        def get_available_apis(self) -> Dict[str, bool]:
            """Restituisce API disponibili"""
            return {
                "claude": self.claude_available,
                "openai": self.openai_available
            }
        
        def analyze_with_preferred_api(self, loadings_csv: str, pc_number: int, 
                                     threshold: float, data_type: str, 
                                     claude_key: str = "", openai_key: str = "",
                                     preferred: str = "claude") -> Tuple[bool, str, str]:
            """
            Analizza con API preferita, fallback su altre se fallisce.
            
            Returns:
            --------
            Tuple[bool, str, str] : (success, result, api_used)
            """
            
            apis_to_try = []
            
            # Ordina API per preferenza
            if preferred == "claude" and claude_key and self.claude_available:
                apis_to_try.append(("claude", claude_key))
            if openai_key and self.openai_available:
                apis_to_try.append(("openai", openai_key))
            if preferred == "claude" and claude_key and self.claude_available:
                # Claude già aggiunto
                pass
            elif claude_key and self.claude_available:
                apis_to_try.append(("claude", claude_key))
            
            # Prova le API in ordine
            for api_name, api_key in apis_to_try:
                try:
                    logger.info(f"Trying {api_name} API...")
                    
                    if api_name == "claude":
                        success, result = analyze_loadings_with_claude(
                            loadings_csv, pc_number, threshold, data_type, api_key
                        )
                    else:  # openai
                        from pca_ai_utils import robust_ai_analysis
                        success, result = robust_ai_analysis(
                            loadings_csv, pc_number, threshold, data_type, api_key
                        )
                    
                    if success:
                        return True, result, api_name
                    else:
                        logger.warning(f"{api_name} API failed, trying next...")
                        
                except Exception as e:
                    logger.error(f"{api_name} API error: {str(e)}")
                    continue
            
            # Se tutte le API falliscono, usa analisi locale
            logger.info("All APIs failed, falling back to local analysis")
            from pca_ai_utils import run_loadings_analysis
            
            result = run_loadings_analysis(
                loadings=pd.read_csv(StringIO(loadings_csv)).set_index('Variable' if 'Variable' in pd.read_csv(StringIO(loadings_csv)).columns else pd.read_csv(StringIO(loadings_csv)).columns[0]),
                pc_number=pc_number,
                threshold=threshold, 
                data_type=data_type,
                api_key="",
                mode="local"
            )
            
            return result["success"], result["result"], "local"
    
    return MultiAPIAnalyzer()

# =============================================================================
# CONFRONTO CLAUDE VS OPENAI
# =============================================================================

def compare_ai_analyses(loadings_csv: str, pc_number: int, threshold: float,
                       data_type: str, claude_key: str, openai_key: str) -> Dict[str, Any]:
    """
    Confronta analisi di Claude vs OpenAI sugli stessi dati.
    
    Returns:
    --------
    Dict con risultati di entrambe le API per confronto
    """
    
    results = {
        "claude": {"success": False, "result": "", "error": ""},
        "openai": {"success": False, "result": "", "error": ""},
        "comparison": ""
    }
    
    # Analisi Claude
    if claude_key:
        try:
            success, result = analyze_loadings_with_claude(
                loadings_csv, pc_number, threshold, data_type, claude_key
            )
            results["claude"]["success"] = success
            results["claude"]["result"] = result
        except Exception as e:
            results["claude"]["error"] = str(e)
    
    # Analisi OpenAI
    if openai_key:
        try:
            from pca_ai_utils import robust_ai_analysis
            success, result = robust_ai_analysis(
                loadings_csv, pc_number, threshold, data_type, openai_key
            )
            results["openai"]["success"] = success  
            results["openai"]["result"] = result
        except Exception as e:
            results["openai"]["error"] = str(e)
    
    # Genera confronto se entrambe hanno successo
    if results["claude"]["success"] and results["openai"]["success"]:
        results["comparison"] = f"""
## 🔍 **Confronto Analisi AI**

### 🧠 **Claude Analysis Highlights:**
{results["claude"]["result"][:300]}...

### 🤖 **OpenAI Analysis Highlights:**  
{results["openai"]["result"][:300]}...

### 📊 **Osservazioni:**
- **Claude**: Tende a fornire analisi più dettagliate e contestuali
- **OpenAI**: Spesso più conciso e strutturato
- **Consenso**: Cerca pattern comuni tra le due interpretazioni
"""
    
    return results

# =============================================================================
# STREAMLIT UI COMPONENTS PER CLAUDE
# =============================================================================

def claude_api_interface():
    """
    Interfaccia Streamlit per configurazione Claude API.
    """
    
    st.markdown("#### 🧠 Claude API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        claude_key = st.text_input(
            "🔑 Anthropic API Key", 
            type="password",
            help="API key per Claude. Ottieni su: https://console.anthropic.com/",
            placeholder="sk-ant-..."
        )
        
        model_choice = st.selectbox(
            "📊 Modello Claude:",
            ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
            help="Sonnet: Bilanciato qualità/velocità, Opus: Massima qualità, Haiku: Veloce"
        )
    
    with col2:
        if st.button("ℹ️ Come ottenere API key"):
            st.info("""
            **Per ottenere Anthropic API key:**
            1. 🌐 Vai su https://console.anthropic.com/
            2. 📝 Crea account o fai login
            3. 💳 Aggiungi metodo di pagamento
            4. 🔑 Vai su "API Keys" e crea nuova key
            
            **💰 Costi tipici (approssimativi):**
            - Sonnet: ~$3 per 1M token
            - Haiku: ~$0.25 per 1M token  
            - 1 analisi = ~1,000 token = ~$0.003
            """)
        
        if claude_key:
            st.success("✅ Claude API key inserita!")
    
    return claude_key, model_choice

def multi_ai_selector():
    """
    Interfaccia per selezione tra Claude, OpenAI, o entrambi.
    """
    
    st.markdown("#### 🤖 AI Provider Selection")
    
    provider_choice = st.radio(
        "Scegli provider AI:",
        ["🧠 Solo Claude", "🤖 Solo OpenAI", "⚡ Claude Preferito (fallback OpenAI)", 
         "🔄 OpenAI Preferito (fallback Claude)", "🔍 Confronta Entrambi"],
        help="Claude è spesso migliore per analisi scientifiche, OpenAI per velocità"
    )
    
    # Input API keys basati sulla scelta
    claude_key = openai_key = ""
    
    if "Claude" in provider_choice or "Confronta" in provider_choice:
        claude_key = st.text_input("🧠 Anthropic API Key", type="password", 
                                  placeholder="sk-ant-...")
    
    if "OpenAI" in provider_choice or "Confronta" in provider_choice:
        openai_key = st.text_input("🤖 OpenAI API Key", type="password",
                                  placeholder="sk-...")
    
    return provider_choice, claude_key, openai_key

# =============================================================================
# INSTALLAZIONE E SETUP
# =============================================================================

def check_claude_dependencies() -> Tuple[bool, str]:
    """
    Controlla se le dipendenze per Claude sono installate.
    
    Returns:
    --------
    Tuple[bool, str] : (available, message)
    """
    
    try:
        import datapizza
        from datapizza.clients.anthropic import AnthropicClient
        return True, "✅ Claude integration disponibile!"
    
    except ImportError as e:
        missing_deps = []
        
        try:
            import datapizza
        except ImportError:
            missing_deps.append("datapizza-ai")
        
        try:
            from datapizza.clients.anthropic import AnthropicClient
        except ImportError:
            missing_deps.append("datapizza-ai-clients-anthropic")
        
        install_cmd = "pip install " + " ".join(missing_deps)
        
        message = f"""
❌ **Dipendenze Claude mancanti**

**Installa:**
```bash
{install_cmd}
```

**Oppure usa solo analisi locale** (sempre disponibile)
"""
        
        return False, message

def installation_guide():
    """
    Guida installazione completa per supporto Claude.
    """
    
    st.markdown("### 📦 Installation Guide")
    
    claude_available, claude_msg = check_claude_dependencies()
    
    st.markdown("#### Dipendenze AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🧠 Claude Support:**")
        if claude_available:
            st.success(claude_msg)
        else:
            st.warning(claude_msg)
    
    with col2:
        st.markdown("**🤖 OpenAI Support:**")
        try:
            from datapizza.clients.openai import OpenAIClient
            st.success("✅ OpenAI integration disponibile!")
        except ImportError:
            st.warning("❌ `pip install datapizza-ai-clients-openai`")
    
    # Comando installazione completo
    st.markdown("#### 🚀 Installazione Completa")
    
    install_cmd = """
pip install datapizza-ai
pip install datapizza-ai-clients-anthropic  # Per Claude
pip install datapizza-ai-clients-openai     # Per OpenAI
"""
    
    st.code(install_cmd, language="bash")
    
    if st.button("📋 Copia comando"):
        st.code(install_cmd.strip())
        st.success("✅ Comando copiato! Incolla nel terminal")

# =============================================================================
# TESTING E DEBUG
# =============================================================================

def test_claude_connection(api_key: str, model: str = "claude-3-5-sonnet-20241022") -> Tuple[bool, str]:
    """
    Testa connessione Claude API.
    """
    
    if not api_key:
        return False, "❌ API key mancante"
    
    try:
        client = create_claude_client(api_key, model)
        if not client:
            return False, "❌ Impossibile creare client Claude"
        
        # Test semplice
        test_prompt = "Rispondi solo con 'OK' se ricevi questo messaggio."
        response = client.invoke(test_prompt)
        
        if hasattr(response, 'text'):
            result = response.text.strip()
        else:
            result = str(response).strip()
        
        if "OK" in result.upper():
            return True, f"✅ Connessione Claude riuscita! Modello: {model}"
        else:
            return False, f"⚠️ Risposta inaspettata: {result[:100]}"
            
    except Exception as e:
        return False, f"❌ Errore connessione: {str(e)}"

if __name__ == "__main__":
    # Test del modulo Claude
    print("🧪 Testing Claude integration...")
    
    available, msg = check_claude_dependencies()
    print(msg)
    
    if available:
        print("✅ Claude integration ready!")
    else:
        print("⚠️ Install dependencies to enable Claude")