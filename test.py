import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx
import PyPDF2
from fuzzywuzzy import fuzz
import base64

# Configuration de la page
st.set_page_config(
    page_title="🎯 Système de Recrutement IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .cv-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    .score-high { color: #4CAF50; font-weight: bold; }
    .score-medium { color: #FF9800; font-weight: bold; }
    .score-low { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class CVProcessor:
    """Classe pour traiter et analyser les CV"""
    
    def __init__(self):
        self.competences_tech = [
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular',
            'node.js', 'django', 'flask', 'tensorflow', 'pytorch', 'scikit-learn',
            'pandas', 'numpy', 'excel', 'power bi', 'tableau', 'r', 'matlab',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'linux', 'windows',
            'git', 'jenkins', 'agile', 'scrum', 'machine learning', 'deep learning',
            'data science', 'big data', 'hadoop', 'spark', 'mongodb', 'postgresql'
        ]
        
        self.langues = ['français', 'anglais', 'espagnol', 'allemand', 'italien', 'chinois', 'arabe']
        
        self.diplomes = [
            'master', 'licence', 'bachelor', 'doctorat', 'phd', 'ingénieur', 'bts', 'dut',
            'mba', 'certificat', 'formation', 'école', 'université'
        ]
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extrait le texte d'un fichier PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.lower()
        except:
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extrait le texte d'un fichier DOCX"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.lower()
        except:
            return ""
    
    def extract_entities(self, text: str) -> Dict:
        """Extrait les entités importantes du CV"""
        text = text.lower()
        
        # Extraction des compétences
        competences_found = []
        for comp in self.competences_tech:
            if comp.lower() in text:
                competences_found.append(comp)
        
        # Extraction des langues
        langues_found = []
        for lang in self.langues:
            if lang.lower() in text:
                langues_found.append(lang)
        
        # Extraction des diplômes
        diplomes_found = []
        for diplome in self.diplomes:
            if diplome.lower() in text:
                diplomes_found.append(diplome)
        
        # Extraction de l'expérience (années)
        experience_pattern = r'(\d+)\s*(ans?|années?)\s*(d\'?expérience|expérience)'
        experience_matches = re.findall(experience_pattern, text)
        experience_years = max([int(match[0]) for match in experience_matches]) if experience_matches else 0
        
        return {
            'competences': competences_found,
            'langues': langues_found,
            'diplomes': diplomes_found,
            'experience_years': experience_years,
            'text': text
        }
    
    def calculate_matching_score(self, cv_data: Dict, job_requirements: Dict) -> Dict:
        """Calcule le score de correspondance entre un CV et une offre"""
        
        # Pondérations
        weights = {
            'competences': 0.4,
            'experience': 0.3,
            'diplomes': 0.2,
            'langues': 0.1
        }
        
        scores = {}
        
        # Score compétences
        job_competences = [comp.lower().strip() for comp in job_requirements.get('competences', [])]
        cv_competences = cv_data['competences']
        
        if job_competences:
            competences_match = len(set(cv_competences) & set(job_competences))
            scores['competences'] = min(100, (competences_match / len(job_competences)) * 100)
        else:
            scores['competences'] = 0
        
        # Score expérience
        required_exp = job_requirements.get('experience_min', 0)
        cv_exp = cv_data['experience_years']
        if required_exp > 0:
            scores['experience'] = min(100, (cv_exp / required_exp) * 100)
        else:
            scores['experience'] = 100 if cv_exp > 0 else 0
        
        # Score diplômes
        job_diplomes = [dip.lower() for dip in job_requirements.get('diplomes', [])]
        cv_diplomes = cv_data['diplomes']
        
        if job_diplomes:
            diplomes_match = len(set(cv_diplomes) & set(job_diplomes))
            scores['diplomes'] = min(100, (diplomes_match / len(job_diplomes)) * 100)
        else:
            scores['diplomes'] = 50
        
        # Score langues
        job_langues = [lang.lower() for lang in job_requirements.get('langues', [])]
        cv_langues = cv_data['langues']
        
        if job_langues:
            langues_match = len(set(cv_langues) & set(job_langues))
            scores['langues'] = min(100, (langues_match / len(job_langues)) * 100)
        else:
            scores['langues'] = 50
        
        # Score global
        global_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            'global_score': round(global_score, 1),
            'detail_scores': scores,
            'matched_competences': list(set(cv_competences) & set(job_competences)),
            'matched_langues': list(set(cv_langues) & set(job_langues))
        }

def main():
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Système de Recrutement Automatisé</h1>
        <p>Analysez et classez vos CV automatiquement grâce à l'IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation du processeur
    if 'processor' not in st.session_state:
        st.session_state.processor = CVProcessor()
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Section offre d'emploi
        st.subheader("📋 Offre d'emploi")
        
        job_title = st.text_input("Intitulé du poste", value="Développeur Python")
        
        # Compétences requises
        competences_input = st.text_area(
            "Compétences clés (séparées par des virgules)",
            value="python, django, sql, git, machine learning"
        )
        
        # Expérience minimale
        experience_min = st.number_input("Expérience minimale (années)", min_value=0, max_value=20, value=2)
        
        # Diplômes
        diplomes_input = st.text_area(
            "Diplômes requis (séparés par des virgules)",
            value="master, ingénieur"
        )
        
        # Langues
        langues_input = st.text_area(
            "Langues requises (séparées par des virgules)",
            value="français, anglais"
        )
        
        # Nombre de candidats à afficher
        top_n = st.slider("Nombre de candidats à afficher", min_value=5, max_value=50, value=10)
    
    # Traitement des requirements
    job_requirements = {
        'title': job_title,
        'competences': [comp.strip() for comp in competences_input.split(',') if comp.strip()],
        'experience_min': experience_min,
        'diplomes': [dip.strip() for dip in diplomes_input.split(',') if dip.strip()],
        'langues': [lang.strip() for lang in langues_input.split(',') if lang.strip()]
    }
    
    # Interface principale
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Import des CV")
        
        uploaded_files = st.file_uploader(
            "Téléchargez vos CV",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Formats acceptés: PDF, DOCX"
        )
        
        if st.button("🚀 Analyser les CV", type="primary"):
            if uploaded_files:
                with st.spinner("Analyse en cours..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, file in enumerate(uploaded_files):
                        # Extraction du texte
                        if file.type == "application/pdf":
                            text = st.session_state.processor.extract_text_from_pdf(file)
                        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text = st.session_state.processor.extract_text_from_docx(file)
                        else:
                            continue
                        
                        if text:
                            # Extraction des entités
                            cv_data = st.session_state.processor.extract_entities(text)
                            
                            # Calcul du score
                            matching_result = st.session_state.processor.calculate_matching_score(
                                cv_data, job_requirements
                            )
                            
                            results.append({
                                'filename': file.name,
                                'cv_data': cv_data,
                                'matching': matching_result
                            })
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Tri par score
                    results.sort(key=lambda x: x['matching']['global_score'], reverse=True)
                    st.session_state.analysis_results = results
                    
                st.success(f"✅ {len(results)} CV analysés avec succès!")
            else:
                st.warning("⚠️ Veuillez télécharger au moins un CV")
    
    with col2:
        st.subheader("📊 Résultats de l'analyse")
        
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            results = st.session_state.analysis_results[:top_n]
            
            # Métriques globales
            avg_score = np.mean([r['matching']['global_score'] for r in results])
            best_score = max([r['matching']['global_score'] for r in results])
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("📈 Score moyen", f"{avg_score:.1f}%")
            with col_m2:
                st.metric("🏆 Meilleur score", f"{best_score:.1f}%")
            with col_m3:
                st.metric("👥 Candidats analysés", len(st.session_state.analysis_results))
            
            # Graphique des scores
            fig = px.bar(
                x=[r['filename'][:20] + '...' if len(r['filename']) > 20 else r['filename'] for r in results],
                y=[r['matching']['global_score'] for r in results],
                title="🎯 Scores de correspondance des candidats",
                color=[r['matching']['global_score'] for r in results],
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Détails des candidats
            st.subheader(f"🏅 Top {len(results)} candidats")
            
            for i, result in enumerate(results):
                with st.expander(f"#{i+1} - {result['filename']} - Score: {result['matching']['global_score']:.1f}%"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("**📊 Scores détaillés:**")
                        for category, score in result['matching']['detail_scores'].items():
                            color_class = "score-high" if score >= 70 else "score-medium" if score >= 40 else "score-low"
                            st.markdown(f"- {category.title()}: <span class='{color_class}'>{score:.1f}%</span>", unsafe_allow_html=True)
                        
                        st.write(f"**⏰ Expérience:** {result['cv_data']['experience_years']} ans")
                    
                    with col_b:
                        st.write("**✅ Compétences correspondantes:**")
                        matched_comp = result['matching']['matched_competences']
                        if matched_comp:
                            for comp in matched_comp:
                                st.write(f"- {comp}")
                        else:
                            st.write("Aucune correspondance exacte")
                        
                        st.write("**🗣️ Langues correspondantes:**")
                        matched_lang = result['matching']['matched_langues']
                        if matched_lang:
                            for lang in matched_lang:
                                st.write(f"- {lang}")
                        else:
                            st.write("Aucune correspondance")
                    
                    # Barre de progression du score global
                    score = result['matching']['global_score']
                    progress_color = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
                    st.progress(score/100)
                    st.write(f"{progress_color} Score global: {score:.1f}%")
        
        else:
            st.info("👆 Téléchargez des CV et cliquez sur 'Analyser' pour voir les résultats")
    
    # Section statistiques avancées
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.subheader("📈 Statistiques avancées")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            # Distribution des scores
            scores = [r['matching']['global_score'] for r in st.session_state.analysis_results]
            fig_hist = px.histogram(
                x=scores,
                nbins=10,
                title="Distribution des scores de correspondance",
                labels={'x': 'Score (%)', 'y': 'Nombre de candidats'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_stat2:
            # Compétences les plus fréquentes
            all_competences = []
            for result in st.session_state.analysis_results:
                all_competences.extend(result['cv_data']['competences'])
            
            if all_competences:
                comp_counts = pd.Series(all_competences).value_counts().head(10)
                fig_comp = px.bar(
                    x=comp_counts.values,
                    y=comp_counts.index,
                    orientation='h',
                    title="Compétences les plus fréquentes",
                    labels={'x': 'Fréquence', 'y': 'Compétences'}
                )
                st.plotly_chart(fig_comp, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🎯 Système de Recrutement Automatisé | Développé Zouzou Freshnes Data scientist
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()