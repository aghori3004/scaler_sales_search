import os
import json
import time
import warnings
import pandas as pd
import difflib
from dotenv import load_dotenv
from typing import TypedDict, Optional
from pydantic import BaseModel, Field

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ==========================================
# 0. DATABASE & SEARCH ENGINE
# ==========================================
def load_databases():
    print("Loading Databases...")
    df = pd.read_parquet('cleaned_master.parquet')
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return df, vectorstore

def hybrid_search(query_json, df, vectorstore):
    print("Calculating Structured Scores...")
    df['struct_score'] = 0.0
    
    weights_tier_1 = {'pre_academy_company': 100, 'post_academy_company': 100, 'role': 100, 'pre_scaler_role': 100}
    weights_tier_2 = {'college': 80, 'degree': 80, 'branch': 80, 'experience': 80, 'current_city': 80, 'growth': 80}
    weights_tier_3 = {'pre_academy_segment': 60, 'post_academy_segment': 60, 'study_domain': 60, 'branch_category': 60, 'experience_zone': 60, 'growth_category': 60, 'seniority_level': 60, 'pre_seniority_level': 60}
    weights_tier_4 = {'college_tier': 40, 'current_city_tier': 40, 'origin_tier': 40, 'batch_year': 40, 'batch': 40, 'batch_category': 40}

    all_weights = {**weights_tier_1, **weights_tier_2, **weights_tier_3, **weights_tier_4}
    
    for field, weight in all_weights.items():
        val = query_json.get(field)
        if val is not None and val != "":
            if isinstance(val, str):
                safe_val = str(val).replace('+', r'\+').replace('(', r'\(').replace(')', r'\)')
                match_mask = df[field].astype(str).str.contains(safe_val, case=False, na=False)
                df.loc[match_mask, 'struct_score'] += weight
            elif isinstance(val, (int, float)):
                col_numeric = pd.to_numeric(df[field], errors='coerce')
                exact_mask = col_numeric == val
                close_mask = (abs(col_numeric - val) <= 1) & ~exact_mask
                df.loc[exact_mask, 'struct_score'] += weight
                df.loc[close_mask, 'struct_score'] += (weight / 2)

    df['struct_rank'] = df['struct_score'].rank(method='min', ascending=False)

    print("Calculating Semantic/Vector Scores...")
    df['vector_rank'] = len(df) 
    unstructured_intent = query_json.get('unstructured_intent')
    if unstructured_intent and str(unstructured_intent).strip() != "":
        results = vectorstore.similarity_search_with_score(str(unstructured_intent), k=150)
        for rank, (doc, score) in enumerate(results, start=1):
            df_index = doc.metadata.get('df_index')
            if df_index is not None and df_index in df.index:
                df.at[df_index, 'vector_rank'] = rank

    print("Fusing Results...")
    df['rrf_score'] = (1 / (50 + df['struct_rank'])) + (1 / (60 + df['vector_rank']))
    
    has_content_mask = (df['video_blog'].astype(str).str.strip() != "") | \
                       (df['quora_blog'].astype(str).str.strip() != "") | \
                       (df['linkedin_blog'].astype(str).str.strip() != "")
    
    df.loc[has_content_mask, 'rrf_score'] *= 1.15
    df['has_content'] = has_content_mask.astype(int) 
    
    sorted_df = df.sort_values(by=['rrf_score', 'has_content'], ascending=[False, False])
    top_50 = sorted_df[sorted_df['rrf_score'] > 0.001].head(50).copy()

    # Generate Deterministic Match Explanations showing EXACT DB Values
    detailed_calcs = []
    for _, row in top_50.iterrows():
        calc_md = f"**Detailed Scoring Breakdown:**\n\n"
        points_awarded = []
        total_struct = 0.0
        
        for field, val in query_json.items():
            weight = all_weights.get(field, 0)
            if val and weight > 0 and pd.notnull(row.get(field)):
                if isinstance(val, str) and str(val).lower() in str(row[field]).lower():
                    points_awarded.append(f"- **{field.replace('_', ' ').title()}:** +{weight} pts *(Matched '{row[field]}')*")
                    total_struct += weight
                elif isinstance(val, (int, float)):
                    num_val = pd.to_numeric(row[field], errors='coerce')
                    if pd.notnull(num_val) and abs(num_val - val) <= 1:
                        if num_val == val:
                            points_awarded.append(f"- **{field.replace('_', ' ').title()}:** +{weight} pts *(Exact match '{val}')*")
                            total_struct += weight
                        else:
                            points_awarded.append(f"- **{field.replace('_', ' ').title()}:** +{weight/2} pts *(Near match '{val}')*")
                            total_struct += (weight/2)
                            
        s_rank = row.get('struct_rank', 0)
        v_rank = row.get('vector_rank', 0)
        has_content = row.get('has_content', 0)
        
        calc_md += "► **Explicit Field Matches:**\n"
        calc_md += "\n".join(points_awarded) if points_awarded else "- None"
        calc_md += f"\n\n► **Total Structured Score:** `{total_struct}` -> resulting in **Rank {s_rank}**"
        calc_md += f"\n► **Semantic Vector Rank:** **{v_rank}**"
        
        boost_str = " * 1.15 (Content Boost)" if has_content else ""
        calc_md += f"\n\n► **Final RRF Equation:** `[{1:.4f} / (50 + {s_rank})] + [{1:.4f} / (60 + {v_rank})]{boost_str}`"
        calc_md += f"\n► **Final Score:** `{row.get('rrf_score', 0):.5f}`"
        
        detailed_calcs.append(calc_md)
        
    top_50['detailed_calc_markdown'] = detailed_calcs
    top_50['match_reason'] = [c.split("► **Total")[0].replace("► **Explicit Field Matches:**\n", "").strip() for c in detailed_calcs]
    return top_50

df_master, vectorstore_master = load_databases()

# ==========================================
# 1. GRAPH STATE & SCHEMA
# ==========================================
class AgentState(TypedDict):
    query: str
    raw_extraction: dict       
    mapped_extraction: dict    
    search_results: list       
    brochure_text: str         

class LeadExtraction(BaseModel):
    college: Optional[str] = Field(default=None)
    degree: Optional[str] = Field(default=None)
    branch: Optional[str] = Field(default=None)
    experience: Optional[float] = Field(default=None)
    batch_year: Optional[int] = Field(default=None)
    batch: Optional[str] = Field(default=None)
    pre_academy_company: Optional[str] = Field(default=None)
    pre_scaler_role: Optional[str] = Field(default=None)
    post_academy_company: Optional[str] = Field(default=None)
    role: Optional[str] = Field(default=None)
    growth: Optional[float] = Field(default=None)
    current_city: Optional[str] = Field(default=None, description="The ASPIRING city to live in for the lead.")
    program: Optional[str] = Field(default=None)
    
    college_tier: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    branch_category: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    experience_zone: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    pre_academy_segment: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    post_academy_segment: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    pre_seniority_level: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    seniority_level: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    growth_category: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    current_city_tier: Optional[str] = Field(default=None, description="Can be implicitly guessed but MUST strictly match unique valid values.")
    origin_tier: Optional[str] = Field(default=None, description="Lead's hometown or origin state. Can be implicitly guessed but MUST strictly match unique valid values.")
    
    unstructured_intent: Optional[str] = Field(default=None)

# ==========================================
# 2. GRAPH NODES
# ==========================================
def get_valid_values_context():
    def get_unique(file, col):
        path = os.path.join('db', file)
        if os.path.exists(path): return ", ".join(pd.read_csv(path)[col].dropna().unique().astype(str))
        return ""
    
    return f"""
    VALID UNIQUE VALUES (If you implicitly guess a category, it MUST exactly match one of these):
    - origin_tier: {get_unique('origin_tier_unique.csv', 'origin_tier')}
    - college_tier: {get_unique('college_mapping.csv', 'college_tier')}
    - branch_category: {get_unique('branch_mapping.csv', 'branch_category')}
    - experience_zone: {get_unique('experience_mapping.csv', 'experience_zone')}
    - pre_academy_segment: {get_unique('pre_academy_company_mapping.csv', 'pre_academy_segment')}
    - post_academy_segment: {get_unique('post_academy_company_mapping.csv', 'post_academy_segment')}
    - pre_seniority_level: {get_unique('pre_scaler_role_mapping.csv', 'pre_seniority_level')}
    - seniority_level: {get_unique('role_mapping.csv', 'seniority_level')}
    - growth_category: {get_unique('growth_mapping.csv', 'growth_category')}
    - current_city_tier: {get_unique('current_city_mapping.csv', 'current_city_tier')}
    """

def node_extract_raw(state: AgentState) -> dict:
    query = state["query"]
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key="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", 
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    parser = PydanticOutputParser(pydantic_object=LeadExtraction)
    
    system_prompt = f"""
    You are a data extraction API. Extract explicit entities and implicitly guess categories based on the user's query.
    CRITICAL RULE 1: You MAY implicitly guess categories (e.g. "from Orissa" -> origin_tier), BUT the datatype MUST perfectly match the unique values provided below.
    CRITICAL RULE 2: You MUST reply with ONLY valid JSON. Do not include markdown formatting (like ```json), do not include any conversational text. Return ONLY the raw JSON object.
    
    {get_valid_values_context()}
    
    {{format_instructions}}
    """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", "{query}")])
    chain = prompt | llm | parser
    
    for _ in range(3):
        try:
            return {"raw_extraction": chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()}).model_dump()}
        except Exception:
            time.sleep(2)
    return {"raw_extraction": {}}

def lookup_csv_with_llm(filename: str, key_col: str, val_col: str, search_val: str, llm) -> Optional[tuple]:
    if not search_val: return None
    path = os.path.join('db', filename)
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    
    match = df[df[key_col].astype(str).str.lower() == str(search_val).lower()]
    if not match.empty: return (str(match.iloc[0][val_col]), str(match.iloc[0][key_col]))
    
    valid_keys = df[key_col].dropna().astype(str).unique().tolist()
    closest_matches = difflib.get_close_matches(str(search_val), valid_keys, n=1, cutoff=0.85)
    if closest_matches: 
        best_match = closest_matches[0]
        return (str(df[df[key_col] == best_match].iloc[0][val_col]), best_match)
        
    try:
        prompt = f"Map the extracted value '{search_val}' to the exact matching correct entity from this list. \nList: {', '.join(valid_keys)}.\nReply with ONLY the exact string from the list. If there is NO logical match at all, reply with 'NONE'."
        response = llm.invoke([("user", prompt)])
        llm_match = response.content.strip()
        if llm_match in valid_keys:
            return (str(df[df[key_col] == llm_match].iloc[0][val_col]), llm_match)
    except Exception as e:
        pass
        
    return None

def node_map_categories(state: AgentState) -> dict:
    mapped = state["raw_extraction"].copy()
    mapping_rules = [
        ('college', 'college_tier', 'college_mapping.csv'),
        ('degree', 'study_domain', 'degree_mapping.csv'),
        ('branch', 'branch_category', 'branch_mapping.csv'),
        ('pre_academy_company', 'pre_academy_segment', 'pre_academy_company_mapping.csv'),
        ('post_academy_company', 'post_academy_segment', 'post_academy_company_mapping.csv'),
        ('pre_scaler_role', 'pre_seniority_level', 'pre_scaler_role_mapping.csv'),
        ('role', 'seniority_level', 'role_mapping.csv'),
        ('current_city', 'current_city_tier', 'current_city_mapping.csv'),
        ('batch_year', 'batch_category', 'batch_year_mapping.csv')
    ]
    
    # Ultra-low constraints to force the LLM to only output the exact dictionary string without hallucination/conversation
    llm_mapper = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key="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", 
        temperature=0.0,
        top_p=0.1,
        max_tokens=1024,
    )
    
    for raw_f, cat_f, filename in mapping_rules:
        if mapped.get(raw_f) and not mapped.get(cat_f):
            result = lookup_csv_with_llm(filename, raw_f, cat_f, mapped[raw_f], llm_mapper)
            if result: 
                mapped[cat_f] = result[0]
                mapped[raw_f] = result[1]

    exp, growth = mapped.get('experience'), mapped.get('growth')
    if exp is not None and not mapped.get('experience_zone'):
        mapped['experience_zone'] = "Beginner" if exp < 5 else "Intermediate" if exp <= 10 else "Expert"
    if growth is not None and not mapped.get('growth_category'):
        mapped['growth_category'] = "Low Growth" if growth < 120 else "Moderate Growth" if growth <= 190 else "High Growth"
        
    return {"mapped_extraction": mapped}

def node_search(state: AgentState) -> dict:
    mapped_data = state["mapped_extraction"]
    results_df = hybrid_search(mapped_data, df_master, vectorstore_master)
    results_list = results_df.where(pd.notnull(results_df), None).to_dict(orient='records')
    return {"search_results": results_list}

search_workflow = StateGraph(AgentState)
search_workflow.add_node("extract", node_extract_raw)
search_workflow.add_node("map", node_map_categories)
search_workflow.add_node("search", node_search)
search_workflow.set_entry_point("extract")
search_workflow.add_edge("extract", "map")
search_workflow.add_edge("map", "search")
search_workflow.add_edge("search", END)
search_engine = search_workflow.compile()

def generate_brochure_from_selection(query: str, selected_profiles: list) -> str:
    if not selected_profiles: return "No suitable alumni matches found."
    profiles_text = ""
    for i, alum in enumerate(selected_profiles, 1):
        profiles_text += f"\n### Profile {i}: {alum.get('name', 'Alumni')}\n"
        profiles_text += f"- **Education Details:** {alum.get('degree')} in {alum.get('branch')} from {alum.get('college')} ({alum.get('college_tier')}). Location: {alum.get('current_city')} ({alum.get('current_city_tier')}). Origin: {alum.get('origin_tier')}\n"
        profiles_text += f"- **Pre-Scaler Background:** {alum.get('experience')} Yrs Exp ({alum.get('experience_zone')}). Role: {alum.get('pre_scaler_role')} ({alum.get('pre_seniority_level')}) at {alum.get('pre_academy_company')} ({alum.get('pre_academy_segment')})\n"
        profiles_text += f"- **Post-Scaler Transition:** Role: {alum.get('role')} ({alum.get('seniority_level')}) at {alum.get('post_academy_company')} ({alum.get('post_academy_segment')}). Growth factor: {alum.get('growth')} ({alum.get('growth_category')})\n"
        if alum.get('linkedin'): profiles_text += f"- **LinkedIn:** {alum.get('linkedin')}\n"
        if alum.get('video_blog'): profiles_text += f"- **Video:** {alum.get('video_blog')}\n"
        if alum.get('quora_blog'): profiles_text += f"- **Quora:** {alum.get('quora_blog')}\n"

    # High creativity and length allowances to enable the model to draft a persuasive, fluid marketing sales pitch
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key="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", 
        temperature=0.7,
        top_p=0.7,
        max_tokens=1024,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert sales copywriter. Output a highly detailed, personalized pitch brochure for the selected alumni. Detail their complete journey (Education, Pre-Scaler struggles/background, and Post-Scaler success). DO NOT include generic fluff, but DO write compelling paragraph descriptions for each alum incorporating all the provided detailed data points (Tiers, Segments, Growth, etc). You MUST explicitly include all the URLs provided for each alum in Markdown."),
        ("user", "Lead Context: {lead_context}\n\nProfiles Data:\n{profiles}\n\nWrite the detailed pitch.")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"lead_context": query, "profiles": profiles_text})

def run_search_pipeline(query: str):
    return search_engine.invoke({"query": query})