import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from graph_pipeline import run_search_pipeline, generate_brochure_from_selection

load_dotenv()

st.set_page_config(page_title="Scaler Sales AI | Brochure Generator", layout="wide")

# Initialize Session States
if 'search_state' not in st.session_state:
    st.session_state.search_state = None
if 'brochure_content' not in st.session_state:
    st.session_state.brochure_content = None

# ==========================================
# VIEW 2: THE BROCHURE VIEW (NEW SCREEN)
# ==========================================
if st.session_state.brochure_content:
    st.title("📄 Your Personalized Brochure")
    
    # Back Button
    if st.button("⬅️ Back to Search Results", type="secondary"):
        st.session_state.brochure_content = None
        st.rerun()
        
    st.success("✅ Personalized Brochure Generated Successfully!")
    
    # Clean container for the final text
    with st.container(border=True):
        st.markdown(st.session_state.brochure_content)

# ==========================================
# VIEW 1: THE MAIN SEARCH & SELECTION UI
# ==========================================
else:
    st.title("🎯 Scaler AI Brochure Generator")

    with st.container():
        query = st.text_area("Enter Lead Context:", height=100, placeholder="e.g. someone from orissa wants to join google")
        
        if st.button("🔍 Search Alumni Database", type="primary"):
            with st.spinner("🤖 AI is extracting, mapping, and searching..."):
                st.session_state.search_state = run_search_pipeline(query)

    if st.session_state.search_state:
        state = st.session_state.search_state
        
        with st.expander("👀 View AI Extracted & Mapped Categories"):
            st.json(state.get("mapped_extraction", {}))

        st.divider()
        results = state.get("search_results", [])
        st.subheader(f"Found {len(results)} Matches (ranked by relevance)")
        
        with st.form("brochure_form"):
            selected_alumni = []
            
            for idx, row in enumerate(results[:50]): 
                col_check, col_img, col_details = st.columns([0.05, 0.15, 0.8])
                
                with col_check:
                    st.write(""); st.write("") 
                    if st.checkbox("", key=f"check_{idx}"):
                        selected_alumni.append(row)
                
                with col_img:
                    img_url = row.get('image_url')
                    if img_url and str(img_url) != 'None' and str(img_url) != 'nan':
                        st.image(img_url, width=100)
                    else:
                        st.image("https://via.placeholder.com/100?text=Alum", width=100)

                with col_details:
                    # Dynamic Exact Calculation Strings
                    s_rank = row.get('struct_rank', 0)
                    v_rank = row.get('vector_rank', 0)
                    boost_str = " * 1.15 (Content Boost)" if row.get('has_content') else ""
                    calc_equation = f"[{1:.4f} / (50 + {s_rank})] + [{1:.4f} / (60 + {v_rank})]{boost_str}"
                    
                    header_col1, header_col2 = st.columns([0.8, 0.2])
                    with header_col1:
                        st.markdown(f"### {row.get('name', 'Alumni')}")
                    with header_col2:
                        st.write("") 
                        # Advanced Popover logic showing the exact math and explaining the K constants
                        with st.popover(f"ℹ️ Score: {row.get('rrf_score', 0):.4f}"):
                            st.markdown(f"**Detailed Calculation Breakdown:**\n\n"
                                        f"► **Equation:** `RRF = {calc_equation}`\n\n"
                                        f"*(Note: 50 and 60 are 'k' smoothing constants in the Reciprocal Rank Fusion formula. The 50 denominator gives slightly more weight to structured exact matches, while the 60 denominator gracefully balances semantic intent rankings.)*\n\n"
                                        f"► **Total RRF Score:** `{row.get('rrf_score', 0):.5f}`\n"
                                        f"► **Structured Points:** `{row.get('struct_score', 0)}` (Rank: {s_rank})\n"
                                        f"► **Semantic Vector Rank:** `{v_rank}`")
                    
                    st.caption(f"Program: {row.get('program_label', 'N/A')} | Batch: {row.get('batch_year', '')} {row.get('batch', '')}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"🎓 **Education & Origin**<br>College: {row.get('college')} ({row.get('college_tier', 'N/A')})<br>Degree: {row.get('degree')} in {row.get('branch')}<br>Location: {row.get('current_city')} ({row.get('current_city_tier', 'N/A')})<br>Origin: {row.get('origin_tier', 'N/A')}", unsafe_allow_html=True)
                    c2.markdown(f"💼 **Before Scaler**<br>Company: {row.get('pre_academy_company')} ({row.get('pre_academy_segment', 'N/A')})<br>Role: {row.get('pre_scaler_role')} ({row.get('pre_seniority_level', 'N/A')})<br>Exp: {row.get('experience')} Yrs ({row.get('experience_zone', 'N/A')})", unsafe_allow_html=True)
                    c3.markdown(f"🚀 **After Scaler**<br>Company: {row.get('post_academy_company')} ({row.get('post_academy_segment', 'N/A')})<br>Role: {row.get('role')} ({row.get('seniority_level', 'N/A')})<br>Growth: {row.get('growth')} ({row.get('growth_category', 'N/A')})", unsafe_allow_html=True)
                    
                    links = []
                    if str(row.get('linkedin')).strip() and str(row.get('linkedin')) != 'None': links.append(f"[🔗 LinkedIn Profile]({row['linkedin']})")
                    if str(row.get('video_blog')).strip() and str(row.get('video_blog')) != 'None': links.append(f"[🎥 Video Testimonial]({row['video_blog']})")
                    if str(row.get('quora_blog')).strip() and str(row.get('quora_blog')) != 'None': links.append(f"[❓ Quora Answer]({row['quora_blog']})")
                    
                    st.markdown("**Available Assets:** " + (" | ".join(links) if links else "_None_"), unsafe_allow_html=True)
                    
                    st.info(f"💡 **Why this matched:**\n{row.get('match_reason', '')}")
                    st.divider()
            
            # The Generate button uses session state to trigger the screen change
            st.markdown("""<style>div[data-testid="stFormSubmitButton"] { position: fixed; bottom: 20px; right: 20px; z-index: 999; }</style>""", unsafe_allow_html=True)
            generate_btn = st.form_submit_button("📝 Generate Brochure with Selected Profiles", type="primary")

        if generate_btn:
            if not selected_alumni:
                st.warning("Please select at least one alum to generate a brochure.")
            else:
                with st.spinner("Drafting highly detailed personalized brochure..."):
                    # Save the result to session state and rerun to trigger the "New Screen" view
                    st.session_state.brochure_content = generate_brochure_from_selection(state["query"], selected_alumni)
                    st.rerun()