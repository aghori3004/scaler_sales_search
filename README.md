# Scaler Sales AI | Brochure Generator

A powerful Streamlit-based web application that streamlines the sales process by leveraging AI to map, search, and rank alumni profiles from a Scaler database based on lead context. It automatically drafts personalized brochures from selected matching alumni to help the sales team close leads effectively.

---

## 🔗 Live Deployment

**[Placeholder for Deployment Link]** *(Update this link once the application is deployed)*

---

## 🌟 Key Features

- **Contextual Semantic Search**: Uses Langchain and FAISS to search through a vector database of alumni.
- **AI-Powered Extraction**: Maps user queries (lead context) to structured search parameters using AI models.
- **Intelligent Ranking System**: Employs Reciprocal Rank Fusion (RRF) to score and rank alumni based on structured exact matches and semantic vector proximity.
- **Dynamic Brochure Generation**: Automatically drafts highly personalized, detailed brochures using the selected alumni profiles.
- **Interactive UI**: View calculation breakpoints, detailed metrics, and available assets (LinkedIn, Video Testimonials, etc.) for each alum.

## 📂 Project Structure

```
Scaler_Search/
│
├── app.py                      # Main Streamlit application entry point
├── graph_pipeline.py           # Core backend logic (search pipeline, RRF, brochure generation)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables file (requires NVIDIA_API_KEY)
├── .gitignore                  # Git ignore rules
├── cleaned_master.parquet      # Main structured data store
├── faiss_index/                # Directory containing the FAISS vector database
└── db/                         # Auxiliary local database files
```

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.9+** (preferably 3.10 or 3.11)
- **pip** (Python package installer)
- Git (if pulling from a repository)

You will also need an **NVIDIA API Key** to use the `langchain-nvidia-ai-endpoints`.

## 🚀 Local Setup & Installation

Follow these steps to get your development environment running locally:

**1. Clone the repository (or navigate to your local folder):**
```bash
cd Scaler_Search
```

**2. Create and activate a Virtual Environment (Recommended):**
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up Environment Variables:**
Create a `.env` file in the root directory (if not present) and add your NVIDIA API Key:
```ini
NVIDIA_API_KEY="your_nvidia_api_key_here"
```

## 💻 Running the Application Locally

Start the Streamlit development server by running:
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

## ☁️ Deployment Instructions

### Option A: Streamlit Community Cloud (Recommended & Easiest)
1. Push this entire project to a public or private GitHub repository. (Ensure `faiss_index` and `cleaned_master.parquet` are pushed if they are within Git size limits, otherwise you may need Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New App"** and select your repository, branch, and set the main file path to `app.py`.
4. **IMPORTANT:** Before clicking "Deploy", go to **Advanced Settings**. Under the "Secrets" section, add your environment variable:
   ```toml
   NVIDIA_API_KEY="your_nvidia_api_key_here"
   ```
5. Click **Deploy!** 

### Option B: Deploying on Render / Railway
1. Push your code to GitHub.
2. Sign up on [Render](https://render.com/) or [Railway](https://railway.app/).
3. Create a **New Web Service** and link your repository.
4. Set the **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
5. Set the **Start Command**:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
6. Add the `NVIDIA_API_KEY` to the service's Environment Variables settings.
7. Deploy the service.

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** - The web framework used for the frontend interface.
- **[LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)** - For AI orchestration and pipelines.
- **[NVIDIA AI Endpoints](https://build.nvidia.com/)** - For LLM inferences and semantic extractions.
- **[FAISS](https://github.com/facebookresearch/faiss)** - For blazing-fast vector similarity search.
- **[Pandas](https://pandas.pydata.org/) & [FastParquet](https://fastparquet.readthedocs.io/)** - For high-performance structured dataset handling.

## 💡 Usage Guide

1. **Enter Lead Context:** Type in your lead's details (e.g., *"Someone from Orissa wants to join Google"*).
2. **Search:** Click "Search Alumni Database". The AI will map the search, extract relevant entities, and fetch results using RRF scoring.
3. **Select Profiles:** Check the boxes next to the alumni profiles that best match the context. You can view deeper details by expanding the score popovers.
4. **Generate Brochure:** Scroll down and click 'Generate Brochure with Selected Profiles' to receive a tailor-made final document that can be directly shared with your lead.
