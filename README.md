<h1>FrameBuddy</h1>
<p>
FrameBuddy is a multimodal artwork retrieval and description system that identifies visually similar artworks from The Metropolitan Museum of Art's public domain collection. It uses SigLIP for image embeddings, FAISS for similarity search, and a vision‑language model to generate descriptive output.
</p>

<hr>

<h2>Features</h2>
<ul>
  <li>Image similarity search using FAISS</li>
  <li>SigLIP Vision Transformer embeddings</li>
  <li>Vision‑language model for final description generation</li>
  <li>Asynchronous ingestion pipeline for efficient dataset building</li>
  <li>Metadata and embedding storage in JSONL and NumPy formats</li>
  <li>Modular architecture for easy debugging and extension</li>
</ul>

<hr>

<h2>Installation</h2>

<h3>1. Clone the repository</h3>
<pre>
git clone https://github.com/&lt;your-username&gt;/framebuddy.git
cd framebuddy
</pre>

<h3>2. Create and activate a virtual environment</h3>
<pre>
python -m venv .venv
</pre>

<p>Activate it (Windows PowerShell):</p>
<pre>
.venv\Scripts\activate
</pre>

<hr>

<h2>Required Dependencies</h2>
<p>Install all dependencies inside your virtual environment.</p>

<h3>Core libraries</h3>
<pre>
pip install "transformers==4.39.3"
pip install "huggingface_hub==0.22.2"
</pre>

<h3>PyTorch (CPU build for Windows)</h3>
<pre>
pip install torch --index-url https://download.pytorch.org/whl/cpu
</pre>

<h3>Image processing</h3>
<pre>
pip install pillow
</pre>

<h3>Async and HTTP</h3>
<pre>
pip install aiohttp
pip install asynciolimiter
</pre>

<h3>Vector search (FAISS CPU)</h3>
<pre>
pip install faiss-cpu
</pre>

<h3>Utilities</h3>
<pre>
pip install numpy
pip install tqdm
</pre>

<hr>

<h2>Project Structure</h2>

<pre>
Final Project/
│
├── run.py
├── src/
│   ├── ingest.py          # Downloads MET images and metadata, builds embeddings
│   ├── indexer.py         # Builds FAISS index
│   ├── controller.py      # Orchestrates retrieval and LLM reasoning
│   ├── embedder.py        # SigLIP embedding logic
│   ├── preprocessing.py   # Image loading
│   ├── retrieval.py       # Metadata lookup
│   └── data/
│       ├── embeddings.npy
│       ├── metadata.jsonl
│       └── index.faiss
</pre>

<hr>

<h2>Building the Dataset</h2>

<h3>1. Run the ingestion pipeline</h3>
<p>This downloads MET artworks, filters them, embeds them, and saves:</p>
<ul>
  <li>metadata.jsonl</li>
  <li>embeddings.npy</li>
</ul>

<pre>
python -m src.ingest
</pre>

<h3>2. Build the FAISS index</h3>
<pre>
python -m src.indexer
</pre>

<hr>

<h2>Running FrameBuddy</h2>

<p>Use the command‑line interface:</p>

<pre>
python run.py "path/to/your/image.jpg"
</pre>

<p>Example:</p>

<pre>
python run.py "C:\Users\YourName\Downloads\image.jpg"
</pre>

<p>The system will:</p>
<ol>
  <li>Embed the input image</li>
  <li>Retrieve nearest artworks from FAISS</li>
  <li>Load corresponding metadata</li>
  <li>Generate a final description using the vision‑language model</li>
</ol>

<hr>

<h2>How It Works</h2>

<h3>Embedding</h3>
<p>SigLIP Vision Transformer converts the input image into a 768‑dimensional embedding.</p>

<h3>Retrieval</h3>
<p>FAISS performs nearest‑neighbor search over MET embeddings.</p>

<h3>Metadata</h3>
<p>Matching metadata is pulled from metadata.jsonl.</p>

<h3>LLM Reasoning</h3>
<p>A vision‑language model generates a final description using the input image and retrieved metadata.</p>

<hr>

<h2>Troubleshooting</h2>

<h3>IndexError: list index out of range</h3>
<p>Your FAISS index and metadata are out of sync. Delete the contents of <code>src/data/</code> and rebuild:</p>
<pre>
python -m src.ingest
python -m src.indexer
</pre>

<h3>ValueError: image contains values outside [0,1]</h3>
<p>Ensure preprocessing returns a PIL image, not a normalized tensor.</p>

<h3>ModuleNotFoundError: No module named 'faiss'</h3>
<p>Install the Windows version:</p>
<pre>
pip install faiss-cpu
</pre>

<hr>

<h2>License</h2>
<p>MIT License (or your preferred license).</p>
