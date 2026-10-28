# chat-with-pdf
# 📄 Chat with PDF

Upload a PDF, ask questions about it in plain English, and get answers from **Google Gemini** with **page references** so you can verify every answer.

Built with [Streamlit](https://streamlit.io), [pypdf](https://pypdf.readthedocs.io) and the [Google Gen AI SDK](https://github.com/googleapis/python-genai).

## Features

- Upload a PDF and extract its text page by page
- Ask questions and get answers grounded **only** in the document
- Page citations like `[Page 3]`, plus an expandable excerpt of each cited page
- Chat history with follow-up questions ("and what about the second one?")
- Handles large documents: small PDFs are sent to Gemini in full, large ones use keyword search (BM25) to pick the most relevant excerpts
- Friendly error messages: corrupted, password-protected, scanned or oversized PDFs, missing or invalid API key, rate limits, wrong model name, network problems, empty/blocked answers
- Basic prompt-injection resistance (the model is told to treat the PDF as untrusted data)

## How it works

```
PDF ──► pypdf extracts text per page
            │
            ▼
   small PDF? ── yes ──► send all pages to Gemini
            │ no
            ▼
   split into chunks ► rank with BM25 ► send top excerpts
            │
            ▼
   Gemini answers using only that text and cites [Page N]
            │
            ▼
   App validates the cited pages and shows them as sources
```

## Limitations (good things to mention in interviews)

- **Scanned PDFs (images of text) are not supported.** There is no OCR. The app tells the user when no text can be extracted.
- Retrieval for very large PDFs is keyword-based (BM25), not semantic. A natural upgrade is embeddings plus a vector store.
- Tables and multi-column layouts may be extracted in a jumbled order by pypdf.
- Chat history lives in the browser session only. It is lost when the page is refreshed.

## Project files

| File | Purpose |
|---|---|
| `app.py` | The whole application |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

---

## 1. Get a Gemini API key

1. Go to **[Google AI Studio → API keys](https://aistudio.google.com/apikey)**.
2. Sign in with your Google account and accept the terms.
3. Click **Create API key**, choose an existing Google Cloud project or create a new one.
4. Copy the key and keep it secret. **Never put it in your code or upload it to GitHub.**

A free tier is available, with rate limits. If you see a "rate limit" message in the app, wait a minute and try again. Check Google's current pricing and data-usage terms before uploading confidential documents, because free-tier data handling can differ from paid tiers.

## 2. Run the app locally

You need **Python 3.10 or newer**. Check with `python --version`.

```bash
# 1. Open a terminal in the project folder (the one containing app.py)

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows (Command Prompt or PowerShell)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

**Giving the app your API key (pick one):**

- **Easiest:** paste it into the *Gemini API key* box in the sidebar. It is kept only in memory for your session.
- **Or** create a file `.streamlit/secrets.toml` next to `app.py`:

  ```toml
  GEMINI_API_KEY = "paste-your-key-here"
  ```

  If you use Git, add `.streamlit/secrets.toml` to a `.gitignore` file so it is never uploaded.

**Optional:** choose a different model with `GEMINI_MODEL = "model-name"` in the same secrets file, or type it into the *Gemini model* box in the sidebar. The default is `gemini-3.7-flash`. Google renames and retires models regularly, so if you see a "model name was not found" error, pick a current Flash model from the [model list](https://ai.google.dev/gemini-api/docs/models).

**Try it:** upload any text-based PDF (a paper, a manual, a report) and ask "Summarize this document" or "What does it say about X?".

## 3. Upload the project to GitHub (website only, no Git needed)

1. Sign in at [github.com](https://github.com) (create a free account if needed).
2. Click **+** (top right) → **New repository**.
3. Enter a name such as `chat-with-pdf`, choose **Public** (Streamlit Community Cloud's free plan works best with public repos), and click **Create repository**.
4. On the new repository page, click **uploading an existing file** (or **Add file → Upload files**).
5. Drag in `app.py`, `requirements.txt` and `README.md`. The files must be at the **top level** of the repository, not inside a folder.
6. Scroll down and click **Commit changes**.

✅ Double-check that your API key is **not** in any uploaded file.

## 4. Deploy on Streamlit Community Cloud (free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account. Allow access to your repositories when asked.
2. Click **Create app** and choose to deploy from an existing GitHub repository.
3. Fill in:
   - **Repository:** `your-username/chat-with-pdf`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** and paste your key into the **Secrets** box:

   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```

5. Click **Deploy**. The first build takes a few minutes while dependencies install. You then get a public URL like `https://your-app-name.streamlit.app`. Add it to your GitHub repo description and your portfolio.

Button and menu names on these websites change from time to time, so follow the closest equivalent if a label looks different.

**Security note:** with the key stored in Secrets, anyone who opens your public app uses *your* quota. For a portfolio demo, consider adding usage limits in Google AI Studio or leaving the secret out so visitors paste their own key in the sidebar (the app supports both).

To update the deployed app, edit or re-upload the file on GitHub. Streamlit Cloud redeploys automatically. To change secrets later, use the app's **Settings → Secrets** in the Streamlit dashboard.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Gemini rejected the API key" | Re-copy the key from AI Studio; check for extra spaces |
| "model name was not found" | Change the *Gemini model* in the sidebar to a current Flash model |
| "rate limit or free quota reached" | Wait a minute, or enable billing for higher limits |
| "No text could be extracted" | The PDF is scanned images. Run OCR on it first |
| "password-protected" | Remove the PDF password and upload again |
| Chat box is greyed out | Upload a PDF and provide an API key |
| `ModuleNotFoundError` locally | Activate your virtual environment and run `pip install -r requirements.txt` |

## What has and has not been tested

**Tested (automated, in a sandbox):**

- PDF text extraction with pypdf on generated PDFs: normal, blank/scanned-like, corrupted, truncated, empty file, password-protected, owner-locked (opens with empty password), and over the page limit
- Chunking, BM25 ranking, full-document vs. retrieval mode, follow-up query handling, page ordering
- Citation parsing: single pages, lists, ranges, invalid or invented page numbers are dropped
- Gemini request building and error handling using a **simulated** Gemini client: roles, system prompt, history truncation, retries on server errors (503), and friendly messages for 400/401/403/404/429, network errors and empty answers
- The full app flow (upload, ask, replay history, replace/remove file, missing key, key from secrets, oversize/corrupt/scanned files, API failures) using a **simulated** Streamlit

**Not tested (could not be done in the sandbox):**

- Running the app under the real Streamlit library, in a real browser (the sandbox could not install Streamlit)
- Any real call to the Gemini API: no API key, and the real `google-genai` package could not be installed. The code follows Google's documented `client.models.generate_content` usage, but the real error classes, the default model name `gemini-3.7-flash` and your account's access to it are unverified
- Installing `requirements.txt` from scratch, and the tested pypdf version (6.x) is newer than the minimum pinned
- Deployment to Streamlit Community Cloud and the GitHub upload steps
- Real-world PDFs with complex layouts, non-English text, very large files and concurrent users

Run the app once with your own key and a sample PDF before showing it to anyone.

## Ideas to extend the project

- Semantic search with embeddings and a vector store (FAISS or Chroma)
- OCR for scanned PDFs
- Multiple PDFs at once
- Streaming answers
- Highlighting the cited text inside a rendered PDF page
- A small evaluation set of questions with expected answers

## License

MIT. Use and modify freely.
