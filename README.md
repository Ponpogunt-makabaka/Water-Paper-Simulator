Of course. Here is a complete, updated English version of the `README.md` file. It reflects the project's current architecture, including the advanced RAG pipeline and detailed setup instructions.

Please copy the content below and replace the content of your `README.md` file with it.

-----

# 🤖 AI-Powered Research Paper Generation System

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)

This project is an automated research paper generation system built upon a **Multi-Agent** and **Retrieval-Augmented Generation (RAG)** architecture. It simulates a complete academic research workflow, from topic ideation, literature review, and analysis to drafting and peer review, ultimately producing a well-structured and context-rich academic paper.

-----

## Core Architecture

The standout feature of this system is its **RAG-driven Multi-Agent workflow**. Each agent has a distinct responsibility, and they collaborate like an efficient research team:

1.  **🧠 The Researcher (ResearcherAgent)**

      * **Responsibility**: To conduct the initial exploration phase of the research.
      * **Workflow**:
          * Brainstorms innovative research angles based on an initial topic.
          * Performs an extensive literature search using the **Semantic Scholar API**.
          * Automatically downloads open-access PDFs and parses them into structured text using **GROBID**.
          * Finally, builds a **Vector Store** from the full text of all sourced papers, creating the knowledge base for the RAG pipeline.

2.  **📊 The Analyst (AnalystAgent)**

      * **Responsibility**: To evaluate research directions and structure the paper's outline.
      * **Workflow**:
          * Evaluates the multiple research plans proposed by the Researcher.
          * Selects the most promising final topic, supported by the literature review.
          * Creates a detailed, structured outline for the chosen topic.

3.  **✍️ The Writer (WriterAgent)**

      * **Responsibility**: To perform high-quality, context-aware content generation using RAG.
      * **Workflow**:
          * When drafting each section (e.g., "Methodology," "Literature Review"), it queries the Vector Store built by the Researcher.
          * Retrieves the most relevant **full-text content** from the literature to use as context.
          * Synthesizes this precise and detailed context to write each section, ensuring the paper has depth and accuracy.

4.  **🧐 The Reviewers (ReviewerAgents)**

      * **Responsibility**: To simulate peer review and provide constructive feedback for revisions.
      * **Workflow**:
          * **Breadth Reviewer**: Checks the paper's macro-level structure, logical coherence, and completeness.
          * **Depth Reviewer**: Scrutinizes technical details, methodological rigor, and the validity of the data.

5.  **🔁 The Triage Router (TriageRouter)**

      * **Responsibility**: To control the iterative workflow.
      * **Workflow**: Based on the reviewers' feedback, it decides whether to send the paper back for rewriting, further research, or to conclude the process.

-----

## Getting Started

Follow these steps to set up and run the project.

### 1\. Prerequisites

  * [Python 3.9+](https://www.python.org/)
  * [Git](https://git-scm.com/)
  * [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 2\. Clone the Repository

```bash
git clone <your-repository-url>
cd Water-Paper-Simulator-main
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Start Local Services (Crucial Step)

Before running the main application, you must start two background services:

  * **Start Ollama (if using local models)**:
    Ensure your Ollama application is running and that the model you intend to use (e.g., `gpt-oss:20b`) has been pulled.

  * **Start GROBID (for PDF parsing)**:
    Open a **new terminal window** and run the following command to start the GROBID service. **Keep this terminal window open** for the entire duration of the project's run.

    ```bash
    docker run -t --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0
    ```

### 5\. Configure the Project

Open the `config.py` file and make the following **essential** configurations:

  * **`TOPIC`**: This is the most important setting. Change the `TOPIC` variable to a **specific and detailed** research topic you want to investigate.
    ```python
    # Example
    TOPIC = "Using reinforcement learning to optimize traffic light control systems in urban environments"
    ```
  * **`MODEL_PROVIDER`**: If you are using the OpenAI API, change this to `"openai"` and fill in your `OPENAI_API_KEY`.

### 6\. Run the System

Open **another new terminal window** and run the main script:

```bash
python main.py
```

The generated paper, logs, and intermediate files will be saved in the `output` directory.

-----

## License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).