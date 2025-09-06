# 🤖 AI驱动的自动化研究论文生成系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)

这是一个基于**多智能体协作**和**检索增强生成 (RAG)** 架构的自动化科研论文生成系统。它通过模拟学术研究的完整工作流，从主题构思、文献检索与分析，到草稿撰写和同行评审，最终生成一篇结构完整、内容详实的学术论文。

---

## 核心架构

本系统最大的特点是其**RAG驱动的多智能体工作流**。每个智能体（Agent）都有明确的职责，它们像一个高效的研究团队一样协同工作：

1.  **🧠 研究员 (ResearcherAgent)**
    * **职责**: 负责研究的初期探索。
    * **工作流**:
        * 根据初始主题构思创新性研究角度。
        * 利用 **Semantic Scholar API** 进行广泛的学术文献检索。
        * 自动下载开放获取的论文PDF，并使用 **GROBID** 将其解析为结构化文本。
        * 最终，将所有论文的全文内容构建成一个 **向量数据库 (Vector Store)**，为后续的RAG流程奠定基础。

2.  **📊 分析师 (AnalystAgent)**
    * **职责**: 评估研究方向并规划论文结构。
    * **工作流**:
        * 评估研究员提出的多个研究计划。
        * 结合文献摘要，选择一个最具潜力的最终主题。
        * 为最终主题创建一份详细、结构化的论文大纲。

3.  **✍️ 作家 (WriterAgent)**
    * **职责**: 基于RAG进行高质量的内容撰写。
    * **工作流**:
        * 在撰写每个章节（如“方法”、“文献综述”）时，它会向研究员构建的向量数据库提出问题。
        * 检索与当前章节最相关的**全文内容**作为上下文。
        * 基于这些精确、详细的上下文，综合地撰写出章节内容，确保了论文的深度和准确性。

4.  **🧐 审稿人 (ReviewerAgents)**
    * **职责**: 模拟同行评审，提供修订反馈。
    * **工作流**:
        * **广度审稿人**: 检查论文的宏观结构、逻辑连贯性和完整性。
        * **深度审稿人**: 审查技术细节、方法论的严谨性和数据的有效性。

5.  **🔁 决策与循环 (TriageRouter)**
    * **职责**: 控制工作流的迭代。
    * **工作流**: 根据审稿人的反馈，决定是将论文发回重写、重新研究，还是宣布流程结束。

---

## 如何使用 (Getting Started)

请按照以下步骤来设置和运行本项目。

### 1. 先决条件
* [Python 3.9+](https://www.python.org/)
* [Git](https://git-scm.com/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 2. 克隆仓库
```bash
git clone <your-repository-url>
cd Water-Paper-Simulator-main
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 启动本地依赖服务 (关键步骤)

在运行主程序之前，您必须先启动两个后台服务：

* **启动 Ollama (如果您使用本地模型)**:
    确保您的 Ollama 应用正在运行，并且您想要使用的模型（如 `gpt-oss:20b`）已经下载。

* **启动 GROBID (用于PDF解析)**:
    打开一个**新的终端窗口**，运行以下命令来启动 GROBID 服务。**在整个项目运行期间，请保持此终端窗口开启**。
    ```bash
    docker run -t --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0
    ```

### 5. 配置项目
打开 `config.py` 文件并进行以下**必要**配置：

* **`TOPIC`**: 这是最重要的部分。请将 `TOPIC` 变量修改为您想研究的**具体、详细的**主题。
    ```python
    # 示例
    TOPIC = "Using reinforcement learning to optimize traffic light control systems in urban environments"
    ```
* **`MODEL_PROVIDER`**: 如果您使用 OpenAI API，请将其改为 `"openai"` 并填写您的 `OPENAI_API_KEY`。

### 6. 运行系统
打开**另一个新的终端窗口**，运行主程序：
```bash
python main.py
```
生成的论文、日志和中间文件将被保存在 `output` 目录下。

---

## 许可证
本项目采用 [MIT License](LICENSE) 授权。