# prompts.py
"""
Optimized prompt templates for all agents.
Uses multiple-choice and yes/no formats to minimize token usage.
Fixed JSON formatting issues by properly escaping braces.
"""

# ==============================================================================
# RESEARCHER PROMPTS
# ==============================================================================

RESEARCH_INNOVATION_PROMPT = """Generate 3 research angles for: {topic}

Select ONE approach per category:
A) Object innovation: [1] New architecture [2] Novel dataset [3] Hybrid system
B) Framework innovation: [1] Theory extension [2] New paradigm [3] Cross-domain
C) Method innovation: [1] Algorithm improvement [2] Optimization [3] Integration

Format (max 1000 chars each):
Angle1: [Choice A#]-[Title]
Angle2: [Choice B#]-[Title]  
Angle3: [Choice C#]-[Title]"""

RESEARCH_KEYWORDS_PROMPT = """Extract search terms from:
{plans}

Choose search type for each angle:
[1] Technical terms only
[2] Application-focused
[3] Theory+implementation

Output JSON format:
{{"queries": ["term1", "term2", "term3"]}}"""

# ==============================================================================
# ANALYST PROMPTS
# ==============================================================================

ANALYST_EVALUATE_PROMPT = """Evaluate 3 research proposals.

Plans:
{plans}

References:
{references}

Rate each (1-5):
- Feasibility: [1]Poor [2]Fair [3]Good [4]Very Good [5]Excellent
- Novelty: [1]Poor [2]Fair [3]Good [4]Very Good [5]Excellent
- Support: [1]Poor [2]Fair [3]Good [4]Very Good [5]Excellent

Select best: [1] [2] or [3]
Output: Choice=[#] Topic=[specific topic <20 words]"""

ANALYST_FOUNDATION_CHECK = """Check theoretical foundation.

Topic: {topic}
References: {references}

Questions:
Q1. Core concepts covered? [YES/NO]
Q2. Methodology supported? [YES/NO]
Q3. Sufficient evidence? [YES/NO]

If any NO, provide search query (<30 words).
Output: [SOLID] or [NEEDS: query]"""

# ==============================================================================
# WRITER PROMPTS - FIXED RAG INTEGRATION
# ==============================================================================

WRITER_SECTION_PROMPT = """Write section: {section_title}

Topic: {topic}
Plan: {section_plan}
Style: {style}

Context from literature:
--- LITERATURE CONTEXT ---
{context}
--- END CONTEXT ---

Requirements:
- Length: {max_length} chars max
- Citations: Use [Author, Year] format
- Focus: {focus_points}
- Use the provided literature context to ensure accuracy and depth
- Ground your writing in the research evidence provided
- Synthesize information from multiple sources when possible

Content:"""

WRITER_REVISION_PROMPT = """Revise draft using literature context.

Issues to fix:
{feedback_points}

Literature context for revision:
--- CONTEXT ---
{context}
--- END CONTEXT ---

Changes required:
[1] Minor edits only
[2] Restructure paragraphs  
[3] Add technical details from literature
[4] Rewrite section with better grounding

Apply changes to:
{draft}

Output revised text:"""

# ==============================================================================
# REVIEWER PROMPTS - FIXED JSON FORMATTING
# ==============================================================================

BREADTH_REVIEW_PROMPT = """Review macro aspects of this paper.

Check these criteria (score each 0.0-1.0):
1. Core argument clarity
2. Literature completeness
3. Logical structure
4. Contribution evidence
5. Writing quality

Draft to review:
{draft}

Respond in this exact JSON format (replace values but keep structure):
{{
    "scores": [0.8, 0.7, 0.9, 0.8, 0.7],
    "issues": [
        {{"category": "Structure", "description": "The introduction lacks clear thesis statement"}}
    ],
    "suggestions": ["Improve the abstract clarity", "Add more recent references"],
    "strengths": ["Good methodology section", "Clear experimental design"]
}}

Each score should be between 0.0 and 1.0. If any score is below 0.7, explain the issue."""

DEPTH_REVIEW_PROMPT = """Review technical depth and rigor.

Check these criteria (score each 0.0-1.0):
1. Methodological rigor
2. Mathematical correctness
3. Experimental design soundness
4. Technical innovation level
5. Implementation detail sufficiency

Technical sections to review:
{draft}

Respond in this exact JSON format (replace values but keep structure):
{{
    "scores": [0.8, 0.7, 0.9, 0.8, 0.7],
    "issues": [
        {{"category": "Methodology", "description": "The experimental setup lacks proper controls"}}
    ],
    "suggestions": ["Add more detailed algorithm explanation", "Include computational complexity analysis"],
    "strengths": ["Rigorous mathematical formulation", "Comprehensive evaluation metrics"]
}}

Each score should be between 0.0 and 1.0. Focus on technical accuracy and methodological soundness."""

# ==============================================================================
# TRIAGE PROMPTS
# ==============================================================================

TRIAGE_DECISION_PROMPT = """Analyze feedback severity.

Breadth: {feedback_breadth}
Depth: {feedback_depth}
Breadth Score: {score_breadth}
Depth Score: {score_depth}

Classify required action:
[1] RERESEARCH - Major flaws, need new literature (avg score < 0.4)
[2] REWRITE - Minor issues, improve existing text (avg score 0.4-0.7)
[3] END - No significant issues (avg score > 0.7)

Decision: [#]"""

# ==============================================================================
# RAG-ENHANCED UTILITY PROMPTS
# ==============================================================================

FORMAT_REFERENCES_PROMPT = """Format bibliography from literature data.

Raw data: {raw_text}

For each entry include:
[1] Title only
[2] Title + Authors
[3] Full citation with year

Format choice: [2]
Output numbered list:"""

EXTRACT_KEYWORDS_PROMPT = """Extract keywords from research content.

Content: {text}

Extract type:
[1] Technical terms
[2] Research concepts  
[3] Application domains

Select [1-3]: [1]
Output: keyword1, keyword2, keyword3"""

RAG_QUERY_GENERATION_PROMPT = """Generate effective search queries for RAG retrieval.

Research topic: {topic}
Section being written: {section}
Current focus: {focus}

Generate 3 specific queries to find relevant literature:
1. Technical query: [specific technical terms]
2. Methodological query: [methods and approaches] 
3. Application query: [use cases and applications]

Output JSON format:
{{"queries": ["query1", "query2", "query3"]}}"""

# ==============================================================================
# ERROR HANDLING AND FALLBACK PROMPTS
# ==============================================================================

FALLBACK_CONTENT_PROMPT = """Generate academic content when literature context is limited.

Topic: {topic}
Section: {section_title}
Requirements: {requirements}

Without extensive literature context, write a foundational section that:
- Establishes key concepts
- Outlines standard approaches
- Identifies research gaps
- Suggests methodological directions

Content:"""

CONTEXT_SYNTHESIS_PROMPT = """Synthesize information from multiple literature sources.

Research question: {question}

Source contexts:
{contexts}

Synthesize into coherent narrative that:
- Identifies common themes
- Highlights disagreements
- Notes methodological differences
- Suggests areas for further investigation

Synthesis:"""