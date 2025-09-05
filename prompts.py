# prompts.py
"""
Optimized prompt templates for all agents.
Uses multiple-choice and yes/no formats to minimize token usage.
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

Output JSON:
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
# WRITER PROMPTS
# ==============================================================================

WRITER_SECTION_PROMPT = """Write section: {section_title}

Topic: {topic}
Plan: {section_plan}
Style: {style}

Requirements:
- Length: {max_length} chars max
- Citations: Use [Author, Year] format
- Focus: {focus_points}

Content:"""

WRITER_REVISION_PROMPT = """Revise draft.

Issues to fix:
{feedback_points}

Changes required:
[1] Minor edits only
[2] Restructure paragraphs
[3] Add technical details
[4] Rewrite section

Apply changes to:
{draft}

Output revised text:"""

# ==============================================================================
# REVIEWER PROMPTS
# ==============================================================================

BREADTH_REVIEW_PROMPT = """Review macro aspects.

Check:
1. Core argument clear? [YES/NO]
2. Literature complete? [YES/NO]
3. Structure logical? [YES/NO]

Draft: {draft}

If any NO, list issues (<{max_feedback} chars).
Output: [PASS] or [ISSUES: list]"""

DEPTH_REVIEW_PROMPT = """Review technical depth.

Check:
1. Methods rigorous? [YES/NO]
2. Math correct? [YES/NO]
3. Evaluation valid? [YES/NO]

Draft: {draft}

If any NO, list issues (<{max_feedback} chars).
Output: [PASS] or [ISSUES: list]"""

# ==============================================================================
# TRIAGE PROMPTS
# ==============================================================================

TRIAGE_DECISION_PROMPT = """Analyze feedback severity.

Breadth: {feedback_breadth}
Depth: {feedback_depth}

Classify required action:
[1] RERESEARCH - Major flaws, need new literature
[2] REWRITE - Minor issues, improve existing text
[3] END - No significant issues

Decision: [#]"""

# ==============================================================================
# UTILITY PROMPTS
# ==============================================================================

FORMAT_REFERENCES_PROMPT = """Format bibliography.

Raw data: {raw_text}

For each entry include:
[1] Title only
[2] Title + Authors
[3] Full citation

Format choice: [2]
Output numbered list:"""

EXTRACT_KEYWORDS_PROMPT = """Extract keywords from: {text}

Type:
[1] Technical terms
[2] Concepts
[3] Applications

Select [1-3]: [1]
Output: keyword1, keyword2, keyword3"""