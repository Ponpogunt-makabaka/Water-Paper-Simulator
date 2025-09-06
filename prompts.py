# prompts.py
"""
Complete and fixed prompt templates for all agents.
Uses multiple-choice and yes/no formats to minimize token usage.
Enhanced with consistency management and quality control prompts.
FIXED: All template formatting issues and missing variables.
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
# BASIC WRITER PROMPTS
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
# ENHANCED WRITER PROMPTS WITH CONSISTENCY MANAGEMENT
# ==============================================================================

WRITER_ENHANCED_SECTION_PROMPT = """Write section: {section_title}

PAPER CONTEXT:
Topic: {topic}
Research Domain: {research_domain}
Main Contribution: {main_contribution}

SECTION REQUIREMENTS:
Plan: {section_plan}
Style: {style}
Focus: {focus_points}
Logical Role: {logical_role}

CONSISTENCY GUIDELINES:
{consistency_guidelines}

CONCEPT REQUIREMENTS:
{concept_context}

Context from literature:
--- LITERATURE CONTEXT ---
{context}
--- END CONTEXT ---

REQUIREMENTS:
- Length: {max_length} chars max
- Citations: Use [Author, Year] format consistently
- Terminology: Use consistent terms (avoid synonyms for key concepts)
- Flow: Ensure logical progression with clear transitions
- Concepts: Define new concepts before using them
- Dependencies: Ensure prerequisite concepts are established
- Coherence: Maintain narrative thread throughout

Content:"""

# ==============================================================================
# CONSISTENCY MANAGEMENT PROMPTS
# ==============================================================================

CONSISTENCY_CHECK_PROMPT = """Check this section for consistency issues:

Section: {section_title}
Expected Plan: {section_plan}

Content to check:
{content}

Consistency Criteria:
1. Outline Alignment: Does content match the planned topics?
2. Concept Dependencies: Are new concepts properly introduced?
3. Terminology: Are terms used consistently?
4. Logical Flow: Are ideas presented in logical order?
5. Narrative Coherence: Does it fit the overall paper narrative?

Rate each criterion (0.0-1.0) and identify specific issues:

SCORES: [outline_score, concept_score, terminology_score, flow_score, coherence_score]
ISSUES: [List specific problems found]
SUGGESTIONS: [List improvements needed]

Response:"""

CONCEPT_DEPENDENCY_PROMPT = """Analyze concept dependencies in this text:

Text: {text}

For each concept mentioned, identify:
1. Is it properly defined before use?
2. Are its dependencies satisfied?
3. Is terminology consistent?

Output format:
CONCEPTS_USED: [list of concepts found]
DEPENDENCY_ISSUES: [concepts used without proper introduction]
TERMINOLOGY_ISSUES: [inconsistent usage of terms]
SUGGESTIONS: [how to fix issues]

Analysis:"""

LOGICAL_FLOW_ENHANCEMENT_PROMPT = """Enhance the logical flow of this text by adding appropriate transitions:

Section: {section_title}
Previous Section: {previous_section}
Next Section: {next_section}

Text to enhance:
{content}

Requirements:
- Add smooth transitions between paragraphs
- Connect to previous section context
- Prepare transition to next section
- Use appropriate connecting phrases
- Maintain academic tone

Enhanced text:"""

TERMINOLOGY_NORMALIZATION_PROMPT = """Normalize terminology in this academic text:

Text: {text}

Requirements:
- Use consistent terms for the same concepts
- Standardize technical terminology
- Ensure acronyms are properly introduced
- Maintain academic precision
- Avoid unnecessary synonyms

Key terms to standardize: {key_terms}

Normalized text:"""

# ==============================================================================
# CONSISTENCY ANALYSIS PROMPTS
# ==============================================================================

OUTLINE_CONSISTENCY_PROMPT = """Check if this content aligns with its planned outline:

Planned Outline:
{outline}

Actual Content:
{content}

Evaluation:
1. Coverage: Does content cover planned topics? [0.0-1.0]
2. Focus: Does content stay focused on outlined scope? [0.0-1.0]
3. Structure: Does content follow planned structure? [0.0-1.0]

COVERAGE_SCORE: [score]
FOCUS_SCORE: [score]
STRUCTURE_SCORE: [score]
MISSING_TOPICS: [topics from outline not covered]
EXTRA_TOPICS: [topics in content not in outline]
SUGGESTIONS: [how to improve alignment]

Assessment:"""

CONCEPT_INTRODUCTION_PROMPT = """Analyze how concepts are introduced in this text:

Text: {text}
Section: {section_title}

For each concept, determine:
1. Is it defined clearly?
2. Are examples provided?
3. Are dependencies established?
4. Is the introduction appropriate for this section?

CONCEPTS_INTRODUCED: [list with quality ratings]
WELL_DEFINED: [concepts with clear definitions]
NEEDS_CLARIFICATION: [concepts needing better explanation]
DEPENDENCY_GAPS: [missing prerequisite concepts]

Analysis:"""

NARRATIVE_COHERENCE_PROMPT = """Check narrative coherence across sections:

Paper Topic: {topic}
Main Contribution: {contribution}

Section Contents:
{section_contents}

Coherence Check:
1. Consistent narrative thread? [YES/NO]
2. Logical section progression? [YES/NO]
3. Unified terminology? [YES/NO]
4. Clear contribution storyline? [YES/NO]

COHERENCE_SCORE: [0.0-1.0]
NARRATIVE_ISSUES: [list of problems]
FLOW_PROBLEMS: [section transition issues]
RECOMMENDATIONS: [specific improvements]

Assessment:"""

# ==============================================================================
# ENHANCED REVISION PROMPTS
# ==============================================================================

ENHANCED_REVISION_PROMPT = """Revise this academic paper draft with enhanced consistency management:

FEEDBACK TO ADDRESS:
{feedback_points}

CONSISTENCY REQUIREMENTS:
- Maintain consistent terminology throughout
- Preserve logical flow and transitions  
- Ensure concept dependencies are satisfied
- Keep coherent narrative thread

REVISION GUIDELINES:
{revision_guidelines}

LITERATURE CONTEXT:
--- CONTEXT ---
{context}
--- END CONTEXT ---

ORIGINAL DRAFT:
{draft}

Provide a revised version that addresses the feedback while maintaining consistency:"""

INCREMENTAL_IMPROVEMENT_PROMPT = """Apply incremental improvements to this draft:

Target Issue: {issue_type}
Specific Feedback: {feedback_detail}

Current Text:
{text_section}

Improvement Strategy:
{improvement_strategy}

Apply the improvement while maintaining:
- Consistent terminology
- Logical flow
- Academic tone
- Factual accuracy

Improved text:"""

CITATION_IMPROVEMENT_PROMPT = """Improve the citation patterns in this academic text:

Text: {text}

Requirements:
- Add citations where claims are made
- Use consistent [Author, Year] format
- Ensure appropriate citation density
- Cite recent and relevant sources
- Avoid over-citation or under-citation

Guidelines:
- Factual claims need citations
- Novel contributions should reference prior work
- Methodological choices should be justified
- Results should reference supporting literature

Improved text with enhanced citations:"""

CLARITY_ENHANCEMENT_PROMPT = """Enhance the clarity of this academic text:

Text: {text}

Clarity Issues to Address:
{clarity_issues}

Enhancement Strategies:
- Break down complex sentences
- Add explanatory phrases
- Improve paragraph structure
- Clarify technical concepts
- Enhance logical connections

Maintain:
- Technical accuracy
- Academic tone
- Consistent terminology
- Logical flow

Enhanced text:"""

# ==============================================================================
# REVIEWER PROMPTS - SIMPLIFIED FOR RELIABILITY
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

Respond in this format:
SCORES: [score1, score2, score3, score4, score5]
ISSUES: [List specific problems found]
SUGGESTIONS: [List improvements needed]
STRENGTHS: [List positive aspects]

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

Respond in this format:
SCORES: [score1, score2, score3, score4, score5]
ISSUES: [List specific technical problems]
SUGGESTIONS: [List technical improvements needed]
STRENGTHS: [List technical strengths]

Each score should be between 0.0 and 1.0. Focus on technical accuracy and methodological soundness."""

# ==============================================================================
# TRIAGE PROMPTS - ENHANCED WITH CONSISTENCY METRICS
# ==============================================================================

TRIAGE_DECISION_PROMPT = """Analyze feedback severity with consistency metrics.

Breadth: {feedback_breadth}
Depth: {feedback_depth}
Breadth Score: {score_breadth}
Depth Score: {score_depth}
Consistency Score: {consistency_score}
Outline Alignment: {outline_alignment}

Classify required action:
[1] RERESEARCH - Major flaws, need new literature (combined score < 0.4)
[2] REWRITE - Minor issues, improve existing text (combined score 0.4-0.7)
[3] END - No significant issues (combined score > 0.7)

Consider both review scores and consistency metrics.
Combined Score = (Review Score * 0.7) + (Consistency Score * 0.3)

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

# ==============================================================================
# QUALITY ASSURANCE PROMPTS
# ==============================================================================

QUALITY_CHECK_PROMPT = """Perform comprehensive quality check on this academic text:

Text: {text}

Check for:
1. Factual consistency
2. Logical flow
3. Citation appropriateness
4. Terminology consistency
5. Concept dependency satisfaction
6. Narrative coherence
7. Technical accuracy

Rate each aspect (0.0-1.0) and provide specific feedback:

QUALITY_SCORES: [fact, flow, citation, terminology, concepts, narrative, technical]
CRITICAL_ISSUES: [Issues that must be fixed]
MINOR_ISSUES: [Issues that should be improved]
STRENGTHS: [What works well]
OVERALL_ASSESSMENT: [Summary evaluation]

Quality assessment:"""

FINAL_POLISH_PROMPT = """Apply final polish to this academic paper:

Draft: {draft}

Polish areas:
- Language fluency and academic tone
- Transition smoothness
- Citation formatting
- Terminology consistency
- Paragraph structure
- Overall coherence

Requirements:
- Maintain all technical content
- Preserve original meaning
- Enhance readability
- Ensure professional presentation

Polished version:"""

# ==============================================================================
# SIMPLE FALLBACK PROMPTS FOR TESTING
# ==============================================================================

SIMPLE_SECTION_PROMPT = """Write a {section_title} section about {topic}.

Keep it academic and informative.
Length: approximately {max_length} characters.

Content:"""

SIMPLE_REVIEW_PROMPT = """Review this academic text and provide a score from 0.0 to 1.0:

Text: {draft}

Score: [your_score]
Comments: [brief feedback]"""

