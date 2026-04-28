from typing import Final

from langchain.prompts import PromptTemplate


CONCEPT_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["resource_text", "source_name", "source_type"],
    template="""
# Instruction

You are an expert linguistic AI assistant specializing in ontology engineering for low-resource languages from linguistic resources.
Your task is to analyze the provided segment of a linguistic resource and extract:
1. **Candidate Classes (Linguistic Concepts/Entities)**: These are the fundamental entities used in morphological and syntactic analysis (e.g., Sentence, Token, CandidateAnalysis, Lemma, Morpheme, FeatureBundle, PartOfSpeech, Case, Number, Person, Tense, Mood, DependencyRelation, AgreementPattern, GovernmentPattern, Construction, Postposition).
2. **Candidate Properties**: These define attributes of classes or relationships between classes (e.g., hasForm, hasLemma, hasCandidatePOS, hasCandidateFeature, candidateOf, belongsToSentence, agreesWith, governsCase, hasSuffix, followsToken).

The source may be:
- a grammar textbook,
- an annotation guideline,
- analyzer documentation,
- corpus notes,
- or expert linguistic commentary.

# Important Modeling Principle

The same surface token may participate in multiple alternative analyses. This causes ambiguity in downstream systems if represented incorrectly.
Thus, prefer:
- a base class such as `Token`,
- and separate `CandidateAnalysis` entities linked to the token via a property such as `candidateOf`.

Example:
POOR:
- Token1Noun
- Token1Pronoun
with no connection between them

GOOD:
- Token1 is a Token
- Analysis1 is a CandidateAnalysis linked to Token1
- Analysis2 is a CandidateAnalysis linked to Token1
- properties encode the alternatives

Also, prefer properties over duplicating overlapping classes when the same entity may bear multiple roles.

For each extracted item, provide a confidence score (0.0 to 1.0), a brief explanation, and explicit provenance:
- source_name,
- source_type,
- evidence from the input.

# Important Context for Knowledge Application

The ontology you create (TBox) will be used in two critical ways:
1. To guide the extraction of sentence-specific facts (ABox) from user queries, raw sentences, UniParser candidate analyses, corpus examples, and optional manual annotations.
2. To structure Python code that will build a functional knowledge graph, apply constraints, and resolve morphological and syntactic ambiguity.

Therefore, your extracted concepts should:
- Be atomic/granular rather than composite
- Avoid concepts that already encode final disambiguation decisions
- Focus on raw observable data points
- Enable later rule-based reasoning
- Support agreement, government, compatibility, and context-sensitive filtering

# Example Use Cases

Your ontology should support answering queries like:

Example 1:
Description: Sentence: Кин ке но пилиське...
The token "пилиське" has two candidate analyses: STEM-PASS-IMP.PL and STEM-PASS-PRS.3SG.
Query: Which analysis should be selected for the token "пилиське"?
Expected answer: STEM-PASS-PRS.3SG

Example 2:
Description: Sentence: Атай, сое.
The token "сое" is annotated as a case-marked form with accusative morphology.
Query: What case should be assigned to the token "сое"?
Expected answer: Case=ACC

Example 3:
Description: Sentence: куддыр потэ ноку уз быр.
The token "потэ" has two candidate analyses: STEM-PRS.3SG and STEM-IMP.PL.
Query: Which analysis is compatible with a declarative finite-clause reading?
Expected answer: PRS.3SG

Example 4:
Description: Sentence: Мыным туж ӝож луэ, ку мон малпасько Шундыкар сярысь.
The token "малпасько" has three candidate analyses: STEM-PASS-PRS.3PL, STEM-PASS-FUT, and STEM-PRS.12.
Query: Which candidate analyses should remain after applying agreement and clause-context constraints?
Expected answer: STEM-PRS.12

# Extraction Guidelines

When extracting classes and properties:

1. For **Classes**:
   - Focus on fundamental linguistic entities
   - Prefer reusable concepts over corpus-example-specific names
   - Think about what entities must appear in the ABox for sentence analysis

2. For **Properties**:
   - Extract relations between entities (e.g., candidateOf, belongsToSentence, agreesWith)
   - Extract intrinsic attributes (e.g., hasForm, hasLemma, hasCase, hasNumber)
   - For linguistic values, prefer raw features over bundled interpretations
   - Include properties that will be needed by Python code to build a functional knowledge graph

3. Do NOT create properties that already encode final system decisions
   - AVOID: isBestAnalysis, resolvedHead, preferredFinalCase
   - PREFER: hasCandidateCase, hasCandidatePOS, hasDependencyCandidate, hasSuffix

4. Remember that Python code will later implement all disambiguation procedures based on your ontology, so focus on capturing the raw linguistic data points that such code would need to resolve morphological and syntactic ambiguity for cases like the examples above.

5. When the source explicitly provides complementary linguistic oppositions (e.g., finite vs non-finite, singular vs plural, animate vs inanimate), preserve them when useful. Do NOT invent artificial negations that are not supported by the source.

6. Every extracted item must be traceable to the provided source segment. Do not invent unsupported concepts.

# Input Resource Metadata

Source name: {{ source_name }}
Source type: {{ source_type }}

# Input Resource Text

{{ resource_text }}

# Output Format

Return a single, valid JSON object that strictly adheres to the following structure. Do NOT include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{
  "classes": [
    {
      "object": {
        "name": "ClassNameInPascalCase",
        "description": "Concise description of the class."
      },
      "confidence": 0.9,
      "source": {
        "source_name": "{{ source_name }}",
        "source_type": "{{ source_type }}",
        "evidence": "Short supporting fragment or paraphrase from the resource."
      },
      "explanation": "Reason for extracting this class and determining its fields."
    }
  ],
  "properties": [
    {
      "object": {
        "type": "unary_or_object_or_datatype",
        "name": "propertyNameInCamelOrPascalCase",
        "arguments": ["Arg1", "Arg2_if_applicable"],
        "description": "Concise description of the property."
      },
      "confidence": 0.85,
      "source": {
        "source_name": "{{ source_name }}",
        "source_type": "{{ source_type }}",
        "evidence": "Short supporting fragment or paraphrase from the resource."
      },
      "explanation": "Reason for extracting this property, its type, arguments, and other fields."
    }
  ]
}
```

Detailed Field Explanations for the "object" within "properties":

* **type**: Must be one of "unary", "object", or "datatype".
    - "unary": Represents a characteristic of a single class (e.g., IsNoun). arguments will be
      a list with one string: the class name it applies to (e.g., ["Noun"]).
    - "object": Represents a relationship between two classes, arguments will be a list of two strings.
    - "datatype": Represents an attribute of a class with a literal value, arguments will be a list of two string. Literal types can include string, integer, decimal, boolean and date.

* **name**: The name of the property. Use camelCase or PascalCase (e.g., isNoun, HasGender).

* **arguments**: A list of strings as described above, depending on the property type. These should
  refer to Class names defined in classes or common literal types.
    - unary -> ["ClassName"]
    - object -> ["DomainClassName", "RangeClassName"]
    - datatype -> ["DomainClassName", "LiteralType"]

* **description**: A concise explanation of the property.

Ensure all class and property names are descriptive and consistently cased.
Focus on extracting core, reusable concepts and properties directly supported by the provided text.
If no relevant classes or properties are found in a segment, an empty list for that category is acceptable.
The overall JSON must be valid.
""",
)

RULE_FORMULATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["resource_text", "source_name", "source_type"],
    template="""
# Instruction

You are an expert linguistic AI assistant specializing in extracting logical rules for ontology engineering from linguistic resources for low-resource languages.
Your task is to analyze the provided segment of a linguistic resource and extract candidate logical rules that can support morphological and syntactic disambiguation.

Each rule defines a new property (the implication) based on a set of linguistic conditions (the FOL expression/premises). The rule should describe a reusable linguistic regularity, not a one-off annotation decision.

# Important Direction
- Focus on extracting linguistic relationships that can help resolve ambiguity: agreement, government, compatibility of features, clause context, syntactic position, dependency relations, finiteness, polarity, tense/mood compatibility, and postpositional/case patterns
- Do NOT create rules that encode final labels directly without conditions
- Do NOT create single-predicate rules
- Do NOT invent rules unsupported by the source segment
- Prefer rules that can later be converted into SWRL-like rules or checked as ontology constraints.
- If the source only provides examples and no general rule can be inferred confidently, return an empty list.

# Input Resource Metadata
```
Source name: {source_name}
Source type: {source_type}
```
# Input Resource Text
```
{resource_text}
```

# Task

Extract logical rules that define reusable linguistic relationships. For each rule:

- `fol_expression`: Write the First-Order Logic (FOL) expression representing the **premises (conditions) only** only.
  Use predicates that correspond to explicit properties likely to exist in the ontology.
  Examples:
  - "CandidateOf(A, T) & HasCandidatePOS(A, Verb) & HasCandidateTense(A, Present) & InDeclarativeClause(T, C)"
  - "CandidateOf(A, T) & HasCandidateCase(A, Accusative) & GovernedBy(T, V) & GovernsCase(V, Accusative)"
  - "CandidateOf(A, T) & HasCandidateNumber(A, Singular) & AgreesWith(T, H) & HasNumber(H, Singular)"
- `implication_property_name`: Specify the name of the **NEW property** that is true if the `fol_expression` is true. This property is introduced by this rule.
  Examples:
  - "ContextuallyCompatibleAnalysis"
  - "GovernedCaseCompatibleAnalysis"
  - "AgreementCompatibleAnalysis"
- `implication_property_type`: Specify the type of the **NEW property** - one of 'unary' / 'object' / 'datatype'.
- `implication_property_arguments`: Provide a list of plausible Class names that are the arguments for the `implication_property_name`. For most disambiguation rules, a unary property over CandidateAnalysis is sufficient: ["CandidateAnalysis"].
- `description`: Explain the linguistic motivation of the rule.
- Provide a `confidence` score (0.0 to 1.0) and an `explanation` for the entire rule extraction, including why you chose certain predicate or class names and the evidence from the source segment.

# Examples of What to Extract and What to Avoid

GOOD EXAMPLES (focus on conceptual relationships):
- "CandidateOf(A, T) & HasCandidatePOS(A, Verb) & InFiniteClause(T, C) & HasCandidateMood(A, Indicative)"
  -> "FiniteClauseCompatibleAnalysis(A)"
- "CandidateOf(A, T) & HasCandidateCase(A, Accusative) & GovernedBy(T, H) & GovernsCase(H, Accusative)"
  -> "GovernmentCompatibleAnalysis(A)"

BAD EXAMPLES (numeric thresholds that should be handled by code):
- "HasCandidatePOS(A, Noun)" -> "IsBestAnalysis(A)"
  Reason: single-predicate and directly encodes a final decision.
- "Token(T)" -> "ResolvedToken(T)"
  Reason: no linguistic condition.

Instead, simply extract rules about what entities ARE and their relationships. The Python code will handle all numeric comparisons and threshold checks.

# Output Format

Return a single, valid JSON object that strictly adheres to the following structure. Do NOT include markdown formatting (e.g., ```json) around the JSON object itself.

```json
{{
  "rules": [
    {{
      "object": {{
        "fol_expression": "CandidateOf(A, T) & HasCandidateCase(A, Accusative) & GovernedBy(T, H) & GovernsCase(H, Accusative)",
        "implication_property_name": "GovernmentCompatibleAnalysis",
        "implication_property_type": "unary",
        "implication_property_arguments": ["CandidateAnalysis"],
        "description": "A candidate analysis is compatible if its case feature matches the case governed by its syntactic head."
      }},
      "confidence": 0.85,
      "source": {{
        "source_name": "{source_name}",
        "source_type": "{source_type}",
        "evidence": "Short supporting fragment or paraphrase from the resource."
      }},
      "explanation": "The rule is derived from the source statement that a governor imposes a case requirement on its dependent."
    }}
  ]
}}
```

If no rules can be confidently extracted, return an empty list for "rules".
The overall JSON must be valid.
""",
)

RULE_INTEGRATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "candidate_classes",
        "candidate_properties",
        "candidate_rules",
        "human_feedback",
        "last_tbox",
        "tbox_validation_issues",
    ],
    template_format="jinja2",
    template="""
# Instruction

You are an expert ontology engineer AI specializing in linguistic ontologies for low-resource languages. Your task is to integrate candidate classes, properties, and rules into a single, coherent, and consistent Terminological Box (TBox) for morphological and syntactic disambiguation.
The candidate elements were extracted by separate, parallel processes and may contain redundancies, inconsistencies, or require refinement.

# Inputs

## 1. Candidate Classes:
Each class includes the extracted object, confidence score, and an explanation.
```json
{{ candidate_classes }}
```

## 2. Candidate Properties (Explicitly Extracted):
These are properties identified directly from the text. Each includes the extracted object, confidence score, and an explanation.
```json
{{ candidate_properties }}
```

## 3. Candidate Rules:
Each rule includes the extracted object (premises, implied property name & args, description), confidence, and explanation.
The `implication_property_name` in a rule introduces a new property that is defined *by that rule*.
```json
{{ candidate_rules }}
```

# Task: Construct the Final TBox

Create a final TBox with three main lists: `classes`, `properties` (explicit only), and `rules`.

You are allowed to modify input classes, properties and rules if it's necessary. The inputs are only
suggestions. In particular, in case of a subsequent iteration, it's more important to incorporate feedback
than to preserve the structure of the input.

## Steps & Considerations:

### A. Finalize Classes:
1.  **Review `candidate_classes`**: Consider their descriptions, confidence, and explanations.
2.  **Consolidate & Deduplicate**: Merge classes that represent the same concept. Standardize names to PascalCase.
3.  **Filter**: You may discard very low-confidence or irrelevant classes.
4.  **Output**: A final list of `TBoxClass` objects.

### B. Finalize Explicit Properties:
1.  **Review `candidate_properties`**: These are the properties extracted directly, not those implied by rules. Consider their type, arguments, descriptions, confidence, and explanations.
2.  **Consolidate & Deduplicate**: Merge `candidate_properties` that represent the same concept. Standardize names (camelCase or PascalCase).
3.  **Validate Arguments**: Ensure all class names used in property `arguments` for these explicit properties exist in your finalized list of classes (from Step A). Adjust if necessary.
4.  **Filter**: Discard low-confidence or redundant explicit properties.
5.  **Output**: A final list of `TBoxProperty` objects. This list should *ONLY* contain these explicitly defined and refined properties. Properties introduced by rule implications (e.g., `implication_property_name` from a rule) are defined by the rules themselves and are *NOT* to be added to this explicit `properties` list.

### C. Finalize Rules:
1.  **Review `candidate_rules`**: Consider their FOL expressions, implied properties, descriptions, confidence, and explanations.
2.  **Validate Predicates in `fol_expression`**:
    -   The predicates used in a rule's `fol_expression` should correspond to property names in your finalized list of explicit properties (from Step B).
    -   If a predicate in a rule's premises does not match an existing finalized explicit property, try to map it to one. If it's essential and clearly implied by the text as a premise-predicate, you might need to add it to the explicit properties list in Step B (and document this decision in the summary). Avoid inventing premise-predicates that aren't supported.
3.  **Validate `implication_property_name`, `implication_property_type` and `implication_property_arguments`**:
    -   For each rule, its `implication_property_name` defines a *new* predicate. Ensure this name is well-formed (e.g., PascalCase) and does *not* conflict with names in the finalized list of explicit properties (from Step B). It is defined *by* the rule.
    -   Ensure the class names used in the rule's `implication_property_arguments` are consistent with your finalized list of classes (from Step A).
    -   Ensure `implication_property_type` is one of 'unary' / 'object' / 'datatype',
    -   Ensure that when type is unary, there's exactly one argument referring to a valid class name,
    -   Ensure that when type is object, there are exactly two arguments, both referring to valid class names,
    -   Ensure that then type is datatype, there are exactly two arguments, first referring to a class, second is a datatype literal (one of 'string' / 'integer' / 'decimal' / 'boolean')
4.  **Filter**: Discard low-confidence, inconsistent, or redundant rules. A rule is inconsistent if its premise predicates cannot be mapped to finalized explicit properties or its implication arguments don't map to finalized classes.
5.  **Output**: A final list of `TBoxRule` objects. Each rule implicitly defines the property named in its `implication_property_name`.

{% if last_tbox %}
# Previous Iteration

CRITICAL! It is more important to incorporate human feedback than preserve the input structure of classes and properties.

Current ask is a subsequent iteration on the same problem. Last attempt:
{{ last_tbox }}

{% if human_feedback %}
## Human feedback for that iteration
{{ human_feedback }}
{% endif %}

{% if tbox_validation_issues %}
## TBox validation issues found
{{ tbox_validation_issues }}
{% endif %}
{% endif %}

# Output Format

Return a single, valid JSON object. Do NOT include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{
  "tbox": {
    "classes": [
        {
        "name": "FinalClassName",
        "description": "Final description."
        }
        // ... more classes
    ],
    "properties": [ // Explicitly defined properties ONLY
        {
        "type": "unary",
        "name": "finalExplicitPropertyName",
        "arguments": ["FinalClassName"],
        "description": "Final description from candidate property."
        }
        // ... more explicit properties
    ],
    "rules": [ // Rules define their own implied properties
        {
        "fol_expression": "FinalExplicitPredicateName(X) And AnotherFinalExplicitPredicate(X, Y)",
        "implication_property_name": "NewlyDefinedByRulePropertyName",
        "implication_property_type": "unary",
        "implication_property_arguments": ["FinalClassName"],
        "description": "Final rule description, this rule defines 'NewlyDefinedByRulePropertyName'."
        }
        // ... more rules
    ],
  }
}
```
The output must be a valid JSON object.
""",
)

CODE_GENERATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "classes",
        "properties",
        "statute",
        "last_error",
        "last_interpreter",
    ],
    template_format="jinja2",
    template="""
# Instruction

You are an expert AI assistant specializing in translating a linguistic ontology into executable
Python code. Your task is to generate Python code that uses a defined ontology vocabulary to filter and rank
candidate analyses for tokens in a sentence.

# Ontology Vocabulary

## Classes
These represent the core legal concepts/entities in the language:
```json
{{ classes }}
```

## Properties

These properties represent relationships and attributes of the classes:
```json
{{ properties }}
```

# Rules

{{ rules }}

# Task: Generate Executable Python Code

Create a Python module with a specific function that:
1. Accepts a user query and ABox assertions as input (in a precisely defined format)
2. Interprets the query to identify the candidate analyses for each token.
3. Applies ontology-derived compatibility rules and constraints
4. Rejects logically incompatible candidates when possible.
5. Returns the selected or remaining candidate analyses with explanations

## Required Function Signature

Your code MUST implement exactly this function signature:

```python
def disambiguate(abox: dict, sentence_id: str) -> dict:
    \"\"\"
    Resolves candidate morphological and syntactic analyses for tokens in a sentence.

    Args:
        abox (dict): Dictionary with the following structure:
            {
                "individuals": {
                    "sent1": {"type": "Sentence"},
                    "tok1": {"type": "Token"},
                    "ana1": {"type": "CandidateAnalysis"},
                    ...
                },
                "assertions": [
                    {"predicate": "belongsToSentence", "args": ["tok1", "sent1"]},
                    {"predicate": "candidateOf", "args": ["ana1", "tok1"]},
                    {"predicate": "hasForm", "args": ["tok1", "потэ"]},
                    {"predicate": "hasLemma", "args": ["ana1", "потыны"]},
                    {"predicate": "hasCandidatePOS", "args": ["ana1", "Verb"]},
                    {"predicate": "hasCandidateTense", "args": ["ana1", "Present"]},
                    ...
                ]
            }

        sentence_id (str): ID of the sentence to analyze.

    Returns:
        dict: A dictionary with selected or remaining candidate analyses and explanations.
    \"\"\"

    <ABoxRequirements>
        ...
    </ABoxRequirements>
    ...
```

## Chain-of-Code Technique

You MUST use the chain-of-code approach where each logical step is:
1. Explained clearly in comments BEFORE the code implementation.
2. Implemented with descriptive variable names aligned with the linguistic ontology.
3. Organized in a logical sequence that follows the disambiguation pipeline:
   sentence validation -> token collection -> candidate collection -> feature extraction
   -> rule application -> candidate filtering -> final decision with explanation.

Example of chain-of-code style:

```python
# Step 1: Validate that the ABox has the required top-level structure
# This ensures that the disambiguation function receives individuals and assertions
# in the expected ontology-based format.
if not isinstance(abox, dict):
    raise ValueError("ABox must be a dictionary")

if "individuals" not in abox or "assertions" not in abox:
    raise ValueError("ABox must contain 'individuals' and 'assertions' fields")

individuals = abox["individuals"]
assertions = abox["assertions"]

# Step 2: Verify that the requested sentence exists in the ABox
# The sentence_id must refer to an individual of type Sentence.
if sentence_id not in individuals:
    raise ValueError(f"Sentence ID '{sentence_id}' not found in the ABox")

if individuals[sentence_id].get("type") != "Sentence":
    raise ValueError(f"Individual '{sentence_id}' is not typed as Sentence")

# Step 3: Collect all tokens belonging to the requested sentence
# Tokens are connected to the sentence by the belongsToSentence property.
sentence_tokens = []

for assertion in assertions:
    if (
        assertion.get("predicate") == "belongsToSentence"
        and len(assertion.get("args", [])) == 2
        and assertion["args"][1] == sentence_id
    ):
        token_id = assertion["args"][0]
        if token_id in individuals and individuals[token_id].get("type") == "Token":
            sentence_tokens.append(token_id)

# Step 4: Collect all candidate analyses for each token
# Candidate analyses preserve ambiguity: one token may have several possible analyses.
token_to_candidates = {token_id: [] for token_id in sentence_tokens}

for assertion in assertions:
    if (
        assertion.get("predicate") == "candidateOf"
        and len(assertion.get("args", [])) == 2
    ):
        candidate_id, token_id = assertion["args"]
        if token_id in token_to_candidates:
            if candidate_id in individuals and individuals[candidate_id].get("type") == "CandidateAnalysis":
                token_to_candidates[token_id].append(candidate_id)

# Step 5: Extract linguistic features for every candidate analysis
# This step converts raw ABox assertions into a convenient internal representation
# used by compatibility rules.
candidate_features = {}

for token_id, candidate_ids in token_to_candidates.items():
    for candidate_id in candidate_ids:
        candidate_features[candidate_id] = {
            "token": token_id,
            "lemma": None,
            "pos": None,
            "features": set(),
            "dependencies": [],
            "supporting_rules": [],
            "rejecting_constraints": []
        }

for assertion in assertions:
    predicate = assertion.get("predicate")
    args = assertion.get("args", [])

    if len(args) < 2:
        continue

    subject = args[0]
    value = args[1]

    if subject not in candidate_features:
        continue

    if predicate == "hasLemma":
        candidate_features[subject]["lemma"] = value

    elif predicate == "hasCandidatePOS":
        candidate_features[subject]["pos"] = value

    elif predicate in {
        "hasCandidateFeature",
        "hasCandidateCase",
        "hasCandidateNumber",
        "hasCandidatePerson",
        "hasCandidateTense",
        "hasCandidateMood"
    }:
        candidate_features[subject]["features"].add((predicate, value))

# Step 6: Apply ontology-derived compatibility rules conservatively
# A candidate is rejected only if an explicit rule or constraint is applicable
# and the candidate violates it. If no rule applies, the candidate remains unresolved.
for candidate_id, features in candidate_features.items():
    token_id = features["token"]

    # Example rule pattern:
    # If the ontology states that a finite declarative clause requires a finite verb,
    # then a candidate with non-finite morphology may be rejected in that context.
    # This is only an example; generated code should implement rules derived from the TBox.
    is_rejected = False

    if ("hasCandidatePOS", "Verb") in features["features"]:
        features["supporting_rules"].append(
            "Candidate is compatible with verbal-context constraints when applicable"
        )

    if is_rejected:
        features["rejecting_constraints"].append(
            "Candidate violates an ontology-derived compatibility constraint"
        )

# Step 7: Select or retain candidate analyses for each token
# If exactly one candidate remains after filtering, it is selected.
# If multiple candidates remain, the token is marked as unresolved.
# If all candidates are rejected, the original candidates are restored and the case is marked as contradictory.
result = {
    "sentence_id": sentence_id,
    "tokens": {}
}

for token_id, candidate_ids in token_to_candidates.items():
    remaining_candidates = []
    rejected_candidates = []

    for candidate_id in candidate_ids:
        if candidate_features[candidate_id]["rejecting_constraints"]:
            rejected_candidates.append(candidate_id)
        else:
            remaining_candidates.append(candidate_id)

    if len(remaining_candidates) == 1:
        status = "resolved"
        selected = remaining_candidates[0]

    elif len(remaining_candidates) > 1:
        status = "unresolved"
        selected = None

    else:
        status = "contradiction"
        selected = None
        remaining_candidates = candidate_ids

    result["tokens"][token_id] = {
        "status": status,
        "selected_candidate": selected,
        "remaining_candidates": remaining_candidates,
        "rejected_candidates": rejected_candidates,
        "explanations": {
            candidate_id: {
                "supporting_rules": candidate_features[candidate_id]["supporting_rules"],
                "rejecting_constraints": candidate_features[candidate_id]["rejecting_constraints"]
            }
            for candidate_id in candidate_ids
        }
    }

## Requirements for the Generated Code:

1. Query Processing:
- Extract the taxpayer ID and given year from the query tuple
- Verify that the taxpayer exists in the ABox individuals collection
- Use this information to filter ABox assertions to focus only on the relevant taxpayer
- Ensure calculations account for the specified time period
- Code must handle entities with multiple roles. For example:
    - If taxpayer_id refers to an individual who is also an employer, you must calculate BOTH personal income tax AND employer excise tax.
- Remember to traverse relationships to find all relevant properties for the entity

2. ABox Processing:
- Use the EXACT ABox format specified in the function signature
- Extract all relevant values needed for calculation
- Validate that required properties are present

3. Calculation Logic:
- Implement numerical calculations as specified in the statute
- Use the exact numerical values, thresholds, and rates from the statute
- Calculate all intermediate values with appropriate precision
- Return the final result as a float value

4. Documentation:
- Use chain-of-code comments that explain WHAT and WHY for each step
- Reference specific sections of the statute in comments
- Use variable names that align with the ontology terms

5. (CRITICAL!) ABox Documentation
- In the Python docstring of the main method, within the <ABoxRequirement></ABoxRequirement>
  tags, thoroughly document the EXACT expected structure of the ABox for successful calculation
- List ALL required properties that must exist in the ABox, and what they represent
- Explain the meaning of each critical property and how it affects the calculation
- Document EACH property's expected format, data type, and purpose
- Explain how missing properties should be handled (errors, defaults, etc.)
- For each type of calculation, document which properties are required
- For joint vs. individual filings, clearly document how property values should be structured
- If year-specific rules apply, document how the year parameter affects property interpretation
- Provide complete examples showing property combinations for different filing scenarios
- Document property relationships and dependencies: which properties must co-exist

+ ADD THESE CRITICAL MULTI-ROLE INSTRUCTIONS:
+ - Document how entities with multiple roles must be represented and connected in the ABox
+ - Do not invent linguistic facts that are not in the ABox
+ - Do not hard-code Udmurt-specific examples unless they are explicitly present in the resource text
+ - Prefer conservative filtering: if a rule is not applicable, do not reject a candidate
+ - If the ontology cannot decide, return unresolved candidates rather than forcing a choice
+ - Use the rules as compatibility checks, not as direct gold labels

6. Error Handling:
- Validate inputs and handle missing or invalid values
- Raise appropriate exceptions for invalid inputs with clear error messages
- Handle cases where the query doesn't match any individuals in the ABox

{% if last_error %}

# Previous Attempts

This is an attempt to fix the previous code generation. The code was evaluated against
training dataset and it failed, here's the test case and error:

{{ last_error }}

The exact code we evaluated on:

```python
{{ last_interpreter }}
```

Make sure the next iteration improves with regards to robustness and accuracy.

{% endif %}

# Output Format

Return ONLY the Python code without any explanations or markdown formatting. The code must:
1. Include the exact function signature def calculate(abox: dict, sent_id: str) -> float
2. Be complete and executable without additional dependencies
3. Follow the chain-of-code technique with clear comment explanations
4. Implement all numerical calculations from the statute
5. Return a single float value as the final result

Do not include any explanation before or after the code - just provide the complete, executable Python code.
Do not wrap the output with any markdown formatting. Plain python only.
""",
)


FACT_EXTRACTION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "classes",
        "properties",
        "description",
        "tbox_interpreter_docstring",
    ],
    template_format="jinja2",
    template="""
# Task: Linguistic Answer Generation

Based on the user query, available ABox individuals, and the disambiguation result, produce a concise answer.

# TBox Schema

## Classes
{{ classes }}

## Properties
{{ properties }}

# User's Situation Description
{{ description }}

# User's Question
The extracted ABox, will be used to answer the following question:

    {{ question }}

Make sure that all properties required by the target function to answer this question are present in the resulting ABox.

# CRITICAL INSTRUCTIONS - READ CAREFULLY

The calculation function has specific requirements for the ABox structure. These requirements are specified below.
You MUST ensure ALL required properties are extracted, even if they need to be reasonably inferred.

ABox Structure Requirements from the calculation function:
{{ tbox_interpreter_docstring }}

## Step-by-Step Extraction Process

1. First, identify all individuals mentioned in the situation description and assign them appropriate class types.
   * Look for all taxpayers, dependents, employers, etc.
   * Assign meaningful and consistent IDs to each individual

2. Parse the description to find EVERY property mentioned in the text:
   * All explicit property values stated in the text
   * Any properties that can be reasonably inferred
   * Use the class and property definitions to guide your extraction

3. CRITICAL CHECK: Review the ABoxRequirements section above and ensure you extract EVERY required property
   * Make sure all properties needed for calculation are present
   * Missing required properties will cause calculation failures

4. For each property, provide:
   * A confidence score (0.0 to 1.0)
   * A specific explanation of why you created this assertion
   * The source in the text that supports this assertion

5. For any numeric values mentioned (incomes, wages, tax thresholds, etc.):
   * Extract ALL numeric values as datatype property assertions
   * Connect each value to the appropriate individual through relevant properties
   * Ensure these are correctly linked to the appropriate entities

# ALWAYS EXTRACT THESE CRITICAL PROPERTIES

Regardless of the specific scenario, ALWAYS extract the following critical properties when relevant:
- Tax filing status (e.g., filing jointly, separately, as head of household, etc.)
- Income and financial information (e.g., hasAdjustedGrossIncomeAmount, hasTaxableIncomeAmount)
- Personal attributes (e.g., age, blindness, dependent status)
- Business attributes (e.g., employer status, wage payments)
- Relationships between individuals (e.g., spouse, dependent)
- Standard deduction information if mentioned

CRITICAL! The input may contain some temporal concepts, not possible to be represented by by the ABox.
Always lean towards the 'year' asked for in the User's Question. Double check if the output is not
contradictory, e.g., [isFoo(x), isNotFoo(x)], but when taken only the year from question, [isFoo(x)] is the only predicate.

# Output Format

Return a single, valid JSON object. Do NOT include any markdown formatting (e.g., ```json) around the JSON object itself.

```json
{
    "abox": {
        "individuals": [
            {
                "object": {
                    "name": "individualname",
                    "class_name": "ClassNameFromTBox"
                },
                "confidence": 0.8,
                "explanation": "brief explanation why this individual was extracted"
        ],
        "assertions": [
            {
                "object": {
                    "property_name": "NameOfTBoxProperty",
                    "arguments": ["ClassNames", "OrLiterals"]
                },
                "confidence": 0.95,
                "explanation": "brief explanation why this property was extracted"
            }
        ]
    }
}
```
""",
)

ANSWER_GENERATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=[
        "query",
        "individuals",
    ],
    template="""
Based on the following user query and ABox data, determine the appropriate query tuple to pass to the calculate function.

# User Query:
{query}

# Available Individuals in ABox
{individuals}

# Instruction
1. Use only token IDs and candidate IDs that appear in the ABox or disambiguation result.
2. Do not invent, modify, or combine IDs. Use the exact strings as they appear in the individuals list.
3. If a single candidate was selected, report it and briefly explain why.
4. If multiple candidates remain, report that the case is unresolved and list the remaining candidates.
5. If multiple candidates remain, report that the case is unresolved and list the remaining candidates.
6. If the result contains contradictions, report the contradiction and the rejected constraints.
7. Keep the answer concise and interpretable.

Your task is to analyze this information and determine the appropriate query tuple to pass to the disambiguate function.
The query should include:
- sentence_id: MUST be an exact match from the list of individual IDs above
- sentence_id MUST refer to an individual whose class/type is Sentence
""",
)
