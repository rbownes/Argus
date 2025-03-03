"""
Diverse queries organized into 10 themes with 10 queries each.
These can be used to test LLM performance across different domains.
"""

DIVERSE_QUERIES = {
    "science_technology": [
        "Explain quantum entanglement to a high school student",
        "How do mRNA vaccines work?",
        "What are the implications of artificial general intelligence?",
        "Describe the process of nuclear fusion and its potential as an energy source",
        "How does machine learning differ from traditional programming?",
        "What are black holes and how do they affect space-time?",
        "Explain how CRISPR gene editing technology works",
        "What are the main challenges in developing quantum computers?",
        "How does the internet route data around the world?",
        "What's the current understanding of dark matter and dark energy?"
    ],
    "arts_literature": [
        "Compare the writing styles of Jane Austen and Virginia Woolf",
        "How did Impressionism change the course of art history?",
        "Analyze the themes in 'One Hundred Years of Solitude'",
        "What makes Shakespeare's works enduring across centuries?",
        "Explain the significance of Cubism in modern art",
        "How has jazz influenced other music genres?",
        "Discuss the cultural impact of manga and anime globally",
        "What are the defining characteristics of magical realism?",
        "Analyze the narrative techniques in 'To the Lighthouse'",
        "How did Romanticism respond to the Industrial Revolution?"
    ],
    "history_culture": [
        "What factors led to the fall of the Roman Empire?",
        "How did the Silk Road influence cultural exchange?",
        "Explain the significance of the Magna Carta",
        "How did the Renaissance change European society?",
        "What was the impact of the Columbian Exchange on global food systems?",
        "Describe how coffee houses influenced the Age of Enlightenment",
        "Compare and contrast Ancient Greek and Roman political systems",
        "How did the invention of the printing press change society?",
        "What role did women play in the Civil Rights Movement?",
        "How did the Cold War shape international relations?"
    ],
    "philosophy_ethics": [
        "What is the categorical imperative according to Kant?",
        "How does utilitarianism approach ethical dilemmas?",
        "Compare Eastern and Western philosophical approaches to the self",
        "What is the problem of free will in philosophy?",
        "How does existentialism view human responsibility?",
        "Explain the trolley problem and its ethical implications",
        "What is the philosophical concept of the 'state of nature'?",
        "How does virtue ethics differ from consequentialism?",
        "What does Plato's allegory of the cave teach us about perception?",
        "How should we approach the ethics of artificial intelligence?"
    ],
    "business_economics": [
        "Explain how supply and demand determine market prices",
        "What are the differences between macroeconomics and microeconomics?",
        "How do central banks use monetary policy to control inflation?",
        "What is game theory and how is it applied in business strategy?",
        "Explain the concept of comparative advantage in international trade",
        "How does blockchain technology impact financial systems?",
        "What factors contribute to income inequality?",
        "How do network effects create monopolies in digital markets?",
        "What are ESG criteria and why are they important for investors?",
        "Explain how behavioral economics challenges rational choice theory"
    ],
    "health_medicine": [
        "How does the immune system recognize and fight pathogens?",
        "What is the gut-brain connection and how does it affect health?",
        "Explain how antibiotics work and why resistance is a concern",
        "What are the biological mechanisms of aging?",
        "How do vaccines provide immunity at the cellular level?",
        "What is the microbiome and why is it important for health?",
        "Explain how precision medicine personalizes healthcare",
        "What are the different types of diabetes and how are they managed?",
        "How does chronic stress affect physical health?",
        "What advances are being made in regenerative medicine?"
    ],
    "environment_sustainability": [
        "Explain the carbon cycle and how human activities disrupt it",
        "How do ocean currents regulate global climate?",
        "What are realistic solutions to plastic pollution?",
        "Explain how regenerative agriculture can reverse soil degradation",
        "How does biodiversity loss affect ecosystem stability?",
        "What are the challenges and benefits of transitioning to renewable energy?",
        "How do urban planning decisions impact sustainability?",
        "What is the water cycle and how is it affected by climate change?",
        "Explain the concept of a circular economy",
        "How can conservation efforts balance human needs with wildlife protection?"
    ],
    "personal_development": [
        "What techniques can improve critical thinking skills?",
        "How does deliberate practice differ from repetition?",
        "What is growth mindset and how can it be cultivated?",
        "Explain effective strategies for habit formation",
        "How does mindfulness meditation affect cognitive function?",
        "What are evidence-based approaches to goal setting?",
        "How can cognitive biases be recognized and mitigated?",
        "What is emotional intelligence and how can it be developed?",
        "Explain the concept of flow state and how to achieve it",
        "What factors contribute to intrinsic motivation?"
    ],
    "social_issues_politics": [
        "How do different electoral systems affect political representation?",
        "What are the root causes of wealth inequality?",
        "How do social media algorithms affect political polarization?",
        "Explain different approaches to criminal justice reform",
        "What is the relationship between education and social mobility?",
        "How have privacy concerns evolved in the digital age?",
        "What approaches can address homelessness in urban areas?",
        "How do immigration policies affect economic outcomes?",
        "What are the challenges in designing fair healthcare systems?",
        "How do generational differences influence political views?"
    ],
    "mathematics_logic": [
        "Explain Gödel's incompleteness theorems in simple terms",
        "How is calculus applied to solve real-world problems?",
        "What is game theory and how does it model decision-making?",
        "Explain the concept of mathematical infinity",
        "How do prime numbers function in cryptography?",
        "What is the P versus NP problem in computer science?",
        "Explain Bayesian inference and its practical applications",
        "How do different logical fallacies undermine arguments?",
        "What is topology and how does it relate to geometry?",
        "Explain the mathematics behind machine learning algorithms"
    ]
}

def get_all_queries() -> list:
    """Return a flat list of all queries across all themes."""
    return [query for theme_queries in DIVERSE_QUERIES.values() for query in theme_queries]

def get_queries_by_theme(theme: str) -> list:
    """Return queries for a specific theme."""
    return DIVERSE_QUERIES.get(theme, [])

def get_themes() -> list:
    """Return a list of all theme names."""
    return list(DIVERSE_QUERIES.keys())

if __name__ == "__main__":
    # Print statistics about the queries
    print(f"Total themes: {len(get_themes())}")
    print(f"Total queries: {len(get_all_queries())}")
    for theme in get_themes():
        print(f"Theme '{theme}' has {len(get_queries_by_theme(theme))} queries")