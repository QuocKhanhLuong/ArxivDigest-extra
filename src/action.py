from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

import argparse
import yaml
import os
from dotenv import load_dotenv
import openai
from relevancy import generate_relevance_score, process_subject_fields
from download_new_papers import get_papers
from datetime import date

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

# Hackathon quality code. Don't judge too harshly.
# Feel free to submit pull requests to improve the code.

topics = {
    "Physics": "",
    "Mathematics": "math",
    "Computer Science": "cs",
    "Quantitative Biology": "q-bio",
    "Quantitative Finance": "q-fin",
    "Statistics": "stat",
    "Electrical Engineering and Systems Science": "eess",
    "Economics": "econ",
}

physics_topics = {
    "Astrophysics": "astro-ph",
    "Condensed Matter": "cond-mat",
    "General Relativity and Quantum Cosmology": "gr-qc",
    "High Energy Physics - Experiment": "hep-ex",
    "High Energy Physics - Lattice": "hep-lat",
    "High Energy Physics - Phenomenology": "hep-ph",
    "High Energy Physics - Theory": "hep-th",
    "Mathematical Physics": "math-ph",
    "Nonlinear Sciences": "nlin",
    "Nuclear Experiment": "nucl-ex",
    "Nuclear Theory": "nucl-th",
    "Physics": "physics",
    "Quantum Physics": "quant-ph",
}


# TODO: surely theres a better way
category_map = {
    "Astrophysics": [
        "Astrophysics of Galaxies",
        "Cosmology and Nongalactic Astrophysics",
        "Earth and Planetary Astrophysics",
        "High Energy Astrophysical Phenomena",
        "Instrumentation and Methods for Astrophysics",
        "Solar and Stellar Astrophysics",
    ],
    "Condensed Matter": [
        "Disordered Systems and Neural Networks",
        "Materials Science",
        "Mesoscale and Nanoscale Physics",
        "Other Condensed Matter",
        "Quantum Gases",
        "Soft Condensed Matter",
        "Statistical Mechanics",
        "Strongly Correlated Electrons",
        "Superconductivity",
    ],
    "General Relativity and Quantum Cosmology": ["None"],
    "High Energy Physics - Experiment": ["None"],
    "High Energy Physics - Lattice": ["None"],
    "High Energy Physics - Phenomenology": ["None"],
    "High Energy Physics - Theory": ["None"],
    "Mathematical Physics": ["None"],
    "Nonlinear Sciences": [
        "Adaptation and Self-Organizing Systems",
        "Cellular Automata and Lattice Gases",
        "Chaotic Dynamics",
        "Exactly Solvable and Integrable Systems",
        "Pattern Formation and Solitons",
    ],
    "Nuclear Experiment": ["None"],
    "Nuclear Theory": ["None"],
    "Physics": [
        "Accelerator Physics",
        "Applied Physics",
        "Atmospheric and Oceanic Physics",
        "Atomic and Molecular Clusters",
        "Atomic Physics",
        "Biological Physics",
        "Chemical Physics",
        "Classical Physics",
        "Computational Physics",
        "Data Analysis, Statistics and Probability",
        "Fluid Dynamics",
        "General Physics",
        "Geophysics",
        "History and Philosophy of Physics",
        "Instrumentation and Detectors",
        "Medical Physics",
        "Optics",
        "Physics and Society",
        "Physics Education",
        "Plasma Physics",
        "Popular Physics",
        "Space Physics",
    ],
    "Quantum Physics": ["None"],
    "Mathematics": [
        "Algebraic Geometry",
        "Algebraic Topology",
        "Analysis of PDEs",
        "Category Theory",
        "Classical Analysis and ODEs",
        "Combinatorics",
        "Commutative Algebra",
        "Complex Variables",
        "Differential Geometry",
        "Dynamical Systems",
        "Functional Analysis",
        "General Mathematics",
        "General Topology",
        "Geometric Topology",
        "Group Theory",
        "History and Overview",
        "Information Theory",
        "K-Theory and Homology",
        "Logic",
        "Mathematical Physics",
        "Metric Geometry",
        "Number Theory",
        "Numerical Analysis",
        "Operator Algebras",
        "Optimization and Control",
        "Probability",
        "Quantum Algebra",
        "Representation Theory",
        "Rings and Algebras",
        "Spectral Theory",
        "Statistics Theory",
        "Symplectic Geometry",
    ],
    "Computer Science": [
        "Artificial Intelligence",
        "Computation and Language",
        "Computational Complexity",
        "Computational Engineering, Finance, and Science",
        "Computational Geometry",
        "Computer Science and Game Theory",
        "Computer Vision and Pattern Recognition",
        "Computers and Society",
        "Cryptography and Security",
        "Data Structures and Algorithms",
        "Databases",
        "Digital Libraries",
        "Discrete Mathematics",
        "Distributed, Parallel, and Cluster Computing",
        "Emerging Technologies",
        "Formal Languages and Automata Theory",
        "General Literature",
        "Graphics",
        "Hardware Architecture",
        "Human-Computer Interaction",
        "Information Retrieval",
        "Information Theory",
        "Logic in Computer Science",
        "Machine Learning",
        "Mathematical Software",
        "Multiagent Systems",
        "Multimedia",
        "Networking and Internet Architecture",
        "Neural and Evolutionary Computing",
        "Numerical Analysis",
        "Operating Systems",
        "Other Computer Science",
        "Performance",
        "Programming Languages",
        "Robotics",
        "Social and Information Networks",
        "Software Engineering",
        "Sound",
        "Symbolic Computation",
        "Systems and Control",
    ],
    "Quantitative Biology": [
        "Biomolecules",
        "Cell Behavior",
        "Genomics",
        "Molecular Networks",
        "Neurons and Cognition",
        "Other Quantitative Biology",
        "Populations and Evolution",
        "Quantitative Methods",
        "Subcellular Processes",
        "Tissues and Organs",
    ],
    "Quantitative Finance": [
        "Computational Finance",
        "Economics",
        "General Finance",
        "Mathematical Finance",
        "Portfolio Management",
        "Pricing of Securities",
        "Risk Management",
        "Statistical Finance",
        "Trading and Market Microstructure",
    ],
    "Statistics": [
        "Applications",
        "Computation",
        "Machine Learning",
        "Methodology",
        "Other Statistics",
        "Statistics Theory",
    ],
    "Electrical Engineering and Systems Science": [
        "Audio and Speech Processing",
        "Image and Video Processing",
        "Signal Processing",
        "Systems and Control",
    ],
    "Economics": ["Econometrics", "General Economics", "Theoretical Economics"],
}


def generate_body(topic, categories, interest, threshold):
    f_papers = []
    if topic == "Physics":
        raise RuntimeError("You must choose a physics subtopic.")
    elif topic in physics_topics:
        abbr = physics_topics[topic]
    elif topic in topics:
        abbr = topics[topic]
    else:
        raise RuntimeError(f"Invalid topic {topic}")
    if categories:
        for category in categories:
            if category not in category_map[topic]:
                raise RuntimeError(f"{category} is not a category of {topic}")
        papers = get_papers(abbr)

        papers = [
            t
            for t in papers
            if bool(set(process_subject_fields(t["subjects"])) & set(categories))
        ]

    else:
        papers = get_papers(abbr)
    if interest:
        # Use Gemini as primary model if available, fallback to OpenAI
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        if gemini_key:
            # Use Gemini directly for analysis
            print("🤖 Using Gemini API for paper analysis")
            from gemini_utils import analyze_papers_with_gemini
            all_analyzed = analyze_papers_with_gemini(
                papers,
                query={"interest": interest},
                model_name="gemini-1.5-flash"
            )
            
            # Filter by threshold
            relevancy = []
            for paper in all_analyzed:
                score = paper.get("Relevancy score", 0)
                if isinstance(score, str):
                    try:
                        if '/' in score:
                            score = int(score.split('/')[0])
                        else:
                            score = int(score)
                    except (ValueError, TypeError):
                        score = 0
                
                if score >= threshold:
                    relevancy.append(paper)
                    print(f"✅ Paper '{paper.get('title', 'Unknown')[:50]}...' - Score: {score}")
                else:
                    print(f"❌ Paper '{paper.get('title', 'Unknown')[:50]}...' - Score: {score} (below threshold {threshold})")
            
            print(f"🎯 Final result: {len(relevancy)} papers passed threshold {threshold} out of {len(all_analyzed)} analyzed")
            hallucination = False
        elif openai_key:
            # Fallback to OpenAI
            print("🤖 Using OpenAI API for paper analysis")
            relevancy, hallucination = generate_relevance_score(
                papers,
                query={"interest": interest},
                threshold_score=threshold,
                num_paper_in_prompt=2,
            )
        else:
            raise RuntimeError("No supported AI API key found for paper analysis")

        body = "<br><br>".join(
            [
                f'<b>Subject: </b>{paper.get("subjects", "N/A")}<br><b>Title:</b> <a href="{paper.get("main_page", "#")}">{paper.get("title", "Unknown")}</a><br><b>Authors:</b> {paper.get("authors", "Unknown")}<br>'
                f'<b>Score:</b> {paper.get("Relevancy score", "N/A")}<br><b>Reason:</b> {paper.get("Reasons for match", "N/A")}<br>'
                f'<b>Goal:</b> {paper.get("Goal", "N/A")}<br><b>Data</b>: {paper.get("Data", "N/A")}<br><b>Methodology:</b> {paper.get("Methodology", "N/A")}<br>'
                f'<b>Experiments & Results</b>: {paper.get("Experiments & Results", "N/A")}<br><b>Git</b>: {paper.get("Git", "N/A")}<br>'
                f'<b>Discussion & Next steps</b>: {paper.get("Discussion & Next steps", "N/A")}'
                for paper in relevancy
            ]
        )
        if hallucination:
            body = (
                "Warning: the model hallucinated some papers. We have tried to remove them, but the scores may not be accurate.<br><br>"
                + body
            )
    else:
        body = "<br><br>".join(
            [
                f'Title: <a href="{paper["main_page"]}">{paper["title"]}</a><br>Authors: {paper["authors"]}'
                for paper in papers
            ]
        )
    return body

def get_date():
    today = date.today()
    formatted_date = today.strftime("%d%m%Y")
    return formatted_date

if __name__ == "__main__":
    # Load the .env file.
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="yaml config file to use", default="config.yaml"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Check for available AI API keys - prioritize Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not any([gemini_key, openai_key, anthropic_key]):
        raise RuntimeError("No AI API key found. Please set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
    
    # Prioritize Gemini API for best performance and cost
    if gemini_key:
        print("✅ Using Gemini API (recommended)")
    elif openai_key:
        openai.api_key = openai_key
        print("✅ Using OpenAI API")
    elif anthropic_key:
        print("✅ Using Anthropic API")

    topic = config["topic"]
    categories = config["categories"]
    from_email = os.environ.get("FROM_EMAIL")
    to_email = os.environ.get("TO_EMAIL")
    threshold = config["threshold"]
    interest = config["interest"]
    body = generate_body(topic, categories, interest, threshold)
    today_date = get_date()
    with open(f"digest_{today_date}.html", "w") as f:
        f.write(body)
    if os.environ.get("SENDGRID_API_KEY", None):
        sg = SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
        from_email = Email(from_email)  # Change to your verified sender
        to_email = To(to_email)
        subject = date.today().strftime("Personalized arXiv Digest, %d %b %Y")
        content = Content("text/html", body)
        mail = Mail(from_email, to_email, subject, content)
        mail_json = mail.get()

        # Send an HTTP POST request to /mail/send
        response = sg.client.mail.send.post(request_body=mail_json)
        if response.status_code >= 200 and response.status_code <= 300:
            print("Send test email: Success!")
        else:
            print("Send test email: Failure ({response.status_code}, {response.text})")
    else:
        print("No sendgrid api key found. Skipping email")
