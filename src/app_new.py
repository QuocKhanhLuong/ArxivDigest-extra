import gradio as gr
from download_new_papers import get_papers
import utils
from relevancy import generate_relevance_score, process_subject_fields
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content

import os
import openai
import datetime
import yaml
from paths import DATA_DIR, DIGEST_DIR
from model_manager import model_manager, ModelProvider
from gemini_utils import setup_gemini_api, get_topic_clustering

# Load config file
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"threshold": 2}  # Default threshold if config loading fails

config = load_config()

# Helper function to filter papers by threshold
def filter_papers_by_threshold(papers, threshold):
    """Filter papers by relevancy score threshold"""
    print(f"\n===== FILTERING PAPERS =====")
    print(f"Only showing papers with relevancy score >= {threshold}")
    print(f"(Change this value in config.yaml if needed)")
    
    # Debug the paper scores
    for i, paper in enumerate(papers):
        print(f"Paper {i+1} - Title: {paper.get('title', 'No title')}")
        print(f"   - Score: {paper.get('Relevancy score', 'No score')}")
        print(f"   - Fields: {list(paper.keys())}")
    
    # First extract data from gemini_analysis if it exists and Relevancy score doesn't
    for paper in papers:
        if "gemini_analysis" in paper and "Relevancy score" not in paper:
            print(f"Extracting analysis data for paper: {paper.get('title')}")
            gemini_data = paper["gemini_analysis"]
            
            # Map Gemini analysis fields to expected fields
            field_mapping = {
                "relevance_score": "Relevancy score",
                "relationship_score": "Relevancy score",
                "paper_relevance": "Relevancy score",
                "paper's_relationship_to_the_user's_interests": "Relevancy score",
                "key_innovations": "Key innovations",
                "critical_analysis": "Critical analysis",
                "methodology_summary": "Methodology",
                "technical_significance": "Critical analysis",
                "related_research": "Related work"
            }
            
            # Copy fields using mapping
            for gemini_field, expected_field in field_mapping.items():
                if gemini_field in gemini_data:
                    paper[expected_field] = gemini_data[gemini_field]
                    print(f"  - Mapped {gemini_field} to {expected_field}")
                
            # If we have no score yet, look for a number in other fields
            if "Relevancy score" not in paper:
                # Try to find a relevance score in any field
                for field, value in gemini_data.items():
                    if isinstance(value, (int, float)) and 1 <= value <= 10:
                        paper["Relevancy score"] = value
                        print(f"  - Found score {value} in field {field}")
                        break
                    elif isinstance(value, str) and "score" in field.lower():
                        try:
                            # Try to extract a number from the string
                            import re
                            numbers = re.findall(r'\d+', value)
                            if numbers:
                                score = int(numbers[0])
                                if 1 <= score <= 10:  # Validate score range
                                    paper["Relevancy score"] = score
                                    print(f"  - Extracted score {score} from {field}: {value}")
                                    break
                        except:
                            pass
            
            # If still no score, default to threshold to include paper
            if "Relevancy score" not in paper:
                paper["Relevancy score"] = threshold
                print(f"  - Assigned default score {threshold}")
                
            # Add some reasonable defaults for missing fields
            if "Reasons for match" not in paper and "topic_classification" in gemini_data:
                paper["Reasons for match"] = gemini_data.get("topic_classification", "Not provided")
                
            # Set missing fields with default values
            for field in ["Key innovations", "Critical analysis", "Goal", "Data", "Methodology", 
                         "Implementation details", "Experiments & Results", "Discussion & Next steps",
                         "Related work", "Practical applications", "Key takeaways"]:
                if field not in paper:
                    paper[field] = "Not available in analysis"
    
    # Ensure scores are properly parsed to integers
    for paper in papers:
        if "Relevancy score" in paper and not isinstance(paper["Relevancy score"], int):
            try:
                if isinstance(paper["Relevancy score"], str) and "/" in paper["Relevancy score"]:
                    paper["Relevancy score"] = int(paper["Relevancy score"].split("/")[0])
                else:
                    paper["Relevancy score"] = int(paper["Relevancy score"])
            except (ValueError, TypeError):
                print(f"WARNING: Could not convert score '{paper.get('Relevancy score')}' to integer for paper: {paper.get('title')}")
                paper["Relevancy score"] = threshold  # Use threshold as default
    
    # Make sure all papers have required fields
    required_fields = [
        "Relevancy score", "Reasons for match", "Key innovations", "Critical analysis",
        "Goal", "Data", "Methodology", "Implementation details", "Experiments & Results",
        "Git", "Discussion & Next steps", "Related work", "Practical applications", 
        "Key takeaways"
    ]
    
    for paper in papers:
        # Make sure it has a relevancy score
        if "Relevancy score" not in paper:
            paper["Relevancy score"] = threshold
            print(f"Assigned default threshold score to paper: {paper.get('title')}")
            
        # Add missing fields with default values
        for field in required_fields:
            if field not in paper:
                paper[field] = f"Not available in analysis"
                print(f"Added missing field {field} to paper: {paper.get('title')}")
    
    # Now filter papers
    filtered_papers = [p for p in papers if p.get("Relevancy score", 0) >= threshold]
    print(f"After filtering: {len(filtered_papers)} papers remain out of {len(papers)}")
    
    if len(filtered_papers) == 0 and len(papers) > 0:
        print("WARNING: No papers passed the threshold filter. Using all papers.")
        filtered_papers = papers
        
    return filtered_papers
from design_automation import (
    is_design_automation_paper, 
    categorize_design_paper, 
    analyze_design_techniques,
    extract_design_metrics,
    get_related_design_papers,
    create_design_analysis_prompt
)

topics = {
    "Physics": "",
    "Mathematics": "math",
    "Computer Science": "cs",
    "Quantitative Biology": "q-bio",
    "Quantitative Finance": "q-fin",
    "Statistics": "stat",
    "Electrical Engineering and Systems Science": "eess",
    "Economics": "econ"
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
    "Quantum Physics": "quant-ph"
}

categories_map = {
    "Astrophysics": ["Astrophysics of Galaxies", "Cosmology and Nongalactic Astrophysics", "Earth and Planetary Astrophysics", "High Energy Astrophysical Phenomena", "Instrumentation and Methods for Astrophysics", "Solar and Stellar Astrophysics"],
    "Condensed Matter": ["Disordered Systems and Neural Networks", "Materials Science", "Mesoscale and Nanoscale Physics", "Other Condensed Matter", "Quantum Gases", "Soft Condensed Matter", "Statistical Mechanics", "Strongly Correlated Electrons", "Superconductivity"],
    "General Relativity and Quantum Cosmology": ["None"],
    "High Energy Physics - Experiment": ["None"],
    "High Energy Physics - Lattice": ["None"],
    "High Energy Physics - Phenomenology": ["None"],
    "High Energy Physics - Theory": ["None"],
    "Mathematical Physics": ["None"],
    "Nonlinear Sciences": ["Adaptation and Self-Organizing Systems", "Cellular Automata and Lattice Gases", "Chaotic Dynamics", "Exactly Solvable and Integrable Systems", "Pattern Formation and Solitons"],
    "Nuclear Experiment": ["None"],
    "Nuclear Theory": ["None"],
    "Physics": ["Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics", "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics"],
    "Quantum Physics": ["None"],
    "Mathematics": ["Algebraic Geometry", "Algebraic Topology", "Analysis of PDEs", "Category Theory", "Classical Analysis and ODEs", "Combinatorics", "Commutative Algebra", "Complex Variables", "Differential Geometry", "Dynamical Systems", "Functional Analysis", "General Mathematics", "General Topology", "Geometric Topology", "Group Theory", "History and Overview", "Information Theory", "K-Theory and Homology", "Logic", "Mathematical Physics", "Metric Geometry", "Number Theory", "Numerical Analysis", "Operator Algebras", "Optimization and Control", "Probability", "Quantum Algebra", "Representation Theory", "Rings and Algebras", "Spectral Theory", "Statistics Theory", "Symplectic Geometry"],
    "Computer Science": ["Artificial Intelligence", "Computation and Language", "Computational Complexity", "Computational Engineering, Finance, and Science", "Computational Geometry", "Computer Science and Game Theory", "Computer Vision and Pattern Recognition", "Computers and Society", "Cryptography and Security", "Data Structures and Algorithms", "Databases", "Digital Libraries", "Discrete Mathematics", "Distributed, Parallel, and Cluster Computing", "Emerging Technologies", "Formal Languages and Automata Theory", "General Literature", "Graphics", "Hardware Architecture", "Human-Computer Interaction", "Information Retrieval", "Information Theory", "Logic in Computer Science", "Machine Learning", "Mathematical Software", "Multiagent Systems", "Multimedia", "Networking and Internet Architecture", "Neural and Evolutionary Computing", "Numerical Analysis", "Operating Systems", "Other Computer Science", "Performance", "Programming Languages", "Robotics", "Social and Information Networks", "Software Engineering", "Sound", "Symbolic Computation", "Systems and Control"],
    "Quantitative Biology": ["Biomolecules", "Cell Behavior", "Genomics", "Molecular Networks", "Neurons and Cognition", "Other Quantitative Biology", "Populations and Evolution", "Quantitative Methods", "Subcellular Processes", "Tissues and Organs"],
    "Quantitative Finance": ["Computational Finance", "Economics", "General Finance", "Mathematical Finance", "Portfolio Management", "Pricing of Securities", "Risk Management", "Statistical Finance", "Trading and Market Microstructure"],
    "Statistics": ["Applications", "Computation", "Machine Learning", "Methodology", "Other Statistics", "Statistics Theory"],
    "Electrical Engineering and Systems Science": ["Audio and Speech Processing", "Image and Video Processing", "Signal Processing", "Systems and Control"],
    "Economics": ["Econometrics", "General Economics", "Theoretical Economics"]
}


def generate_html_report(papers, title="ArXiv Digest Results", topic=None, category=None):
    """Generate an HTML report for the papers and save to file.
    
    Args:
        papers: List of paper dictionaries
        title: Title for the HTML report
        topic: Optional topic name for filename
        category: Optional category name for filename
        
    Returns:
        Path to the HTML file
    """
    # Debug: Log what fields are available in each paper
    print(f"Generating HTML report for {len(papers)} papers")
    for i, paper in enumerate(papers):
        print(f"Paper {i+1} fields: {list(paper.keys())}")
        if "Key innovations" in paper:
            print(f"Paper {i+1} has Key innovations: {paper['Key innovations'][:50]}...")
        if "Critical analysis" in paper:
            print(f"Paper {i+1} has Critical analysis: {paper['Critical analysis'][:50]}...")
    
    # Create a date for the filename (without time)
    date = datetime.datetime.now().strftime("%Y%m%d")
    
    # Create filename with topic if provided
    if topic:
        # Clean up topic name for filename (remove spaces, etc.)
        topic_clean = topic.lower().replace(" ", "_").replace("/", "_")
        html_file = os.path.join(DIGEST_DIR, f"arxiv_digest_{topic_clean}_{date}.html")
    else:
        html_file = os.path.join(DIGEST_DIR, f"arxiv_digest_{date}.html")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; max-width: 1200px; margin: 0 auto; }}
            .paper {{ margin-bottom: 40px; border-bottom: 2px solid #ddd; padding-bottom: 30px; }}
            .title {{ font-size: 22px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
            .authors {{ font-style: italic; margin: 8px 0; color: #34495e; }}
            .categories {{ color: #3498db; margin-bottom: 15px; font-size: 14px; }}
            .abstract {{ margin: 15px 0; line-height: 1.6; background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .score {{ font-weight: bold; color: #e74c3c; margin: 12px 0; font-size: 18px; }}
            .reason {{ margin: 15px 0; background-color: #f5f9f9; padding: 15px; border-left: 4px solid #2ecc71; border-radius: 0 5px 5px 0; }}
            .techniques {{ margin: 12px 0; color: #16a085; }}
            a {{ color: #2980b9; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .stats {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .footer {{ margin-top: 40px; font-size: 12px; color: #7f8c8d; text-align: center; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 15px; margin-bottom: 30px; }}
            .section {{ margin: 15px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .section-title {{ font-weight: bold; color: #2c3e50; margin-bottom: 5px; font-size: 16px; }}
            .design-info {{ background-color: #e8f6f3; padding: 15px; margin: 15px 0; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .links {{ margin-top: 20px; background-color: #f8f9fa; padding: 10px; border-radius: 5px; display: inline-block; }}
            .key-section {{ margin: 15px 0; padding: 15px; background-color: #fdf5e6; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .implementation {{ margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .experiments {{ margin: 15px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .discussion {{ margin: 15px 0; padding: 15px; background-color: #f0fff0; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .paper-navigation {{ position: sticky; top: 20px; float: right; background-color: white; padding: 15px; border: 1px solid #eee; border-radius: 5px; margin-left: 20px; max-width: 250px; }}
            .paper-navigation ul {{ list-style-type: none; padding: 0; margin: 0; }}
            .paper-navigation li {{ margin: 5px 0; }}
            .paper-navigation a {{ text-decoration: none; }}
            
            @media print {{
                .paper-navigation {{ display: none; }}
                .paper {{ page-break-after: always; }}
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="stats">
            <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Found {len(papers)} papers</p>
            <p>Topics: {topic or "All"}</p>
        </div>
        
        <!-- Create table of contents -->
        <div class="paper-navigation">
            <h3>Papers</h3>
            <ul>
                {' '.join([f'<li><a href="#paper-{i}">{p.get("title", "Paper " + str(i+1))[:40]}...</a></li>' for i, p in enumerate(papers)])}
            </ul>
        </div>
    """
    
    # Check if we have any papers
    if not papers:
        html += """
        <div class="paper">
            <div class="title">No papers found matching your criteria</div>
            <div class="abstract">
                <p>No papers met the relevancy threshold criteria. You can:</p>
                <ul>
                    <li>Lower the threshold in config.yaml (currently set to {threshold})</li>
                    <li>Try different research interests</li>
                    <li>Check different categories or topics</li>
                </ul>
            </div>
        </div>
        """.format(threshold=config.get("threshold", 2))
    
    # Add papers
    for i, paper in enumerate(papers):
        paper_id = f"paper-{i}"
        html += f"""
        <div id="{paper_id}" class="paper">
            <div class="title"><a href="{paper.get("main_page", "#")}" target="_blank">{paper.get("title", "No title")}</a></div>
            <div class="authors">{paper.get("authors", "Unknown authors")}</div>
            <div class="categories">Subject: {paper.get("subjects", "N/A")}</div>
            """
        
        # Add relevancy score and reasons if available
        if "Relevancy score" in paper:
            html += f'<div class="score">Relevancy Score: {paper.get("Relevancy score", "N/A")}</div>'
        
        if "Reasons for match" in paper:
            html += f'<div class="reason"><b>Reason for Relevance:</b> {paper.get("Reasons for match", "")}</div>'
            
        # Add design information if available
        if "design_category" in paper or "design_techniques" in paper:
            html += '<div class="design-info">'
            if "design_category" in paper:
                html += f'<div><b>Design Category:</b> {paper.get("design_category", "")}</div>'
            if "design_techniques" in paper:
                html += f'<div><b>Design Techniques:</b> {", ".join(paper.get("design_techniques", []))}</div>'
            html += '</div>'
            
        # Add abstract
        if "abstract" in paper:
            html += f'<div class="abstract"><b>Abstract:</b> {paper.get("abstract", "")}</div>'
        
        # Add key innovations and critical analysis with special styling
        if "Key innovations" in paper:
            html += f'<div class="key-section"><div class="section-title">Key Innovations:</div> {paper.get("Key innovations", "")}</div>'
            print(f"Added Key innovations for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Key innovations field")
        
        if "Critical analysis" in paper:
            html += f'<div class="key-section"><div class="section-title">Critical Analysis:</div> {paper.get("Critical analysis", "")}</div>'
            print(f"Added Critical analysis for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Critical analysis field")
            
        # Add goal
        if "Goal" in paper:
            html += f'<div class="key-section"><div class="section-title">Goal:</div> {paper.get("Goal", "")}</div>'
            print(f"Added Goal for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Goal field")
            
        # Add Data
        if "Data" in paper:
            html += f'<div class="implementation"><div class="section-title">Data:</div> {paper.get("Data", "")}</div>'
            print(f"Added Data for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Data field")
            
        # Add Methodology
        if "Methodology" in paper:
            html += f'<div class="implementation"><div class="section-title">Methodology:</div> {paper.get("Methodology", "")}</div>'
            print(f"Added Methodology for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Methodology field")
            
        # Add implementation details
        if "Implementation details" in paper:
            html += f'<div class="implementation"><div class="section-title">Implementation Details:</div> {paper.get("Implementation details", "")}</div>'
            print(f"Added Implementation details for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Implementation details field")
            
        # Add experiments and results
        if "Experiments & Results" in paper:
            html += f'<div class="experiments"><div class="section-title">Experiments & Results:</div> {paper.get("Experiments & Results", "")}</div>'
            print(f"Added Experiments & Results for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Experiments & Results field")
            
        # Add discussion and next steps
        if "Discussion & Next steps" in paper:
            html += f'<div class="discussion"><div class="section-title">Discussion & Next Steps:</div> {paper.get("Discussion & Next steps", "")}</div>'
            print(f"Added Discussion & Next steps for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Discussion & Next steps field")
            
        # Add Related work
        if "Related work" in paper:
            html += f'<div class="discussion"><div class="section-title">Related Work:</div> {paper.get("Related work", "")}</div>'
            print(f"Added Related work for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Related work field")
            
        # Add Practical applications
        if "Practical applications" in paper:
            html += f'<div class="discussion"><div class="section-title">Practical Applications:</div> {paper.get("Practical applications", "")}</div>'
            print(f"Added Practical applications for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Practical applications field")
            
        # Add Key takeaways
        if "Key takeaways" in paper:
            html += f'<div class="key-section"><div class="section-title">Key Takeaways:</div> {paper.get("Key takeaways", "")}</div>'
            print(f"Added Key takeaways for paper {i+1}")
        else:
            print(f"Paper {i+1} is missing Key takeaways field")
        
        # Add remaining sections that weren't already handled specifically above
        for key, value in paper.items():
            if key in ["title", "authors", "subjects", "main_page", "Relevancy score", "Reasons for match", 
                      "design_category", "design_techniques", "summarized_text", "abstract",
                      "Key innovations", "Critical analysis", "Goal", "Data", "Methodology", 
                      "Implementation details", "Experiments & Results", "Discussion & Next steps",
                      "Related work", "Practical applications", "Key takeaways"]:
                continue
            
            if isinstance(value, str) and value.strip():
                # Choose appropriate styling based on the section
                section_class = "section"
                
                if "goal" in key.lower() or "aim" in key.lower():
                    section_class = "key-section"
                elif "data" in key.lower() or "methodology" in key.lower():
                    section_class = "implementation"
                elif "related" in key.lower() or "practical" in key.lower() or "takeaway" in key.lower():
                    section_class = "discussion"
                
                html += f'<div class="{section_class}"><div class="section-title">{key}:</div> {value}</div>'
                print(f"Added additional field {key} for paper {i+1}")
            
        # Add links
        html += f"""
            <div class="links">
                <a href="{paper.get("pdf", paper.get("main_page", "#") + ".pdf")}" target="_blank">PDF</a> | 
                <a href="{paper.get("main_page", "#")}" target="_blank">arXiv</a> |
                <a href="#" onclick="window.scrollTo(0,0); return false;">Back to Top</a>
            </div>
        </div>
        """
    
    html += """
        <div class="footer">
            <p>Generated by ArXiv Digest</p>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open(html_file, "w") as f:
        f.write(html)
    
    print(f"Saved HTML report to {html_file}")
    return html_file

def sample(email, topic, physics_topic, categories, interest, use_openai, use_gemini, use_anthropic, 
           openai_model, gemini_model, anthropic_model, special_analysis, mechanistic_interpretability, technical_ai_safety,
           design_automation, design_reference_paper, design_techniques, design_categories):
    if not topic:
        raise gr.Error("You must choose a topic.")
    if topic == "Physics":
        if isinstance(physics_topic, list):
            raise gr.Error("You must choose a physics topic.")
        topic = physics_topic
        abbr = physics_topics[topic]
    else:
        abbr = topics[topic]
    
    # Check if at least one model is selected
    if not (use_openai or use_gemini or use_anthropic):
        raise gr.Error("You must select at least one model provider (OpenAI, Gemini, or Claude)")
    
    # Get papers based on categories
    if categories:
        papers = get_papers(abbr)
        papers = [
            t for t in papers
            if bool(set(process_subject_fields(t['subjects'])) & set(categories))][:4]
    else:
        papers = get_papers(abbr, limit=4)
    
    if interest:
        # Build list of providers to use
        providers = []
        model_names = {}
        
        if use_openai:
            if not model_manager.is_provider_available(ModelProvider.OPENAI):
                if not openai.api_key:
                    raise gr.Error("Set your OpenAI API key in the OpenAI tab first")
                else:
                    model_manager.register_openai(openai.api_key)
            providers.append(ModelProvider.OPENAI)
            model_names[ModelProvider.OPENAI] = openai_model
            
        if use_gemini:
            if not model_manager.is_provider_available(ModelProvider.GEMINI):
                raise gr.Error("Set your Gemini API key in the Gemini tab first")
            providers.append(ModelProvider.GEMINI)
            model_names[ModelProvider.GEMINI] = gemini_model
            
        if use_anthropic:
            if not model_manager.is_provider_available(ModelProvider.ANTHROPIC):
                raise gr.Error("Set your Anthropic API key in the Anthropic tab first")
            providers.append(ModelProvider.ANTHROPIC)
            model_names[ModelProvider.ANTHROPIC] = anthropic_model
        
        # Check if we need to find design automation papers
        if design_automation:
            # Filter for design automation papers
            design_papers = [p for p in papers if is_design_automation_paper(p)]
            
            # Filter by techniques if specified
            if design_techniques:
                filtered_papers = []
                for paper in design_papers:
                    paper_techniques = analyze_design_techniques(paper)
                    if any(technique in design_techniques for technique in paper_techniques):
                        filtered_papers.append(paper)
                design_papers = filtered_papers if filtered_papers else design_papers
                
            # Filter by categories if specified
            if design_categories:
                filtered_papers = []
                for paper in design_papers:
                    paper_category = categorize_design_paper(paper)
                    if any(category in paper_category for category in design_categories):
                        filtered_papers.append(paper)
                design_papers = filtered_papers if filtered_papers else design_papers
                
            # Find related papers if reference paper is specified
            if design_reference_paper:
                related_papers = get_related_design_papers(design_reference_paper, papers)
                if related_papers:
                    design_papers = related_papers
            
            # Use these papers if we found any, otherwise fallback to regular papers
            if design_papers:
                papers = design_papers
        
        # Process papers directly instead of using model_manager
        print("\n===== ANALYZING PAPERS FOR EMAIL =====")
        print(f"Processing {len(papers)} papers...")
        relevancy = []
        hallucination = False
        
        # Use OpenAI if selected
        if use_openai and model_manager.is_provider_available(ModelProvider.OPENAI):
            try:
                # Import directly to avoid circular imports
                from relevancy import generate_relevance_score
                openai_results, hallu = generate_relevance_score(
                    papers,
                    query={"interest": interest},
                    model_name=openai_model,
                    threshold_score=0,  # We'll filter later
                    num_paper_in_prompt=2
                )
                hallucination = hallucination or hallu
                relevancy.extend(openai_results)
                print(f"OpenAI analysis added {len(openai_results)} papers")
            except Exception as e:
                print(f"Error during OpenAI analysis: {e}")
        
        # Use Gemini if selected and no papers yet
        if use_gemini and model_manager.is_provider_available(ModelProvider.GEMINI) and len(relevancy) == 0:
            try:
                # Import directly to avoid circular imports
                from gemini_utils import analyze_papers_with_gemini
                gemini_papers = analyze_papers_with_gemini(
                    papers,
                    query={"interest": interest},
                    model_name=gemini_model
                )
                # Process papers to ensure they have the right fields
                for paper in gemini_papers:
                    if 'gemini_analysis' in paper:
                        # Copy all fields from gemini_analysis to the paper object
                        for key, value in paper['gemini_analysis'].items():
                            paper[key] = value
                
                relevancy.extend(gemini_papers)
                print(f"Gemini analysis added {len(gemini_papers)} papers")
            except Exception as e:
                print(f"Error during Gemini analysis: {e}")
                
        # Use Anthropic if selected and no papers yet
        if use_anthropic and model_manager.is_provider_available(ModelProvider.ANTHROPIC) and len(relevancy) == 0:
            print("Anthropic/Claude analysis not yet implemented")
        
        print(f"Total papers after analysis: {len(relevancy)}")
        
        # Filter papers by threshold from config
        threshold = config.get("threshold", 2)
        relevancy = filter_papers_by_threshold(relevancy, threshold)
        
        # Add design automation information if requested
        if design_automation and relevancy:
            for paper in relevancy:
                paper["design_category"] = categorize_design_paper(paper)
                paper["design_techniques"] = analyze_design_techniques(paper)
                paper["design_metrics"] = extract_design_metrics(paper)
                
                # Perform detailed design automation analysis on highest scored papers
                if paper.get("Relevancy score", 0) >= 7 and (use_openai or use_gemini or use_anthropic):
                    # Select provider for design analysis
                    provider = None
                    model = None
                    
                    if use_openai and model_manager.is_provider_available(ModelProvider.OPENAI):
                        provider = ModelProvider.OPENAI
                        model = openai_model
                    elif use_gemini and model_manager.is_provider_available(ModelProvider.GEMINI):
                        provider = ModelProvider.GEMINI
                        model = gemini_model
                    elif use_anthropic and model_manager.is_provider_available(ModelProvider.ANTHROPIC):
                        provider = ModelProvider.ANTHROPIC
                        model = anthropic_model
                        
                    if provider:
                        design_analysis = model_manager.analyze_design_automation(
                            paper,
                            provider=provider,
                            model_name=model
                        )
                        if design_analysis and "error" not in design_analysis:
                            paper["design_analysis"] = design_analysis
        
        # Add specialized analysis if requested
        if special_analysis and len(relevancy) > 0:
            # Get topic clustering from Gemini if available
            if use_gemini and model_manager.is_provider_available(ModelProvider.GEMINI):
                try:
                    clusters = get_topic_clustering(relevancy, model_name=gemini_model)
                    cluster_info = "\n\n=== TOPIC CLUSTERS ===\n"
                    for i, cluster in enumerate(clusters.get("clusters", [])):
                        cluster_info += f"\nCluster {i+1}: {cluster.get('name')}\n"
                        cluster_info += f"Papers: {', '.join([str(p) for p in cluster.get('papers', [])])}\n"
                        cluster_info += f"Description: {cluster.get('description')}\n"
                    
                    # Add cluster info to the output
                    cluster_summary = "\n\n" + cluster_info + "\n\n"
                except Exception as e:
                    cluster_summary = f"\n\nError generating clusters: {str(e)}\n\n"
            else:
                cluster_summary = ""
                
            # Add specialized mechanistic interpretability analysis if requested
            if mechanistic_interpretability and len(relevancy) > 0:
                # Use the first available provider in order of preference
                preferred_providers = [
                    (ModelProvider.ANTHROPIC, anthropic_model if use_anthropic else None),
                    (ModelProvider.OPENAI, openai_model if use_openai else None),
                    (ModelProvider.GEMINI, gemini_model if use_gemini else None)
                ]
                
                provider = None
                model = None
                for p, m in preferred_providers:
                    if model_manager.is_provider_available(p) and m:
                        provider = p
                        model = m
                        break
                        
                if provider:
                    try:
                        interp_analysis = model_manager.get_mechanistic_interpretability_analysis(
                            relevancy[0],  # Analyze the most relevant paper
                            provider=provider,
                            model_name=model
                        )
                        
                        interp_summary = "\n\n=== MECHANISTIC INTERPRETABILITY ANALYSIS ===\n"
                        for key, value in interp_analysis.items():
                            if key != "error" and key != "raw_content":
                                interp_summary += f"\n{key}: {value}\n"
                        
                        # Add interpretability analysis to the output
                        interpretability_info = "\n\n" + interp_summary + "\n\n"
                    except Exception as e:
                        interpretability_info = f"\n\nError generating interpretability analysis: {str(e)}\n\n"
                else:
                    interpretability_info = "\n\nNo available provider for interpretability analysis.\n\n"
            else:
                interpretability_info = ""
                
            # Generate HTML report
            html_file = generate_html_report(relevancy, title=f"ArXiv Digest: {topic} papers")
            
            # Create summary texts for display
            summary_texts = []
            for paper in relevancy:
                if "summarized_text" in paper:
                    summary_texts.append(paper["summarized_text"])
                else:
                    # Create a summary if summarized_text doesn't exist
                    summary = f"Title: {paper.get('title', 'No title')}\n"
                    summary += f"Authors: {paper.get('authors', 'Unknown')}\n"
                    summary += f"Score: {paper.get('Relevancy score', 'N/A')}\n"
                    summary += f"Abstract: {paper.get('abstract', 'No abstract')[:200]}...\n"
                    summary_texts.append(summary)
                    
            result_text = cluster_summary + "\n\n".join(summary_texts) + interpretability_info
            return result_text + f"\n\nHTML report saved to: {html_file}"
        else:
            # Generate HTML report
            html_file = generate_html_report(relevancy, title=f"ArXiv Digest: {topic} papers")
            
            # Create summary texts for display
            summary_texts = []
            for paper in relevancy:
                if "summarized_text" in paper:
                    summary_texts.append(paper["summarized_text"])
                else:
                    # Create a summary if summarized_text doesn't exist
                    summary = f"Title: {paper.get('title', 'No title')}\n"
                    summary += f"Authors: {paper.get('authors', 'Unknown')}\n" 
                    summary += f"Score: {paper.get('Relevancy score', 'N/A')}\n"
                    summary += f"Abstract: {paper.get('abstract', 'No abstract')[:200]}...\n"
                    summary_texts.append(summary)
                    
            result_text = "\n\n".join(summary_texts)
            return result_text + f"\n\nHTML report saved to: {html_file}"
    else:
        # Generate HTML report for basic results
        html_file = generate_html_report(papers, title=f"ArXiv Digest: {topic} papers")
        result_text = "\n\n".join(f"Title: {paper['title']}\nAuthors: {paper['authors']}" for paper in papers)
        return result_text + f"\n\nHTML report saved to: {html_file}"


def change_subsubject(subject, physics_subject):
    if subject != "Physics":
        return gr.Dropdown.update(choices=categories_map[subject], value=[], visible=True)
    else:
        if physics_subject and not isinstance(physics_subject, list):
            return gr.Dropdown.update(choices=categories_map[physics_subject], value=[], visible=True)
        else:
            return gr.Dropdown.update(choices=[], value=[], visible=False)


def change_physics(subject):
    if subject != "Physics":
        return gr.Dropdown.update(visible=False, value=None)
    else:
        return gr.Dropdown.update(choices=list(physics_topics.keys()), visible=True)


def test(email, topic, physics_topic, categories, interest, key, 
         use_openai, use_gemini, use_anthropic, openai_model, gemini_model, anthropic_model,
         special_analysis, mechanistic_interpretability, technical_ai_safety,
         design_automation, design_reference_paper, design_techniques, design_categories):
    if not email: raise gr.Error("Set your email")
    if not key: raise gr.Error("Set your SendGrid key")
    if topic == "Physics":
        if isinstance(physics_topic, list):
            raise gr.Error("You must choose a physics topic.")
        topic = physics_topic
        abbr = physics_topics[topic]
    else:
        abbr = topics[topic]
        
    # Check if at least one model is selected
    if not (use_openai or use_gemini or use_anthropic):
        raise gr.Error("You must select at least one model provider (OpenAI, Gemini, or Claude)")
        
    if categories:
        papers = get_papers(abbr)
        papers = [
            t for t in papers
            if bool(set(process_subject_fields(t['subjects'])) & set(categories))][:4]
    else:
        papers = get_papers(abbr, limit=4)
        
    if interest:
        # Build list of providers to use
        providers = []
        model_names = {}
        
        if use_openai:
            if not model_manager.is_provider_available(ModelProvider.OPENAI):
                if not openai.api_key:
                    raise gr.Error("Set your OpenAI API key in the OpenAI tab first")
                else:
                    model_manager.register_openai(openai.api_key)
            providers.append(ModelProvider.OPENAI)
            model_names[ModelProvider.OPENAI] = openai_model
            
        if use_gemini:
            if not model_manager.is_provider_available(ModelProvider.GEMINI):
                raise gr.Error("Set your Gemini API key in the Gemini tab first")
            providers.append(ModelProvider.GEMINI)
            model_names[ModelProvider.GEMINI] = gemini_model
            
        if use_anthropic:
            if not model_manager.is_provider_available(ModelProvider.ANTHROPIC):
                raise gr.Error("Set your Anthropic API key in the Anthropic tab first")
            providers.append(ModelProvider.ANTHROPIC)
            model_names[ModelProvider.ANTHROPIC] = anthropic_model
            
        # Check if we need to find design automation papers
        if design_automation:
            # Filter for design automation papers
            design_papers = [p for p in papers if is_design_automation_paper(p)]
            
            # Filter by techniques if specified
            if design_techniques:
                filtered_papers = []
                for paper in design_papers:
                    paper_techniques = analyze_design_techniques(paper)
                    if any(technique in design_techniques for technique in paper_techniques):
                        filtered_papers.append(paper)
                design_papers = filtered_papers if filtered_papers else design_papers
                
            # Filter by categories if specified
            if design_categories:
                filtered_papers = []
                for paper in design_papers:
                    paper_category = categorize_design_paper(paper)
                    if any(category in paper_category for category in design_categories):
                        filtered_papers.append(paper)
                design_papers = filtered_papers if filtered_papers else design_papers
                
            # Find related papers if reference paper is specified
            if design_reference_paper:
                related_papers = get_related_design_papers(design_reference_paper, papers)
                if related_papers:
                    design_papers = related_papers
            
            # Use these papers if we found any, otherwise fallback to regular papers
            if design_papers:
                papers = design_papers
        
        # Process papers directly instead of using model_manager
        print("\n===== ANALYZING PAPERS =====")
        print(f"Processing {len(papers)} papers...")
        relevancy = []
        hallucination = False
        
        # Use OpenAI if selected
        if use_openai and model_manager.is_provider_available(ModelProvider.OPENAI):
            try:
                # Import directly to avoid circular imports
                from relevancy import generate_relevance_score
                openai_results, hallu = generate_relevance_score(
                    papers,
                    query={"interest": interest},
                    model_name=openai_model,
                    threshold_score=0,  # We'll filter later
                    num_paper_in_prompt=2
                )
                hallucination = hallucination or hallu
                relevancy.extend(openai_results)
                print(f"OpenAI analysis added {len(openai_results)} papers")
            except Exception as e:
                print(f"Error during OpenAI analysis: {e}")
        
        # Use Gemini if selected and no papers yet
        if use_gemini and model_manager.is_provider_available(ModelProvider.GEMINI) and len(relevancy) == 0:
            try:
                # Import directly to avoid circular imports
                from gemini_utils import analyze_papers_with_gemini
                gemini_papers = analyze_papers_with_gemini(
                    papers,
                    query={"interest": interest},
                    model_name=gemini_model
                )
                # Process papers to ensure they have the right fields
                for paper in gemini_papers:
                    if 'gemini_analysis' in paper:
                        # Copy all fields from gemini_analysis to the paper object
                        for key, value in paper['gemini_analysis'].items():
                            paper[key] = value
                
                relevancy.extend(gemini_papers)
                print(f"Gemini analysis added {len(gemini_papers)} papers")
            except Exception as e:
                print(f"Error during Gemini analysis: {e}")
                
        # Use Anthropic if selected and no papers yet
        if use_anthropic and model_manager.is_provider_available(ModelProvider.ANTHROPIC) and len(relevancy) == 0:
            print("Anthropic/Claude analysis not yet implemented")
        
        print(f"Total papers after analysis: {len(relevancy)}")
        
        # Filter papers by threshold from config
        threshold = config.get("threshold", 2)
        relevancy = filter_papers_by_threshold(relevancy, threshold)
        
        # Add design automation information if requested
        if design_automation and relevancy:
            for paper in relevancy:
                paper["design_category"] = categorize_design_paper(paper)
                paper["design_techniques"] = analyze_design_techniques(paper)
                paper["design_metrics"] = extract_design_metrics(paper)
                
                # Perform detailed design automation analysis on highest scored papers
                if paper.get("Relevancy score", 0) >= 7 and (use_openai or use_gemini or use_anthropic):
                    # Select provider for design analysis
                    provider = None
                    model = None
                    
                    if use_openai and model_manager.is_provider_available(ModelProvider.OPENAI):
                        provider = ModelProvider.OPENAI
                        model = openai_model
                    elif use_gemini and model_manager.is_provider_available(ModelProvider.GEMINI):
                        provider = ModelProvider.GEMINI
                        model = gemini_model
                    elif use_anthropic and model_manager.is_provider_available(ModelProvider.ANTHROPIC):
                        provider = ModelProvider.ANTHROPIC
                        model = anthropic_model
                        
                    if provider:
                        design_analysis = model_manager.analyze_design_automation(
                            paper,
                            provider=provider,
                            model_name=model
                        )
                        if design_analysis and "error" not in design_analysis:
                            paper["design_analysis"] = design_analysis
        
        # Create the email body with better styling
        paper_bodies = []
        for paper in relevancy:
            paper_html = f'''
            <div style="margin-bottom: 30px; border-bottom: 1px solid #ddd; padding-bottom: 20px;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">
                    <a href="{paper["main_page"]}" style="color: #2980b9; text-decoration: none;">{paper["title"]}</a>
                </div>
                <div style="margin-bottom: 5px;"><b>Subject:</b> {paper["subjects"]}</div>
                <div style="margin-bottom: 5px;"><b>Authors:</b> {paper["authors"]}</div>
                <div style="margin-bottom: 10px; font-weight: bold; color: #e74c3c;">
                    <b>Score:</b> {paper["Relevancy score"]}
                </div>
                
                <div style="margin: 15px 0; background-color: #f5f9f9; padding: 15px; border-left: 4px solid #2ecc71;">
                    <b>Reason for Relevance:</b> {paper["Reasons for match"]}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #fdf5e6;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Key Innovations:</div>
                    {paper.get("Key innovations", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #fdf5e6;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Critical Analysis:</div>
                    {paper.get("Critical analysis", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #fdf5e6;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Goal:</div>
                    {paper.get("Goal", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #f0f8ff;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Data:</div>
                    {paper.get("Data", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #f0f8ff;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Methodology:</div>
                    {paper.get("Methodology", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #f0f8ff;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Implementation Details:</div>
                    {paper.get("Implementation details", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #f5f5f5;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Experiments & Results:</div>
                    {paper.get("Experiments & Results", "Not provided")}
                </div>
                
                <div style="margin-bottom: 5px;"><b>Git:</b> {paper.get("Git", "Not provided")}</div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #f0fff0;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Discussion & Next Steps:</div>
                    {paper.get("Discussion & Next steps", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #f0fff0;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Related Work:</div>
                    {paper.get("Related work", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #f0fff0;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Practical Applications:</div>
                    {paper.get("Practical applications", "Not provided")}
                </div>
                
                <div style="margin: 15px 0; padding: 15px; background-color: #fdf5e6;">
                    <div style="font-weight: bold; margin-bottom: 5px;">Key Takeaways:</div>
                    {paper.get("Key takeaways", "Not provided")}
                </div>
            '''
            
            # Add design automation section if available
            if design_automation and (paper.get("design_category") or paper.get("design_techniques")):
                paper_html += f'''
                <div style="margin-top: 20px; border-top: 1px dashed #ccc; padding-top: 15px;">
                    <h3>Design Automation Analysis</h3>
                    <div style="margin-bottom: 5px;"><b>Design Category:</b> {paper.get("design_category", "")}</div>
                    <div style="margin-bottom: 5px;"><b>Design Techniques:</b> {", ".join(paper.get("design_techniques", []))}</div>
                    <div style="margin-bottom: 5px;"><b>Design Metrics:</b> {", ".join(paper.get("design_metrics", []))}</div>
                '''
                
                if "design_analysis" in paper:
                    paper_html += f'''
                    <h4>Detailed Design Analysis</h4>
                    <div style="margin-bottom: 5px;"><b>Design automation focus:</b> {paper.get("design_analysis", {}).get("Design automation focus", "Not provided")}</div>
                    <div style="margin-bottom: 5px;"><b>Technical approach:</b> {paper.get("design_analysis", {}).get("Technical approach", "Not provided")}</div>
                    <div style="margin-bottom: 5px;"><b>Visual outputs:</b> {paper.get("design_analysis", {}).get("Visual outputs", "Not provided")}</div>
                    <div style="margin-bottom: 5px;"><b>Designer interaction:</b> {paper.get("design_analysis", {}).get("Designer interaction", "Not provided")}</div>
                    <div style="margin-bottom: 5px;"><b>Real-world applicability:</b> {paper.get("design_analysis", {}).get("Real-world applicability", "Not provided")}</div>
                    <div style="margin-bottom: 5px;"><b>Capabilities:</b> Replaceable tools: {", ".join(paper.get("design_analysis", {}).get("capabilities", {}).get("replaceable_tools", []))}, 
                    Automation level: {paper.get("design_analysis", {}).get("capabilities", {}).get("automation_level", "Unknown")}</div>
                    '''
                
                paper_html += '</div>'
            
            paper_html += '''
                <div style="margin-top: 20px;">
                    <a href="{}" style="color: #2980b9; text-decoration: none; margin-right: 15px;">PDF</a>
                    <a href="{}" style="color: #2980b9; text-decoration: none;">arXiv</a>
                </div>
            </div>
            '''.format(paper.get("pdf", paper.get("main_page", "#") + ".pdf"), paper.get("main_page", "#"))
            
            paper_bodies.append(paper_html)
        
        body = '''
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; }
                a { color: #2980b9; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>ArXiv Digest Results</h1>
        '''
        
        # Add warning for hallucinations
        if hallucination:
            body += '<p style="color: #e74c3c; font-weight: bold;">Warning: The model hallucinated some papers. We have tried to remove them, but the scores may not be accurate.</p>'
        
        # Add papers
        body += ''.join(paper_bodies)
        
        body += '''
        </body>
        </html>
        '''
            
        # Add specialized analysis if requested
        if special_analysis and len(relevancy) > 0:
            # Get topic clustering from Gemini if available
            if use_gemini and model_manager.is_provider_available(ModelProvider.GEMINI):
                try:
                    clusters = get_topic_clustering(relevancy, model_name=gemini_model)
                    cluster_info = "<h2>Topic Clusters</h2>"
                    for i, cluster in enumerate(clusters.get("clusters", [])):
                        cluster_info += f"<h3>Cluster {i+1}: {cluster.get('name')}</h3>"
                        cluster_info += f"<p><b>Papers:</b> {', '.join([str(p) for p in cluster.get('papers', [])])}</p>"
                        cluster_info += f"<p><b>Description:</b> {cluster.get('description')}</p>"
                    
                    # Add cluster info to the body
                    body = cluster_info + "<hr>" + body
                except Exception as e:
                    body = f"<p><i>Error generating clusters: {str(e)}</i></p>" + body
            
            # Add specialized mechanistic interpretability analysis if requested
            if mechanistic_interpretability and len(relevancy) > 0:
                # Use the first available provider in order of preference
                preferred_providers = [
                    (ModelProvider.ANTHROPIC, anthropic_model if use_anthropic else None),
                    (ModelProvider.OPENAI, openai_model if use_openai else None),
                    (ModelProvider.GEMINI, gemini_model if use_gemini else None)
                ]
                
                provider = None
                model = None
                for p, m in preferred_providers:
                    if model_manager.is_provider_available(p) and m:
                        provider = p
                        model = m
                        break
                        
                if provider:
                    try:
                        interp_analysis = model_manager.get_mechanistic_interpretability_analysis(
                            relevancy[0],  # Analyze the most relevant paper
                            provider=provider,
                            model_name=model
                        )
                        
                        interp_summary = "<h2>Mechanistic Interpretability Analysis</h2>"
                        interp_summary += f"<h3>Analysis for paper: {relevancy[0]['title']}</h3>"
                        
                        for key, value in interp_analysis.items():
                            if key != "error" and key != "raw_content":
                                interp_summary += f"<p><b>{key}:</b> {value}</p>"
                        
                        # Add interpretability analysis to the body
                        body = interp_summary + "<hr>" + body
                    except Exception as e:
                        body = f"<p><i>Error generating interpretability analysis: {str(e)}</i></p>" + body
        
        if hallucination:
            body = "<p><strong style='color:red;'>Warning: The model hallucinated some papers. We have tried to remove them, but the scores may not be accurate.</strong></p><br>" + body
    else:
        body = "<br><br>".join([f'Title: <a href="{paper["main_page"]}">{paper["title"]}</a><br>Authors: {paper["authors"]}' for paper in papers])
    
    # Send email
    sg = sendgrid.SendGridAPIClient(api_key=key)
    from_email = Email(email)
    to_email = To(email)
    subject = "arXiv digest"
    content = Content("text/html", body)
    mail = Mail(from_email, to_email, subject, content)
    mail_json = mail.get()

    # Generate HTML report file
    html_file = generate_html_report(relevancy if interest else papers, 
                                   title=f"ArXiv Digest: {topic} papers")
    
    # Send an HTTP POST request to /mail/send
    response = sg.client.mail.send.post(request_body=mail_json)
    if response.status_code >= 200 and response.status_code <= 300:
        return f"Success! Email sent and HTML report saved to: {html_file}"
    else:
        return f"Email sending failed ({response.status_code}), but HTML report saved to: {html_file}"


def register_openai_token(token):
    openai.api_key = token
    model_manager.register_openai(token)
    
def register_gemini_token(token):
    setup_gemini_api(token)
    model_manager.register_gemini(token)
    
def register_anthropic_token(token):
    model_manager.register_anthropic(token)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Tabs():
                with gr.TabItem("OpenAI"):
                    openai_token = gr.Textbox(label="OpenAI API Key", type="password")
                    openai_token.change(fn=register_openai_token, inputs=[openai_token])
                
                with gr.TabItem("Gemini"):
                    gemini_token = gr.Textbox(label="Gemini API Key", type="password")
                    gemini_token.change(fn=register_gemini_token, inputs=[gemini_token])
                
                with gr.TabItem("Anthropic"):
                    anthropic_token = gr.Textbox(label="Anthropic API Key", type="password")
                    anthropic_token.change(fn=register_anthropic_token, inputs=[anthropic_token])
            
            subject = gr.Radio(
                list(topics.keys()), label="Topic"
            )
            physics_subject = gr.Dropdown(list(physics_topics.keys()), value=None, multiselect=False, label="Physics category", visible=False, info="")
            subsubject = gr.Dropdown(
                    [], value=[], multiselect=True, label="Subtopic", info="Optional. Leaving it empty will use all subtopics.", visible=False)
            subject.change(fn=change_physics, inputs=[subject], outputs=physics_subject)
            subject.change(fn=change_subsubject, inputs=[subject, physics_subject], outputs=subsubject)
            physics_subject.change(fn=change_subsubject, inputs=[subject, physics_subject], outputs=subsubject)

            interest = gr.Textbox(label="A natural language description of what you are interested in. We will generate relevancy scores (1-10) and explanations for the papers in the selected topics according to this statement.", info="Press shift-enter or click the button below to update.", lines=7)
            
            with gr.Row():
                use_openai = gr.Checkbox(label="Use OpenAI", value=True)
                use_gemini = gr.Checkbox(label="Use Gemini", value=False)
                use_anthropic = gr.Checkbox(label="Use Claude", value=False)
            
            with gr.Accordion("Advanced Settings", open=False):
                openai_model = gr.Dropdown(["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"], value="gpt-4", label="OpenAI Model")
                gemini_model = gr.Dropdown(["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"], value="gemini-2.0-flash", label="Gemini Model")
                anthropic_model = gr.Dropdown(["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"], value="claude-3-sonnet-20240229", label="Claude Model")
                
                special_analysis = gr.Checkbox(label="Include specialized analysis for research topics", value=False)
                mechanistic_interpretability = gr.Checkbox(label="Include mechanistic interpretability analysis", value=False)
                technical_ai_safety = gr.Checkbox(label="Include technical AI safety analysis", value=False)
                
                with gr.Accordion("Graphic Design Automation Papers", open=False):
                    design_automation = gr.Checkbox(label="Find graphic design automation papers", value=False)
                    design_reference_paper = gr.Textbox(
                        label="Reference paper ID (optional, e.g., '2412.04237' from VASCAR paper)",
                        placeholder="Enter arXiv paper ID to find similar papers"
                    )
                    design_techniques = gr.CheckboxGroup(
                        choices=[
                            "Generative Adversarial Networks", "Diffusion Models", 
                            "Transformers", "Large Language Models", "Computer Vision",
                            "Neural Style Transfer", "Reinforcement Learning"
                        ],
                        label="Design automation techniques to focus on (optional)"
                    )
                    design_categories = gr.CheckboxGroup(
                        choices=[
                            "Layout Generation", "UI/UX Design", "Graphic Design",
                            "Image Manipulation", "Design Tools", "3D Design",
                            "Multimodal Design"
                        ],
                        label="Design categories to focus on (optional)"
                    )
                
            sample_btn = gr.Button("Generate Digest")
            sample_output = gr.Textbox(label="Results for your configuration.", info="For runtime purposes, this is only done on a small subset of recent papers in the topic you have selected. Papers will not be filtered by relevancy, only sorted on a scale of 1-10.")
        with gr.Column(scale=4):  # Changed from 0.40 to 4
            with gr.Group():  # Changed from gr.Box to gr.Group
                title = gr.Markdown(
                    """
                    # Email Setup, Optional
                    Send an email to the below address using the configuration on the right. Requires a sendgrid token. These values are not needed to use the right side of this page.

                    To create a scheduled job for this, see our [Github Repository](https://github.com/AutoLLM/ArxivDigest)
                    """)
                email = gr.Textbox(label="Email address", type="email", placeholder="")
                sendgrid_token = gr.Textbox(label="SendGrid API Key", type="password")
                with gr.Row():
                    test_btn = gr.Button("Send email")
                    output = gr.Textbox(show_label=False, placeholder="email status")
    # Define all input fields
    all_inputs = [
        email, subject, physics_subject, subsubject, interest, 
        use_openai, use_gemini, use_anthropic,
        openai_model, gemini_model, anthropic_model,
        special_analysis, mechanistic_interpretability, technical_ai_safety,
        design_automation, design_reference_paper, design_techniques, design_categories
    ]
    
    # Email button
    test_btn.click(
        fn=test, 
        inputs=[email, subject, physics_subject, subsubject, interest, sendgrid_token] + all_inputs[5:],
        outputs=output
    )
    
    # Sample button
    sample_btn.click(
        fn=sample, 
        inputs=all_inputs,
        outputs=sample_output
    )
    
    # Register API keys
    openai_token.change(fn=register_openai_token, inputs=[openai_token])
    gemini_token.change(fn=register_gemini_token, inputs=[gemini_token])
    anthropic_token.change(fn=register_anthropic_token, inputs=[anthropic_token])
    
    # Dynamic updates based on selection changes
    subject.change(fn=sample, inputs=all_inputs, outputs=sample_output)
    physics_subject.change(fn=sample, inputs=all_inputs, outputs=sample_output)
    subsubject.change(fn=sample, inputs=all_inputs, outputs=sample_output)
    interest.submit(fn=sample, inputs=all_inputs, outputs=sample_output)

demo.launch(show_api=False)
