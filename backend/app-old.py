from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from fastcoref import FCoref
from typing import List, Dict, Tuple
import spacy
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load models once at startup - using the same models as Jupyter version
print("Loading models...")
sentModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # Same as Jupyter
corefmodel = FCoref(device='cpu')
nlp = spacy.load("en_core_web_sm")
print("Models loaded successfully!")

# === EXACT COPIES FROM JUPYTER VERSION ===

def replace_mentions(text: str) -> str:
    """
    Resolves pronouns in text using FastCoref and replaces them with their referents.
    Returns the resolved text.
    """
    # FastCoref requires list input
    predictions = corefmodel.predict([text])
    prediction = predictions[0]  # Get first (only) prediction

    print(f"\n📌 Original: {text}")

    # Get the actual span clusters - this is the key fix!
    # Use get_clusters(as_strings=False) to get span coordinates
    try:
        # Try different ways to access clusters based on FastCoref version
        if hasattr(prediction, 'get_clusters'):
            clusters = prediction.get_clusters(as_strings=False)
        elif hasattr(prediction, 'clusters'):
            clusters = prediction.clusters
        else:
            # Fallback - access clusters directly from prediction attributes
            clusters = getattr(prediction, '_clusters', [])
    except Exception as e:
        print(f"Error accessing clusters: {e}")
        return text

    # For debugging - print raw clusters
    print(f"🔍 Raw clusters: {clusters}")

    # Build replacement map: {(start, end) -> replacement_text}
    replacements: Dict[Tuple[int, int], str] = {}

    for cluster in clusters:
        if not cluster or len(cluster) < 2:  # Need at least 2 mentions for coreference
            continue

        print(f"🔗 Processing cluster: {cluster}")

        # Find the best referent (usually the first non-pronoun mention)
        main_referent = None
        main_text = ""

        for span in cluster:
            if not isinstance(span, (tuple, list)) or len(span) != 2:
                continue

            start, end = span
            span_text = text[start:end]

            # Skip pronouns for main referent selection
            if span_text.lower() not in {'it', 'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their'}:
                main_referent = span
                main_text = span_text
                break

        # If no non-pronoun referent found, use the first mention
        if main_referent is None and cluster:
            main_referent = cluster[0]
            main_text = text[main_referent[0]:main_referent[1]]

        print(f"   📍 Main referent: '{main_text}' at {main_referent}")

        # Replace pronouns in this cluster
        for span in cluster:
            if not isinstance(span, (tuple, list)) or len(span) != 2:
                continue

            start, end = span
            mention_text = text[start:end]

            # Only replace pronouns, and don't replace the main referent with itself
            if (mention_text.lower() in {'it', 'he', 'she', 'they', 'him', 'her', 'them'}
                and span != main_referent):
                replacements[(start, end)] = main_text
                print(f"   🔄 Will replace '{mention_text}' ({start}-{end}) with '{main_text}'")

    # Apply replacements in reverse order (to preserve positions)
    resolved_text = text
    for (start, end) in sorted(replacements.keys(), reverse=True):
        resolved_text = resolved_text[:start] + replacements[(start, end)] + resolved_text[end:]

    print(f"✅ Resolved: {resolved_text}")
    return resolved_text

def split_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def should_merge(s1, s2, soft_conj=("and", "but", "because"), embed_sim_threshold=0.35):
    if not s2.lower().startswith(tuple(w + ' ' for w in soft_conj)):
        return False
    e1 = sentModel.encode(s1, convert_to_tensor=True)
    e2 = sentModel.encode(s2, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(e1, e2).item()
    return sim >= embed_sim_threshold

def smart_merge_conj_embed(sentences, soft_conj=("and", "but", "because"), embed_sim_threshold=0.35):
    merged = []
    buffer = sentences[0]
    for i in range(1, len(sentences)):
        curr = sentences[i]
        if should_merge(buffer, curr, soft_conj, embed_sim_threshold):
            buffer += " " + curr
        else:
            merged.append(buffer.strip())
            buffer = curr
    merged.append(buffer.strip())
    return merged

def process_text(text):
    """Main preprocessing pipeline - exactly as in Jupyter version"""
    sentences = split_sentences(text)
    # Filter out short sentences (< 3 words)
    # sentences = [s for s in sentences if len(s.split()) >= 3]
    merged_sentences = smart_merge_conj_embed(sentences)
    return [s for s in merged_sentences if len(s.split()) >= 3] 

def compute_staleness(groups: List[List[int]], current_index: int, window: int) -> float:
    stale_count = 0
    for group in groups:
        last_active = max(group)
        if current_index - last_active > window:
            stale_count += 1
    return stale_count / len(groups) if groups else 0

def compute_saturation(groups: List[List[int]], current_index: int) -> float:
    if current_index < 10 or not groups:
        return 0.0
    lookback = min(20, current_index//2)  # Adaptive lookback window
    new_groups = sum(1 for group in groups if min(group) >= current_index - lookback)
    return min(1.0, new_groups / len(groups))

def estimate_speech_length(groups: List[List[int]], current_index: int) -> int:
    if current_index < 15:
        return 50
    topic_rate = len(groups) / current_index
    estimated_length = int(3.8 / max(topic_rate, 0.01))  # Prevents division by zero
    return min(400, max(40, estimated_length))

def calculate_progress(staleness: float, saturation: float, pos: int, est_length: int) -> float:
    staleness_weight = 0.3 + 0.5 * (pos/est_length) ** 2
    freshness_boost = min(0.3, 0.6 * saturation)
    effective_staleness = staleness * (1 - freshness_boost)
    combined = (staleness_weight * effective_staleness +
               (1-staleness_weight) * (1-saturation))

    # Constrain by linear position and smooth
    linear_progress = pos / est_length
    progress = 0.1 * linear_progress + 0.9 * combined
    return min(99, max(1, progress * 100))

def linear_grouping_quantified(
    sentences: List[str],
    model,
    initial_threshold: float = 0.65,
    min_threshold: float = 0.4
):
    # Initialize data structures - EXACTLY as Jupyter version
    embeddings, groups, group_embeddings = [], [], []
    threshold_history, progress_history = [], []
    staleness_history, saturation_history = [], []
    # NOTE: position_history is NOT in original Jupyter version

    # Adaptive parameters
    current_window = 5
    predicted_length = 50
    smoothed_progress = 0

    for i, sentence in enumerate(sentences):
        # Process current sentence
        new_embedding = model.encode(sentence, convert_to_tensor=True)
        embeddings.append(new_embedding)

        if i == 0:
            groups.append([0])
            group_embeddings.append(new_embedding)
            threshold_history.append(initial_threshold)
            continue

        # Adaptive threshold decay
        current_threshold = max(
            min_threshold,
            threshold_history[-1] * (1 - 0.01 * (1 - smoothed_progress/100))
        )

        # Grouping logic
        best_sim, best_group = -1, None
        for group_idx, group_embed in enumerate(group_embeddings):
            sim = util.pytorch_cos_sim(new_embedding, group_embed).item()
            if sim > best_sim:
                best_sim, best_group = sim, group_idx

        if best_sim >= current_threshold:
            groups[best_group].append(i)
            group_embeddings[best_group] = torch.mean(
                torch.stack([embeddings[idx] for idx in groups[best_group]]),
                dim=0
            )
        else:
            groups.append([i])
            group_embeddings.append(new_embedding)

        threshold_history.append(current_threshold)

        if i % current_window == 0:
            predicted_length = estimate_speech_length(groups, i)
            current_window = max(3, min(10, int(0.1 * predicted_length)))

            staleness = compute_staleness(groups, i, current_window)
            saturation = compute_saturation(groups, i)

            current_progress = calculate_progress(
                staleness=staleness,
                saturation=saturation,
                pos=i,
                est_length=predicted_length
            )
            smoothed_progress = 0.9 * smoothed_progress + 0.1 * current_progress

            staleness_history.append(staleness)
            saturation_history.append(saturation)
            progress_history.append(smoothed_progress)
            # NOTE: Original Jupyter version does NOT have position_history.append(i)

            print(f"Pos {i:3d}: Progress={smoothed_progress:5.1f}% | "
                  f"Staleness={staleness:.2f} | "
                  f"Saturation={saturation:.2f} | "
                  f"EstLen={predicted_length}")

    # IMPORTANT: Original Jupyter version does NOT return anything!
    # It just prints and shows plots. We need to return data for API but keep logic identical.
    
    # For API purposes, we'll return the final state but ensure the computation is identical
    return {
        'groups': groups,
        'progress_history': progress_history,
        'threshold_history': threshold_history,
        'staleness_history': staleness_history,
        'saturation_history': saturation_history,
        'sentences': sentences,
        'predicted_length': predicted_length,
        'smoothed_progress': smoothed_progress
    }

# === API ENDPOINTS ===

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text = data.get('text', '')
        use_coref_resolution = data.get('use_coref_resolution', True)
        initial_threshold = data.get('initial_threshold', 0.65)
        min_threshold = data.get('min_threshold', 0.4)
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Apply the EXACT same preprocessing pipeline as Jupyter version
        print("=== Starting Text Processing ===")
        
        # Step 1: Coreference resolution (optional)
        if use_coref_resolution:
            print("Applying coreference resolution...")
            resolved_text = replace_mentions(text)
        else:
            resolved_text = text
        
        # Step 2: Process text (sentence splitting + smart merging)
        print("Processing text (splitting and merging)...")
        sentences = process_text(resolved_text)
        
        print(f"Final processed sentences ({len(sentences)}):")
        for i, s in enumerate(sentences):
            print(f"[{i}] {s}")
        
        if len(sentences) < 2:
            return jsonify({'error': 'Need at least 2 text segments for analysis after preprocessing'}), 400
        
        # Step 3: Run the EXACT same analysis as Jupyter version
        print("Running linear_grouping_quantified analysis...")
        results = linear_grouping_quantified(
            sentences,
            sentModel,
            initial_threshold=initial_threshold,
            min_threshold=min_threshold
        )
        
        # Format results for frontend (this is the only difference from Jupyter)
        formatted_groups = []
        for i, group in enumerate(results['groups']):
            formatted_groups.append({
                'id': i + 1,
                'sentences': [sentences[idx] for idx in group]
            })
        
        # Create progress data for visualization
        # Since original doesn't have position_history, we reconstruct it
        progress_data = []
        window_size = 5  # This matches the original current_window initial value
        for i, progress in enumerate(results['progress_history']):
            position = i * window_size  # Reconstruct the positions
            progress_data.append({
                'position': position,
                'progress': progress,
                'text': f"Position {position}"
            })
        
        response_data = {
            'final_progress': f"{results['smoothed_progress']:.1f}%",
            'estimated_length': results['predicted_length'],
            'total_segments': len(sentences),
            'total_groups': len(results['groups']),  
            'groups': formatted_groups,
            'progress_data': progress_data,
            'progress_history': results['progress_history'],
            'threshold_history': results['threshold_history'],
            'staleness_history': results['staleness_history'],
            'saturation_history': results['saturation_history'],
            'processed_text': resolved_text,
            'original_sentence_count': len(split_sentences(text)),
            'final_sentence_count': len(sentences)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-simple', methods=['POST'])
def analyze_text_simple():
    """Legacy endpoint without preprocessing for backwards compatibility"""
    try:
        data = request.json
        text = data.get('text', '')
        split_method = data.get('split_method', 'sentences')
        custom_delimiter = data.get('custom_delimiter', ',')
        initial_threshold = data.get('initial_threshold', 0.65)
        min_threshold = data.get('min_threshold', 0.4)
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Simple text splitting (old method)
        if split_method == "sentences":
            segments = re.split(r'[.!?]+', text)
        elif split_method == "lines":
            segments = text.split('\n')
        elif split_method == "paragraphs":
            segments = text.split('\n\n')
        else:  # custom
            segments = text.split(custom_delimiter)
        
        sentences = [s.strip() for s in segments if s.strip()]
        
        if len(sentences) < 2:
            return jsonify({'error': 'Need at least 2 text segments for analysis'}), 400
        
        # Run analysis
        results = linear_grouping_quantified(
            sentences,
            sentModel,
            initial_threshold=initial_threshold,
            min_threshold=min_threshold
        )
        
        # Format results
        formatted_groups = []
        for i, group in enumerate(results['groups']):
            formatted_groups.append({
                'id': i + 1,
                'sentences': [sentences[idx] for idx in group]
            })
        
        progress_data = []
        window_size = 5
        for i, progress in enumerate(results['progress_history']):
            position = i * window_size
            if position < len(sentences):
                progress_data.append({
                    'position': position,
                    'progress': f"{progress:.1f}%",
                    'text': sentences[position][:100] + "..." if len(sentences[position]) > 100 else sentences[position]
                })
        
        response_data = {
            'final_progress': f"{results['smoothed_progress']:.1f}%",
            'estimated_length': results['predicted_length'],
            'total_segments': len(sentences),
            'total_groups': len(results['groups']),
            'groups': formatted_groups,
            'progress_data': progress_data,
            'progress_history': results['progress_history'],
            'threshold_history': results['threshold_history'],
            'staleness_history': results['staleness_history'],
            'saturation_history': results['saturation_history']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': sentModel is not None,
        'coref_model_loaded': corefmodel is not None,
        'spacy_loaded': nlp is not None
    })

if __name__ == '__main__':
    print("Starting Speech Analysis Backend...")
    app.run(debug=True, host='0.0.0.0', port=5000)