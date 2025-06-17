from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from fastcoref import FCoref
from typing import List, Dict, Tuple
import spacy
import torch
import numpy as np
from collections import deque
import io
import sys

app = Flask(__name__)
CORS(app)

# Initialize models (same as in your notebook)
nlp = spacy.load("en_core_web_sm")
sentModel = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
corefmodel = FCoref(device='cpu')

def replace_mentions(text: str) -> str:
    """
    Resolves pronouns in text using FastCoref and replaces them with their referents.
    Returns the resolved text.
    """
    # FastCoref requires list input
    predictions = corefmodel.predict([text])
    prediction = predictions[0]  # Get first (only) prediction

    print(f"\n📌 Original: {text}")

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

# Merge adjacent related sentences
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
    sentences = split_sentences(text)
    sentences = [s for s in sentences if len(s.split()) >= 3]
    merged_sentences = smart_merge_conj_embed(sentences)
    return merged_sentences

def compute_trend(history, window=5):
    if len(history) < window:
        return 0.0
    x = np.arange(window)
    y = np.array(list(history)[-window:])
    return np.polyfit(x, y, 1)[0]

def enhanced_staleness(groups, current_index, history=deque(maxlen=10)):
    base = sum(1 for g in groups if current_index - max(g) > 15) / len(groups) if groups else 0
    age_factor = sum(current_index - min(g) for g in groups) / (100 * len(groups)) if groups else 0
    history.append(base)
    trend = compute_trend(history)
    return min(1.0, base * (1.2 + age_factor) + 0.3 * trend)

def enhanced_saturation(groups, current_index, history=deque(maxlen=10)):
    if current_index < 10 or not groups:
        return 0.0

    # New: Calculate population balance across topics
    group_sizes = [len(g) for g in groups]
    avg_size = np.mean(group_sizes)
    size_deviation = np.std(group_sizes) / avg_size if avg_size > 0 else 0

    # New: Calculate topic vitality (recent additions)
    vital_topics = 0
    for g in groups:
        recent_additions = sum(1 for sent_idx in g if current_index - sent_idx < 8)
        if recent_additions > 0:
            vital_topics += 1

    lookback = min(20, current_index//2)
    base = sum(1 for g in groups if min(g) >= current_index - lookback) / len(groups)
    depth = np.mean([min(1, len(g)/8) for g in groups if max(g) >= current_index - 5]) if any(g[-1] >= current_index - 5 for g in groups) else 0

    population_factor = min(1.0, size_deviation * 2)
    vitality_factor = 1 - (vital_topics / len(groups)) if groups else 0

    saturation = min(1.0,
        0.3*base + 0.3*depth + 0.2*population_factor + 0.2*vitality_factor
    )

    history.append(saturation)
    trend = compute_trend(history)
    return min(1.0, saturation + 0.1*trend)

def estimate_speech_length(groups, current_index):
    if current_index < 15:
        return 50
    topic_rate = len(groups) / current_index
    return min(400, max(40, int(3.8 / max(topic_rate, 0.01))))

def calculate_progress(staleness, saturation, pos, est_length):
    time_weight = min(1.0, pos / est_length)
    staleness_weight = 0.4 + 0.3 * time_weight
    saturation_weight = 0.6 - 0.3 * time_weight
    combined = (staleness_weight * staleness + saturation_weight * saturation)
    progress = 100 * (1 - np.exp(-2.5 * combined * (pos/est_length)))
    return min(99, max(1, progress))

def linear_grouping_quantified(sentences, model, initial_threshold=0.65, min_threshold=0.4):
    embeddings, groups, group_embeddings = [], [], []
    threshold_history, progress_history = [], []
    staleness_history, saturation_history = deque(maxlen=10), deque(maxlen=10)
    current_window, predicted_length, smoothed_progress = 5, 50, 0
    progress_data = []

    for i, sentence in enumerate(sentences):
        new_embedding = model.encode(sentence, convert_to_tensor=True)
        embeddings.append(new_embedding)

        if i == 0:
            groups.append([0])
            group_embeddings.append(new_embedding)
            threshold_history.append(initial_threshold)
            continue

        current_threshold = max(min_threshold, threshold_history[-1] * (1 - 0.01 * (1 - smoothed_progress/100)))

        best_sim, best_group = -1, None
        for group_idx, group_embed in enumerate(group_embeddings):
            sim = util.pytorch_cos_sim(new_embedding, group_embed).item()
            if sim > best_sim:
                best_sim, best_group = sim, group_idx

        if best_sim >= current_threshold:
            groups[best_group].append(i)
            group_embeddings[best_group] = torch.mean(torch.stack([embeddings[idx] for idx in groups[best_group]]), dim=0)
        else:
            groups.append([i])
            group_embeddings.append(new_embedding)

        threshold_history.append(current_threshold)

        if i % current_window == 0:
            predicted_length = estimate_speech_length(groups, i)
            current_window = max(3, min(10, int(0.1 * predicted_length)))

            staleness = enhanced_staleness(groups, i, staleness_history)
            saturation = enhanced_saturation(groups, i, saturation_history)
            current_progress = calculate_progress(staleness, saturation, i, predicted_length)
            smoothed_progress = 0.9 * smoothed_progress + 0.1 * current_progress
            progress_history.append(smoothed_progress)
            
            progress_data.append({
                'position': i,
                'progress': smoothed_progress,
                'staleness': staleness,
                'saturation': saturation,
                'threshold': current_threshold,
                'estimated_length': predicted_length
            })

            progress_line = f"Pos {i:3d}: Progress={smoothed_progress:5.1f}% | " \
                           f"Staleness={staleness:.2f} | " \
                           f"Saturation={saturation:.2f} | " \
                           f"Threshold={current_threshold:.2f} | " \
                           f"EstLen={predicted_length}"
            print(progress_line)
            # Also print to actual console (not captured)
            print(progress_line, file=sys.__stdout__)

    # Format groups for output
    formatted_groups = []
    for idx, group in enumerate(groups):
        group_sentences = [sentences[sent_idx] for sent_idx in group]
        formatted_groups.append({
            'id': idx + 1,
            'sentence_indices': group,
            'sentences': group_sentences
        })

    return {
        'groups': formatted_groups,
        'progress_history': progress_history,
        'progress_data': progress_data,
        'final_progress': smoothed_progress,
        'total_groups': len(groups),
        'total_segments': len(sentences),
        'threshold_history': threshold_history
    }

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
            
        use_coref_resolution = data.get('use_coref_resolution', True)
        
        # Capture console output for debugging
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Original sentence count
            original_sentences = split_sentences(text)
            original_sentence_count = len(original_sentences)
            
            # Step 1: Coreference resolution (if enabled)
            if use_coref_resolution:
                print("=== Applying Coreference Resolution ===")
                processed_text = replace_mentions(text)
            else:
                print("=== Skipping Coreference Resolution ===")
                processed_text = text
            
            # Step 2: Process text (split, filter, merge)
            print("=== Processing Text ===")
            processed_sentences = process_text(processed_text)
            final_sentence_count = len(processed_sentences)
            
            # Step 3: Run analysis
            print("=== Running Analysis ===")
            results = linear_grouping_quantified(processed_sentences, sentModel)
            
            # Add preprocessing info
            results.update({
                'original_sentence_count': original_sentence_count,
                'final_sentence_count': final_sentence_count,
                'processed_text': processed_text,
                'use_coref_resolution': use_coref_resolution
            })
            
        finally:
            sys.stdout = old_stdout
            console_output = captured_output.getvalue()
        
        # Add console output for debugging
        results['console_output'] = console_output
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'spacy': nlp is not None,
            'sentence_transformer': sentModel is not None,
            'fastcoref': corefmodel is not None
        }
    })

if __name__ == '__main__':
    print("Starting Flask backend...")
    print("Loading models...")
    print(f"spaCy model loaded: {nlp is not None}")
    print(f"SentenceTransformer model loaded: {sentModel is not None}")  
    print(f"FastCoref model loaded: {corefmodel is not None}")
    print("Backend ready!")
    app.run(debug=True, host='127.0.0.1', port=5000)