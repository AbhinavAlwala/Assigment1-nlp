import os
import hmm
import argparse
import pandas as pd
import numpy as np

def evaluate():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'en_ewt-ud-train.conllu')
    test_file = os.path.join(base_dir, 'data', 'en_ewt-ud-test.conllu')
    
    # 1. Train
    print("Training HMM Model...")
    trans_mat, emit_mat, pi, vocab, tags = hmm.train_hmm(train_file)
    print("Training Complete.")
    
    # 2. Run Viterbi
    print("Running Viterbi on Test Data...")
    predictions, ground_truth_data = hmm.viterbi_algorithm(trans_mat, emit_mat, pi, test_file, vocab, tags)
    
    # 3. Calculate Accuracy
    total_tokens = 0
    correct_tokens = 0
    
    all_pred = []
    all_true = []
    
    for pred_sent, true_sent in zip(predictions, ground_truth_data):
        # true_sent is list of (word, tag)
        true_tags = [t for w, t in true_sent]
        
        # Length check
        if len(pred_sent) != len(true_tags):
            continue
            
        for p, t in zip(pred_sent, true_tags):
            total_tokens += 1
            if p == t:
                correct_tokens += 1
            all_pred.append(p)
            all_true.append(t)
                
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    print(f"Total Tokens: {total_tokens}")
    print(f"Correct Tokens: {correct_tokens}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy*100:.2f}%")
    
    print("\n--- Confusion Matrix (Top 10 Tags) ---")
    # Using Pandas to stay within allowed libraries
    df = pd.DataFrame({'True': all_true, 'Predicted': all_pred})
    
    # Get top 10 most frequent tags to keep matrix readable
    top_tags = df['True'].value_counts().head(10).index
    df_filtered = df[df['True'].isin(top_tags) & df['Predicted'].isin(top_tags)]
    
    cm = pd.crosstab(df_filtered['True'], df_filtered['Predicted'], normalize='index')
    print(cm.round(2))

if __name__ == "__main__":
    evaluate()
