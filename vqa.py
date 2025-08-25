import t2v_metrics
import json
import os
import glob
from pathlib import Path

def evaluate_dataset():
    # Initialize the VQA scoring model
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl')
    
    # Load prompts from JSON file
    with open('/mnt/localssd/shivank/agents/agents-image-gen/eval_benchmark/GenAIBenchmark/genai_image_seed.json', 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # Define the image folder path
    image_folder = "/mnt/localssd/shivank/agents/agents-image-gen/results/qwen_direct"
    
    # Get all image files in the folder
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(image_folder, "*.jpeg")))
    
    results = {}
    scores = []
    
    print(f"Found {len(image_files)} images to evaluate...")
    
    for image_path in image_files:
        # Extract the ID from the filename (e.g., "00000_Qwen-Image.png" -> "00000")
        filename = os.path.basename(image_path)
        image_id = filename.split('_')[0]
        
        if image_id in prompts_data:
            # Get the prompt text for this image
            prompt_text = prompts_data[image_id]["prompt"]
            
            print(f"Evaluating image {image_id}: {filename}")
            print(f"Prompt: {prompt_text}")
            
            # Calculate VQA score
            try:
                score = clip_flant5_score(images=[image_path], texts=[prompt_text])
                score_value = float(score[0]) if isinstance(score, (list, tuple)) else float(score)
                
                results[image_id] = {
                    "image_file": filename,
                    "prompt": prompt_text,
                    "vqa_score": score_value,
                    "random_seed": prompts_data[image_id]["random_seed"]
                }
                
                scores.append(score_value)
                print(f"VQA Score: {score_value:.4f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error evaluating {filename}: {str(e)}")
                results[image_id] = {
                    "image_file": filename,
                    "prompt": prompt_text,
                    "vqa_score": None,
                    "error": str(e),
                    "random_seed": prompts_data[image_id]["random_seed"]
                }
        else:
            print(f"Warning: No prompt found for image ID {image_id}")
    
    # Calculate average score
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"\nAverage VQA Score: {average_score:.4f}")
        print(f"Total images evaluated: {len(scores)}")
    else:
        average_score = None
        print("\nNo scores calculated.")
    
    # Add summary to results
    results["_summary"] = {
        "total_images_evaluated": len(scores),
        "average_vqa_score": average_score,
        "total_images_found": len(image_files),
        "total_prompts_in_dataset": len(prompts_data)
    }
    
    # Save results to JSON file
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to 'results.json'")
    return results

if __name__ == "__main__":
    results = evaluate_dataset()