# 🧠 T2I-Copilot Memory System - Exact Flow Documentation

## 📊 **MEMORY ARCHITECTURE OVERVIEW**

```
T2I-Copilot Generation Process
           ↓
    MemoryManager (Central Hub)
           ↓
    ModelMemoryModule (Per Model)
      ↙           ↘
GlobalMemory    LocalMemory
(SQLite DB)    (Pattern Analysis)
     ↓              ↓
 Evaluation     Pattern Analysis
   Storage       & Adaptive Prompts
```

## 🔄 **EXACT MEMORY FLOW - STEP BY STEP**

### **Phase 1: Initialization**
```python
# 1. System Startup
MemoryManager() 
    → Creates ModelMemoryModule("Qwen-Image", db_path)
    → Creates ModelMemoryModule("Qwen-Image-Edit", db_path)
    
# 2. Model Memory Module Setup
ModelMemoryModule.__init__()
    → GlobalMemory(model_name, db_path)  # Creates SQLite tables
    → LocalMemory(model_name, db_path)   # Pattern analysis setup
        → PatternAnalyzer(model_name, db_path)
        → AdaptivePromptGenerator(model_name, db_path)
```

### **Phase 2: Evaluation Storage (Every Generation)**
```python
# 1. T2I Generation Completes → Evaluation Done
memory_manager.get_model_memory("Qwen-Image").add_evaluation_to_global_memory(evaluation_data)

# 2. Global Memory Storage
ModelMemoryModule.add_evaluation_to_global_memory()
    → GlobalMemory.add_evaluation(evaluation_data)
        → SQLite INSERT into {model_name}_global_memory table
        → Stores: prompt, score, artifacts, missing_elements, timestamp, etc.
        → LOG: "Added evaluation to {model_name} global memory: score={score}"

# 3. Pattern Analysis Trigger Check
    → LocalMemory.check_and_trigger_analysis()
        → PatternAnalyzer.should_trigger_analysis()
            → COUNT evaluations since last analysis
            → IF count >= 200: TRIGGER ANALYSIS
```

### **Phase 3: Pattern Analysis (Every 200+ Evaluations)**
```python
# 1. Analysis Triggered
LocalMemory.check_and_trigger_analysis() [count >= 200]
    → LOG: "Triggering pattern analysis for {model_name}"
    
# 2. Pattern Analysis Execution
    → PatternAnalyzer.analyze_patterns()
        → _get_evaluations_for_analysis() # Get last 300 evaluations
        → _analyze_artifacts()           # Find common artifacts
        → _analyze_missing_elements()    # Find missing elements
        → _analyze_low_score_patterns()  # Correlate low scores
        → _analyze_improvement_themes()  # Extract improvement patterns
        → _calculate_confidence_score()  # Overall confidence (0.0-1.0)
        → Returns PatternAnalysis object
        → LOG: "Pattern analysis completed with confidence: {confidence}"

# 3. Save Pattern Analysis
    → LocalMemory._save_pattern_analysis(analysis)
        → SQLite INSERT into pattern_analysis table
        → LOG: "Saved pattern analysis {id} for {model_name}"
```

### **Phase 4: Adaptive Prompt Generation (After Pattern Analysis)**
```python
# 1. Generate Adaptive Prompts
    → AdaptivePromptGenerator.generate_adaptive_prompts(analysis)
        → _generate_artifact_mitigation_prompts()
        → _generate_missing_element_prompts()
        → _generate_quality_enhancement_prompts()
        → Returns AdaptivePrompt object with:
            - positive_additions: ["high quality", "detailed", "realistic"]
            - negative_additions: ["blurry", "distorted", "low quality"]
            - effectiveness: 1.0 (initial)
        → LOG: "Generated adaptive prompts for {model_name}"

# 2. Save Adaptive Prompts
    → LocalMemory._save_adaptive_prompt(adaptive_prompt)
        → SQLite INSERT into adaptive_prompts table
        → Links to pattern_analysis_id (foreign key)
        → LOG: "Saved adaptive prompt {id} for {model_name}"
        → LOG: "Pattern analysis and adaptive prompts generated for {model_name}"
```

### **Phase 5: Adaptive Prompt Usage (Next Generations)**
```python
# 1. Request Adaptive Prompts
adaptive_prompts = memory_manager.get_adaptive_prompts_for_model("Qwen-Image")
    → ModelMemoryModule.get_adaptive_prompts()
        → LocalMemory.get_active_adaptive_prompts()
            → SQLite SELECT most recent effective prompts
            → Returns AdaptivePrompt object or None

# 2. Enhance Generation Prompts
if adaptive_prompts:
    enhanced_prompt = original_prompt + " " + " ".join(adaptive_prompts.positive_additions)
    enhanced_negative = original_negative + " " + " ".join(adaptive_prompts.negative_additions)

# 3. Track Effectiveness (After Generation)
memory_manager.update_adaptive_prompt_effectiveness("Qwen-Image", evaluation_score, used=True)
    → Updates effectiveness_score in adaptive_prompts table
    → Learns which prompts work best
```

## 🗃️ **DATABASE SCHEMA FLOW**

### **Global Memory Tables (Per Model)**
```sql
-- Stores every single evaluation
qwen_image_global_memory:
    id, model_name, prompt, original_prompt, negative_prompt,
    evaluation_score, artifacts, missing_elements, timestamp, etc.
    
qwen_image_edit_global_memory:
    [Same structure for Image-Edit model]
```

### **Local Memory Tables (Shared)**
```sql
-- Stores pattern analysis results
pattern_analysis:
    id, model_name, analysis_period_start/end,
    common_artifacts (JSON), frequent_missing_elements (JSON),
    confidence_score, created_at
    
-- Stores generated adaptive prompts
adaptive_prompts:
    id, model_name, pattern_analysis_id (FK),
    positive_prompt_additions (JSON), negative_prompt_additions (JSON),
    effectiveness_score, created_at
```

## ⚡ **TRIGGER CONDITIONS**

| Event | Condition | Action |
|-------|-----------|--------|
| **Every Generation** | Evaluation completed | Store in GlobalMemory |
| **Every 200+ Evaluations** | Pattern analysis threshold | Analyze patterns + Generate adaptive prompts |
| **Every Generation** | Adaptive prompts exist | Enhance prompt with adaptive additions |
| **After Enhanced Generation** | Evaluation scored | Update adaptive prompt effectiveness |

## 📈 **DATA FLOW RELATIONSHIPS**

```
Evaluation → GlobalMemory (1:∞)
    ↓
Pattern Analysis (triggered every 200+)
    ↓
AdaptivePrompt Generation (1:1 with analysis)
    ↓
Prompt Enhancement (∞:1 with adaptive prompts)
    ↓
Effectiveness Tracking (feedback loop)
```

## 🔧 **MEMORY STATES**

### **State 1: Cold Start (0-199 evaluations)**
- ✅ Global memory: Storing evaluations
- ❌ Local memory: No pattern analysis yet
- ❌ Adaptive prompts: None available

### **State 2: Pattern Analysis Active (200+ evaluations)**
- ✅ Global memory: Storing evaluations  
- ✅ Local memory: Pattern analysis running every 200+ evals
- ✅ Adaptive prompts: Generated and available

### **State 3: Learning & Optimization (Ongoing)**
- ✅ Global memory: Continuous storage
- ✅ Local memory: Regular pattern updates
- ✅ Adaptive prompts: Effectiveness tracking & refinement

## 🎯 **INTEGRATION POINTS**

### **For Main T2I-Copilot System:**
```python
# 1. Initialize Memory (Once)
memory_manager = MemoryManager()

# 2. After Each Generation
memory_manager.get_model_memory(model_name).add_evaluation_to_global_memory(evaluation_data)

# 3. Before Each Generation (Optional Enhancement)
adaptive_prompts = memory_manager.get_adaptive_prompts_for_model(model_name)
if adaptive_prompts:
    prompt += " " + " ".join(adaptive_prompts.positive_additions)
    negative_prompt += " " + " ".join(adaptive_prompts.negative_additions)

# 4. After Enhanced Generation (Feedback)
memory_manager.update_adaptive_prompt_effectiveness(model_name, score, used=True)
```

This memory system creates a **self-improving feedback loop** where the T2I-Copilot automatically learns from past generations and enhances future prompts to avoid common flaws! 🚀
