"""
Memory Modules for T2I-Copilot Agent System

This module implements memory systems for Qwen-Image and Qwen-Image-Edit models.
Each model has:
- Global Memory: Logs of evaluation results for that specific model
- Local Memory: (To be implemented later)

The memory system uses SQLite for local database storage.
"""

import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMemory:
    """Data structure for storing evaluation results in memory."""
    model_name: str
    prompt: str
    original_prompt: str
    negative_prompt: str
    image_path: str
    evaluation_score: float
    aesthetic_reasoning: str
    alignment_reasoning: str
    overall_reasoning: str
    improvement_suggestions: str
    detected_artifacts: List[str]
    artifact_reasoning: str
    main_subjects_present: bool
    missing_elements: List[str]
    timestamp: str
    session_id: str
    regeneration_count: int
    seed: int
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMemory':
        """Create instance from dictionary."""
        return cls(**data)

@dataclass
class PatternAnalysis:
    """Data structure for storing pattern analysis results."""
    model_name: str
    analysis_period_start: str
    analysis_period_end: str
    total_evaluations_analyzed: int
    common_artifacts: List[str]
    frequent_missing_elements: List[str]
    low_score_patterns: Dict[str, Any]
    improvement_themes: List[str]
    prompt_type_issues: Dict[str, List[str]]
    confidence_score: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternAnalysis':
        """Create instance from dictionary."""
        return cls(**data)

@dataclass
class AdaptivePrompt:
    """Data structure for storing adaptive prompt enhancements."""
    model_name: str
    pattern_analysis_id: int
    positive_prompt_additions: str
    negative_prompt_additions: str
    target_flaws: List[str]
    effectiveness_score: float
    usage_count: int
    last_used: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptivePrompt':
        """Create instance from dictionary."""
        return cls(**data)

class PatternAnalyzer:
    """Analyzes patterns in evaluation data to identify common flaws and generate adaptive prompts."""
    
    def __init__(self, model_name: str, db_path: str):
        self.model_name = model_name
        self.db_path = db_path
        self.logger = logger
        
        # Configuration
        self.ANALYSIS_TRIGGER_COUNT = 200
        self.MIN_CONFIDENCE_THRESHOLD = 0.7
        self.LOW_SCORE_THRESHOLD = 7.0
        
    def should_trigger_analysis(self) -> bool:
        """Check if pattern analysis should be triggered based on evaluation count."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                table_name = f"{self.model_name.replace('-', '_').lower()}_global_memory"
                
                # Get count of evaluations since last analysis
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE created_at > COALESCE(
                        (SELECT MAX(analysis_period_end) FROM pattern_analysis WHERE model_name = ?),
                        '1970-01-01'
                    )
                """, (self.model_name,))
                
                count = cursor.fetchone()[0]
                self.logger.debug(f"Evaluations since last analysis for {self.model_name}: {count}")
                
                return count >= self.ANALYSIS_TRIGGER_COUNT
                
        except Exception as e:
            self.logger.error(f"Error checking analysis trigger: {e}")
            return False
    
    def analyze_patterns(self) -> Optional[PatternAnalysis]:
        """Perform comprehensive pattern analysis on recent evaluations."""
        try:
            # Get recent evaluations for analysis
            evaluations = self._get_evaluations_for_analysis()
            if len(evaluations) < 50:  # Need minimum data for meaningful analysis
                self.logger.warning(f"Insufficient data for analysis: {len(evaluations)} evaluations")
                return None
                
            self.logger.info(f"Analyzing patterns for {self.model_name} with {len(evaluations)} evaluations")
            
            # Perform different types of analysis
            common_artifacts = self._analyze_artifacts(evaluations)
            missing_elements = self._analyze_missing_elements(evaluations)
            low_score_patterns = self._analyze_low_score_patterns(evaluations)
            improvement_themes = self._analyze_improvement_themes(evaluations)
            prompt_type_issues = self._analyze_prompt_type_issues(evaluations)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(evaluations, common_artifacts, missing_elements)
            
            # Create pattern analysis object
            analysis = PatternAnalysis(
                model_name=self.model_name,
                analysis_period_start=evaluations[-1].timestamp,  # oldest
                analysis_period_end=evaluations[0].timestamp,     # newest
                total_evaluations_analyzed=len(evaluations),
                common_artifacts=common_artifacts,
                frequent_missing_elements=missing_elements,
                low_score_patterns=low_score_patterns,
                improvement_themes=improvement_themes,
                prompt_type_issues=prompt_type_issues,
                confidence_score=confidence,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"Pattern analysis completed with confidence: {confidence:.2f}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            return None
    
    def _get_evaluations_for_analysis(self) -> List[EvaluationMemory]:
        """Get recent evaluations for pattern analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                table_name = f"{self.model_name.replace('-', '_').lower()}_global_memory"
                
                # Get evaluations since last analysis, or last 200 if no previous analysis
                cursor.execute(f"""
                    SELECT * FROM {table_name}
                    WHERE created_at > COALESCE(
                        (SELECT MAX(analysis_period_end) FROM pattern_analysis WHERE model_name = ?),
                        datetime('now', '-30 days')
                    )
                    ORDER BY created_at DESC
                    LIMIT 300
                """, (self.model_name,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                evaluations = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    # Convert JSON strings back to lists
                    row_dict['detected_artifacts'] = json.loads(row_dict['detected_artifacts'])
                    row_dict['missing_elements'] = json.loads(row_dict['missing_elements'])
                    # Remove database-specific fields
                    row_dict.pop('id', None)
                    row_dict.pop('created_at', None)
                    
                    evaluations.append(EvaluationMemory.from_dict(row_dict))
                
                return evaluations
                
        except Exception as e:
            self.logger.error(f"Error getting evaluations for analysis: {e}")
            return []
    
    def _analyze_artifacts(self, evaluations: List[EvaluationMemory]) -> List[str]:
        """Analyze common artifacts across evaluations."""
        artifact_counter = Counter()
        
        for eval_mem in evaluations:
            for artifact in eval_mem.detected_artifacts:
                artifact_counter[artifact.lower().strip()] += 1
        
        # Get artifacts that appear in at least 15% of evaluations
        min_frequency = max(3, len(evaluations) * 0.15)
        common_artifacts = [
            artifact for artifact, count in artifact_counter.most_common()
            if count >= min_frequency
        ]
        
        self.logger.debug(f"Common artifacts: {common_artifacts}")
        return common_artifacts[:10]  # Top 10 most common
    
    def _analyze_missing_elements(self, evaluations: List[EvaluationMemory]) -> List[str]:
        """Analyze frequently missing elements."""
        missing_counter = Counter()
        
        for eval_mem in evaluations:
            for element in eval_mem.missing_elements:
                missing_counter[element.lower().strip()] += 1
        
        # Get elements missing in at least 10% of evaluations
        min_frequency = max(2, len(evaluations) * 0.10)
        frequent_missing = [
            element for element, count in missing_counter.most_common()
            if count >= min_frequency
        ]
        
        self.logger.debug(f"Frequent missing elements: {frequent_missing}")
        return frequent_missing[:8]  # Top 8 most frequent
    
    def _analyze_low_score_patterns(self, evaluations: List[EvaluationMemory]) -> Dict[str, Any]:
        """Analyze patterns in low-scoring evaluations."""
        low_score_evals = [e for e in evaluations if e.evaluation_score < self.LOW_SCORE_THRESHOLD]
        
        if not low_score_evals:
            return {}
        
        # Artifacts in low-scoring images
        low_score_artifacts = Counter()
        for eval_mem in low_score_evals:
            for artifact in eval_mem.detected_artifacts:
                low_score_artifacts[artifact.lower().strip()] += 1
        
        # Average score by artifact presence
        artifact_score_impact = {}
        all_artifacts = set()
        for eval_mem in evaluations:
            all_artifacts.update(eval_mem.detected_artifacts)
        
        for artifact in all_artifacts:
            with_artifact = [e.evaluation_score for e in evaluations if artifact in e.detected_artifacts]
            without_artifact = [e.evaluation_score for e in evaluations if artifact not in e.detected_artifacts]
            
            if with_artifact and without_artifact:
                impact = sum(without_artifact) / len(without_artifact) - sum(with_artifact) / len(with_artifact)
                if impact > 0.5:  # Significant negative impact
                    artifact_score_impact[artifact] = round(impact, 2)
        
        return {
            "low_score_count": len(low_score_evals),
            "low_score_percentage": round(len(low_score_evals) / len(evaluations) * 100, 1),
            "common_low_score_artifacts": dict(low_score_artifacts.most_common(5)),
            "artifact_score_impact": artifact_score_impact
        }
    
    def _analyze_improvement_themes(self, evaluations: List[EvaluationMemory]) -> List[str]:
        """Extract common themes from improvement suggestions using text analysis."""
        all_suggestions = " ".join([
            eval_mem.improvement_suggestions.lower() 
            for eval_mem in evaluations 
            if eval_mem.improvement_suggestions
        ])
        
        # Common improvement keywords/phrases
        improvement_patterns = [
            r'\b(better|improve|enhance|increase)\s+(\w+)',
            r'\b(reduce|decrease|minimize|avoid)\s+(\w+)',
            r'\b(add|include|incorporate)\s+(more\s+)?(\w+)',
            r'\b(fix|correct|adjust)\s+(\w+)',
            r'\b(\w+)\s+(quality|resolution|detail|balance)',
            r'\b(lighting|color|composition|focus|sharpness|contrast)',
        ]
        
        themes = []
        for pattern in improvement_patterns:
            matches = re.findall(pattern, all_suggestions)
            for match in matches:
                if isinstance(match, tuple):
                    theme = " ".join([part for part in match if part and len(part) > 2])
                else:
                    theme = match
                
                if len(theme) > 3:
                    themes.append(theme.strip())
        
        # Count and return most common themes
        theme_counter = Counter(themes)
        return [theme for theme, count in theme_counter.most_common(8) if count >= 2]
    
    def _analyze_prompt_type_issues(self, evaluations: List[EvaluationMemory]) -> Dict[str, List[str]]:
        """Analyze issues specific to different prompt types."""
        prompt_categories = {
            "portrait": ["portrait", "person", "face", "human", "man", "woman", "child"],
            "landscape": ["landscape", "nature", "mountain", "forest", "beach", "sky", "outdoor"],
            "object": ["object", "item", "product", "tool", "furniture", "vehicle"],
            "abstract": ["abstract", "artistic", "creative", "surreal", "fantasy"],
            "animal": ["animal", "cat", "dog", "bird", "wildlife", "pet"]
        }
        
        categorized_issues = defaultdict(list)
        
        for eval_mem in evaluations:
            prompt_lower = eval_mem.original_prompt.lower()
            
            # Categorize prompt
            prompt_type = "other"
            for category, keywords in prompt_categories.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    prompt_type = category
                    break
            
            # Collect issues for this prompt type
            if eval_mem.evaluation_score < 8.0:  # Focus on problematic evaluations
                issues = eval_mem.detected_artifacts + eval_mem.missing_elements
                categorized_issues[prompt_type].extend(issues)
        
        # Get most common issues per category
        result = {}
        for prompt_type, issues in categorized_issues.items():
            if issues:
                issue_counter = Counter(issues)
                result[prompt_type] = [issue for issue, count in issue_counter.most_common(5)]
        
        return result
    
    def _calculate_confidence_score(self, evaluations: List[EvaluationMemory], 
                                  artifacts: List[str], missing_elements: List[str]) -> float:
        """Calculate confidence score for the pattern analysis."""
        factors = []
        
        # Factor 1: Sample size
        size_factor = min(1.0, len(evaluations) / 200)
        factors.append(size_factor)
        
        # Factor 2: Pattern consistency (how often patterns appear)
        if artifacts or missing_elements:
            pattern_consistency = len(artifacts + missing_elements) / 20  # max 20 patterns
            factors.append(min(1.0, pattern_consistency))
        else:
            factors.append(0.3)  # Low confidence if no patterns found
        
        # Factor 3: Score variance (more variance = more reliable patterns)
        scores = [e.evaluation_score for e in evaluations]
        if len(set(scores)) > 1:
            score_variance = (max(scores) - min(scores)) / 10.0  # normalize to 0-1
            factors.append(min(1.0, score_variance))
        else:
            factors.append(0.5)
        
        # Factor 4: Data recency (newer data = higher confidence)
        try:
            newest = datetime.fromisoformat(evaluations[0].timestamp)
            oldest = datetime.fromisoformat(evaluations[-1].timestamp)
            days_span = (newest - oldest).days
            recency_factor = max(0.5, 1.0 - (days_span / 30))  # Decay over 30 days
            factors.append(recency_factor)
        except:
            factors.append(0.7)
        
        # Calculate weighted average
        confidence = sum(factors) / len(factors)
        return round(confidence, 3)

class AdaptivePromptGenerator:
    """Generates adaptive prompts based on pattern analysis to mitigate identified flaws."""
    
    def __init__(self, model_name: str, db_path: str):
        self.model_name = model_name
        self.db_path = db_path
        self.logger = logger
        
        # Predefined mappings for common issues
        self.artifact_to_negative = {
            "blur": "blur, blurry, soft focus, out of focus, unclear",
            "distortion": "distortion, distorted, warped, stretched, deformed",
            "pixelation": "pixelated, low resolution, blocky, jagged edges",
            "noise": "noise, grainy, rough texture, artifacts",
            "oversaturation": "oversaturated, too bright, excessive colors, neon",
            "color bleeding": "color bleeding, color overflow, smudged colors",
            "artifacts": "artifacts, glitches, errors, defects",
            "low quality": "low quality, poor quality, bad quality",
            "watermark": "watermark, text overlay, logos, signatures",
            "multiple heads": "multiple heads, extra heads, duplicated features",
            "extra limbs": "extra limbs, additional arms, extra legs, malformed body",
            "asymmetry": "asymmetrical, unbalanced, lopsided, crooked"
        }
        
        self.artifact_to_positive = {
            "blur": "sharp focus, crisp details, clear definition, high definition",
            "distortion": "proper proportions, accurate shapes, realistic anatomy",
            "pixelation": "high resolution, smooth textures, detailed rendering",
            "noise": "clean image, smooth gradients, pristine quality",
            "oversaturation": "natural colors, balanced saturation, realistic tones",
            "color bleeding": "precise colors, clean edges, well-defined boundaries",
            "artifacts": "flawless rendering, perfect quality, pristine image",
            "low quality": "high quality, professional grade, premium rendering",
            "multiple heads": "single head, proper anatomy, correct proportions",
            "extra limbs": "correct anatomy, proper body structure, realistic form"
        }
        
        self.missing_to_positive = {
            "background": "detailed background, rich environment, complete scene",
            "lighting": "proper lighting, good illumination, natural light",
            "details": "rich details, fine textures, intricate elements",
            "depth": "good depth, dimensional perspective, layered composition",
            "contrast": "good contrast, clear distinction, balanced tones",
            "composition": "well composed, balanced layout, harmonious arrangement",
            "focus": "sharp focus, clear subject, proper emphasis",
            "texture": "rich textures, detailed surfaces, tactile quality"
        }
    
    def generate_adaptive_prompts(self, pattern_analysis: PatternAnalysis) -> Optional[AdaptivePrompt]:
        """Generate adaptive prompts based on pattern analysis."""
        try:
            if pattern_analysis.confidence_score < 0.5:
                self.logger.info(f"Confidence too low ({pattern_analysis.confidence_score}) for adaptive prompts")
                return None
            
            # Generate positive prompt additions
            positive_additions = self._generate_positive_additions(pattern_analysis)
            
            # Generate negative prompt additions
            negative_additions = self._generate_negative_additions(pattern_analysis)
            
            # Identify target flaws this prompt addresses
            target_flaws = (pattern_analysis.common_artifacts + 
                          pattern_analysis.frequent_missing_elements)
            
            # Create adaptive prompt object
            adaptive_prompt = AdaptivePrompt(
                model_name=self.model_name,
                pattern_analysis_id=0,  # Will be set when saved to database
                positive_prompt_additions=positive_additions,
                negative_prompt_additions=negative_additions,
                target_flaws=target_flaws,
                effectiveness_score=0.0,  # Will be updated based on performance
                usage_count=0,
                last_used="",
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"Generated adaptive prompts for {self.model_name}")
            self.logger.debug(f"Positive additions: {positive_additions}")
            self.logger.debug(f"Negative additions: {negative_additions}")
            
            return adaptive_prompt
            
        except Exception as e:
            self.logger.error(f"Error generating adaptive prompts: {e}")
            return None
    
    def _generate_positive_additions(self, analysis: PatternAnalysis) -> str:
        """Generate positive prompt additions to enhance quality."""
        additions = []
        
        # Address common artifacts with positive reinforcement
        for artifact in analysis.common_artifacts:
            if artifact.lower() in self.artifact_to_positive:
                additions.append(self.artifact_to_positive[artifact.lower()])
        
        # Address missing elements
        for element in analysis.frequent_missing_elements:
            if element.lower() in self.missing_to_positive:
                additions.append(self.missing_to_positive[element.lower()])
        
        # Add model-specific quality improvements
        if self.model_name == "Qwen-Image":
            additions.append("professional photography, studio quality, perfect composition")
        elif self.model_name == "Qwen-Image-Edit":
            additions.append("seamless editing, natural transitions, flawless integration")
        
        # Add improvements based on analysis themes
        for theme in analysis.improvement_themes:
            if "lighting" in theme.lower():
                additions.append("professional lighting, well-lit, proper illumination")
            elif "color" in theme.lower():
                additions.append("accurate colors, natural palette, color harmony")
            elif "detail" in theme.lower():
                additions.append("rich details, fine textures, intricate work")
            elif "composition" in theme.lower():
                additions.append("excellent composition, balanced framing, artistic layout")
        
        # Combine and clean up
        positive_text = ", ".join(additions)
        return self._clean_and_limit_prompt(positive_text, max_length=100)
    
    def _generate_negative_additions(self, analysis: PatternAnalysis) -> str:
        """Generate negative prompt additions to avoid identified flaws."""
        additions = []
        
        # Address common artifacts directly
        for artifact in analysis.common_artifacts:
            if artifact.lower() in self.artifact_to_negative:
                additions.append(self.artifact_to_negative[artifact.lower()])
            else:
                additions.append(artifact.lower())
        
        # Address low score patterns
        if analysis.low_score_patterns:
            for artifact in analysis.low_score_patterns.get("common_low_score_artifacts", {}):
                if artifact.lower() not in [a.lower() for a in additions]:
                    additions.append(artifact.lower())
        
        # Add model-specific negative terms
        if self.model_name == "Qwen-Image":
            additions.append("low quality, blurry, distorted, amateur photography")
        elif self.model_name == "Qwen-Image-Edit":
            additions.append("visible edits, harsh transitions, obvious manipulation, artifacts")
        
        # Add prompt-type specific negatives based on issues
        for prompt_type, issues in analysis.prompt_type_issues.items():
            if prompt_type == "portrait":
                additions.extend(["distorted face", "multiple faces", "asymmetrical features"])
            elif prompt_type == "landscape":
                additions.extend(["cluttered composition", "unnatural colors", "poor perspective"])
        
        # Combine and clean up
        negative_text = ", ".join(additions)
        return self._clean_and_limit_prompt(negative_text, max_length=80)
    
    def _clean_and_limit_prompt(self, prompt_text: str, max_length: int = 100) -> str:
        """Clean up and limit the length of prompt additions."""
        # Remove duplicates while preserving order
        seen = set()
        parts = []
        for part in prompt_text.split(", "):
            part = part.strip().lower()
            if part and part not in seen and len(part) > 2:
                seen.add(part)
                parts.append(part)
        
        # Limit total length
        result = ", ".join(parts)
        if len(result) > max_length:
            # Truncate at word boundary
            truncated = result[:max_length]
            last_comma = truncated.rfind(", ")
            if last_comma > max_length * 0.7:  # Don't truncate too much
                result = truncated[:last_comma]
            else:
                result = truncated
        
        return result.strip()

class ModelMemoryModule:
    """Memory module for a specific model (Qwen-Image or Qwen-Image-Edit)."""
    
    def __init__(self, model_name: str, db_path: str = "model_memories.db"):
        self.model_name = model_name
        self.db_path = db_path
        self.global_memory = GlobalMemory(model_name, db_path)
        self.local_memory = LocalMemory(model_name, db_path)
        
    def add_evaluation_to_global_memory(self, evaluation_data: EvaluationMemory):
        """Add evaluation result to global memory and check for pattern analysis trigger."""
        self.global_memory.add_evaluation(evaluation_data)
        
        # Check if we should trigger pattern analysis
        self.local_memory.check_and_trigger_analysis()
        
    def get_global_memory_stats(self) -> Dict[str, Any]:
        """Get statistics from global memory."""
        return self.global_memory.get_stats()
        
    def get_recent_evaluations(self, limit: int = 10) -> List[EvaluationMemory]:
        """Get recent evaluations from global memory."""
        return self.global_memory.get_recent_evaluations(limit)
        
    def search_evaluations_by_prompt(self, prompt_keywords: str, limit: int = 10) -> List[EvaluationMemory]:
        """Search evaluations by prompt keywords."""
        return self.global_memory.search_by_prompt(prompt_keywords, limit)
    
    def get_adaptive_prompts(self) -> Optional[AdaptivePrompt]:
        """Get active adaptive prompts for enhancing generation."""
        return self.local_memory.get_active_adaptive_prompts()
    
    def update_adaptive_prompt_effectiveness(self, prompt_id: int, score: float, used: bool = True):
        """Update adaptive prompt effectiveness based on evaluation results."""
        self.local_memory.update_adaptive_prompt_effectiveness(prompt_id, score, used)
    
    def get_pattern_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of recent pattern analyses."""
        try:
            analyses = self.local_memory.get_pattern_analysis_history(3)
            if not analyses:
                return {"status": "no_analyses", "message": "No pattern analyses available"}
            
            latest = analyses[0]
            return {
                "status": "active",
                "latest_analysis_date": latest.timestamp,
                "confidence_score": latest.confidence_score,
                "evaluations_analyzed": latest.total_evaluations_analyzed,
                "common_artifacts": latest.common_artifacts[:5],  # Top 5
                "missing_elements": latest.frequent_missing_elements[:3],  # Top 3
                "total_analyses": len(analyses)
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern analysis summary: {e}")
            return {"status": "error", "message": str(e)}

class GlobalMemory:
    """Global memory for storing all evaluation logs for a specific model."""
    
    def __init__(self, model_name: str, db_path: str):
        self.model_name = model_name
        self.db_path = db_path
        self.table_name = f"{model_name.replace('-', '_').lower()}_global_memory"
        self._init_database()
        
    def _init_database(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create table for global memory
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        original_prompt TEXT NOT NULL,
                        negative_prompt TEXT,
                        image_path TEXT NOT NULL,
                        evaluation_score REAL NOT NULL,
                        aesthetic_reasoning TEXT,
                        alignment_reasoning TEXT,
                        overall_reasoning TEXT,
                        improvement_suggestions TEXT,
                        detected_artifacts TEXT,  -- JSON string
                        artifact_reasoning TEXT,
                        main_subjects_present BOOLEAN,
                        missing_elements TEXT,  -- JSON string
                        timestamp TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        regeneration_count INTEGER,
                        seed INTEGER,
                        confidence_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster searches
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
                    ON {self.table_name} (timestamp)
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_prompt 
                    ON {self.table_name} (prompt)
                """)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_score 
                    ON {self.table_name} (evaluation_score)
                """)
                
                conn.commit()
                logger.info(f"Database initialized for {self.model_name} global memory")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
            
    def add_evaluation(self, evaluation: EvaluationMemory):
        """Add an evaluation result to global memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert lists to JSON strings for storage
                detected_artifacts_json = json.dumps(evaluation.detected_artifacts)
                missing_elements_json = json.dumps(evaluation.missing_elements)
                
                cursor.execute(f"""
                    INSERT INTO {self.table_name} (
                        model_name, prompt, original_prompt, negative_prompt, image_path,
                        evaluation_score, aesthetic_reasoning, alignment_reasoning, overall_reasoning,
                        improvement_suggestions, detected_artifacts, artifact_reasoning,
                        main_subjects_present, missing_elements, timestamp, session_id,
                        regeneration_count, seed, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evaluation.model_name, evaluation.prompt, evaluation.original_prompt,
                    evaluation.negative_prompt, evaluation.image_path, evaluation.evaluation_score,
                    evaluation.aesthetic_reasoning, evaluation.alignment_reasoning,
                    evaluation.overall_reasoning, evaluation.improvement_suggestions,
                    detected_artifacts_json, evaluation.artifact_reasoning,
                    evaluation.main_subjects_present, missing_elements_json,
                    evaluation.timestamp, evaluation.session_id, evaluation.regeneration_count,
                    evaluation.seed, evaluation.confidence_score
                ))
                
                conn.commit()
                logger.info(f"Added evaluation to {self.model_name} global memory: score={evaluation.evaluation_score}")
                
        except sqlite3.Error as e:
            logger.error(f"Error adding evaluation to global memory: {e}")
            raise
            
    def get_recent_evaluations(self, limit: int = 10) -> List[EvaluationMemory]:
        """Get recent evaluations from global memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    SELECT * FROM {self.table_name}
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                evaluations = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    # Convert JSON strings back to lists
                    row_dict['detected_artifacts'] = json.loads(row_dict['detected_artifacts'])
                    row_dict['missing_elements'] = json.loads(row_dict['missing_elements'])
                    # Remove database-specific fields
                    row_dict.pop('id', None)
                    row_dict.pop('created_at', None)
                    
                    evaluations.append(EvaluationMemory.from_dict(row_dict))
                
                return evaluations
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving recent evaluations: {e}")
            return []
            
    def search_by_prompt(self, prompt_keywords: str, limit: int = 10) -> List[EvaluationMemory]:
        """Search evaluations by prompt keywords."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Use LIKE for simple text search
                search_pattern = f"%{prompt_keywords}%"
                cursor.execute(f"""
                    SELECT * FROM {self.table_name}
                    WHERE prompt LIKE ? OR original_prompt LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (search_pattern, search_pattern, limit))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                evaluations = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    # Convert JSON strings back to lists
                    row_dict['detected_artifacts'] = json.loads(row_dict['detected_artifacts'])
                    row_dict['missing_elements'] = json.loads(row_dict['missing_elements'])
                    # Remove database-specific fields
                    row_dict.pop('id', None)
                    row_dict.pop('created_at', None)
                    
                    evaluations.append(EvaluationMemory.from_dict(row_dict))
                
                return evaluations
                
        except sqlite3.Error as e:
            logger.error(f"Error searching evaluations: {e}")
            return []
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from global memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get basic stats
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_evaluations,
                        AVG(evaluation_score) as avg_score,
                        MIN(evaluation_score) as min_score,
                        MAX(evaluation_score) as max_score,
                        COUNT(DISTINCT session_id) as unique_sessions
                    FROM {self.table_name}
                """)
                basic_stats = cursor.fetchone()
                
                # Get score distribution
                cursor.execute(f"""
                    SELECT 
                        CASE 
                            WHEN evaluation_score >= 9 THEN 'Excellent (9-10)'
                            WHEN evaluation_score >= 7 THEN 'Good (7-8.9)'
                            WHEN evaluation_score >= 5 THEN 'Average (5-6.9)'
                            ELSE 'Poor (<5)'
                        END as score_range,
                        COUNT(*) as count
                    FROM {self.table_name}
                    GROUP BY score_range
                """)
                score_distribution = cursor.fetchall()
                
                # Get recent performance trend (last 10 evaluations)
                cursor.execute(f"""
                    SELECT evaluation_score FROM {self.table_name}
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                recent_scores = [row[0] for row in cursor.fetchall()]
                
                return {
                    "model_name": self.model_name,
                    "total_evaluations": basic_stats[0] or 0,
                    "average_score": round(basic_stats[1] or 0, 2),
                    "min_score": basic_stats[2] or 0,
                    "max_score": basic_stats[3] or 0,
                    "unique_sessions": basic_stats[4] or 0,
                    "score_distribution": dict(score_distribution),
                    "recent_scores": recent_scores,
                    "recent_average": round(sum(recent_scores) / len(recent_scores), 2) if recent_scores else 0
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "model_name": self.model_name,
                "total_evaluations": 0,
                "average_score": 0,
                "min_score": 0,
                "max_score": 0,
                "unique_sessions": 0,
                "score_distribution": {},
                "recent_scores": [],
                "recent_average": 0
            }

class LocalMemory:
    """Local memory for storing pattern analysis and adaptive prompts."""
    
    def __init__(self, model_name: str, db_path: str):
        self.model_name = model_name
        self.db_path = db_path
        self.logger = logger
        self.pattern_analyzer = PatternAnalyzer(model_name, db_path)
        self.prompt_generator = AdaptivePromptGenerator(model_name, db_path)
        self._init_database()
        
    def _init_database(self):
        """Initialize local memory database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create pattern analysis table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        analysis_period_start TEXT NOT NULL,
                        analysis_period_end TEXT NOT NULL,
                        total_evaluations_analyzed INTEGER NOT NULL,
                        common_artifacts TEXT,          -- JSON
                        frequent_missing_elements TEXT, -- JSON
                        low_score_patterns TEXT,        -- JSON
                        improvement_themes TEXT,        -- JSON
                        prompt_type_issues TEXT,        -- JSON
                        confidence_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create adaptive prompts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS adaptive_prompts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        pattern_analysis_id INTEGER,
                        positive_prompt_additions TEXT NOT NULL,
                        negative_prompt_additions TEXT NOT NULL,
                        target_flaws TEXT,             -- JSON
                        effectiveness_score REAL DEFAULT 0.0,
                        usage_count INTEGER DEFAULT 0,
                        last_used TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (pattern_analysis_id) REFERENCES pattern_analysis (id)
                    )
                """)
                
                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_pattern_analysis_model 
                    ON pattern_analysis (model_name, created_at)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_adaptive_prompts_model 
                    ON adaptive_prompts (model_name, effectiveness_score DESC)
                """)
                
                conn.commit()
                self.logger.info(f"Local memory database initialized for {self.model_name}")
                
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing local memory database: {e}")
            raise
    
    def check_and_trigger_analysis(self) -> bool:
        """Check if pattern analysis should be triggered and execute if needed."""
        try:
            if self.pattern_analyzer.should_trigger_analysis():
                self.logger.info(f"Triggering pattern analysis for {self.model_name}")
                
                # Perform pattern analysis
                analysis = self.pattern_analyzer.analyze_patterns()
                if analysis:
                    # Save pattern analysis
                    analysis_id = self._save_pattern_analysis(analysis)
                    
                    # Generate adaptive prompts
                    adaptive_prompt = self.prompt_generator.generate_adaptive_prompts(analysis)
                    if adaptive_prompt:
                        adaptive_prompt.pattern_analysis_id = analysis_id
                        self._save_adaptive_prompt(adaptive_prompt)
                        
                        self.logger.info(f"Pattern analysis and adaptive prompts generated for {self.model_name}")
                        return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis trigger: {e}")
            return False
    
    def get_active_adaptive_prompts(self) -> Optional[AdaptivePrompt]:
        """Get the most effective active adaptive prompts for the model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the most recent and effective adaptive prompt
                cursor.execute("""
                    SELECT * FROM adaptive_prompts
                    WHERE model_name = ? 
                    AND created_at > datetime('now', '-7 days')
                    ORDER BY effectiveness_score DESC, created_at DESC
                    LIMIT 1
                """, (self.model_name,))
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    row_dict = dict(zip(columns, row))
                    
                    # Convert JSON fields
                    row_dict['target_flaws'] = json.loads(row_dict['target_flaws'])
                    
                    # Use created_at as timestamp for AdaptivePrompt
                    row_dict['timestamp'] = row_dict['created_at']
                    
                    # Remove database-specific fields
                    row_dict.pop('id', None)
                    row_dict.pop('created_at', None)
                    
                    return AdaptivePrompt.from_dict(row_dict)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting adaptive prompts: {e}")
            return None
    
    def update_adaptive_prompt_effectiveness(self, adaptive_prompt_id: int, 
                                           evaluation_score: float, used: bool = True):
        """Update the effectiveness score of an adaptive prompt based on results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if used:
                    # Update usage count and last used
                    cursor.execute("""
                        UPDATE adaptive_prompts 
                        SET usage_count = usage_count + 1,
                            last_used = ?
                        WHERE id = ?
                    """, (datetime.now().isoformat(), adaptive_prompt_id))
                
                # Update effectiveness score using exponential moving average
                cursor.execute("""
                    SELECT effectiveness_score, usage_count FROM adaptive_prompts WHERE id = ?
                """, (adaptive_prompt_id,))
                
                result = cursor.fetchone()
                if result:
                    current_score, usage_count = result
                    
                    # Calculate new effectiveness score
                    if usage_count <= 1:
                        new_score = evaluation_score
                    else:
                        # Exponential moving average with decay factor
                        alpha = 0.3
                        new_score = alpha * evaluation_score + (1 - alpha) * current_score
                    
                    cursor.execute("""
                        UPDATE adaptive_prompts 
                        SET effectiveness_score = ?
                        WHERE id = ?
                    """, (round(new_score, 2), adaptive_prompt_id))
                    
                    conn.commit()
                    self.logger.debug(f"Updated adaptive prompt {adaptive_prompt_id} effectiveness: {new_score:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error updating adaptive prompt effectiveness: {e}")
    
    def get_pattern_analysis_history(self, limit: int = 5) -> List[PatternAnalysis]:
        """Get recent pattern analysis history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM pattern_analysis
                    WHERE model_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (self.model_name, limit))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                analyses = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    # Convert JSON fields
                    row_dict['common_artifacts'] = json.loads(row_dict['common_artifacts'])
                    row_dict['frequent_missing_elements'] = json.loads(row_dict['frequent_missing_elements'])
                    row_dict['low_score_patterns'] = json.loads(row_dict['low_score_patterns'])
                    row_dict['improvement_themes'] = json.loads(row_dict['improvement_themes'])
                    row_dict['prompt_type_issues'] = json.loads(row_dict['prompt_type_issues'])
                    
                    # Remove database-specific fields and rename timestamp
                    row_dict.pop('id', None)
                    row_dict['timestamp'] = row_dict.pop('created_at')
                    
                    analyses.append(PatternAnalysis.from_dict(row_dict))
                
                return analyses
                
        except Exception as e:
            self.logger.error(f"Error getting pattern analysis history: {e}")
            return []
    
    def _save_pattern_analysis(self, analysis: PatternAnalysis) -> int:
        """Save pattern analysis to database and return the ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO pattern_analysis (
                        model_name, analysis_period_start, analysis_period_end,
                        total_evaluations_analyzed, common_artifacts, frequent_missing_elements,
                        low_score_patterns, improvement_themes, prompt_type_issues, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.model_name, analysis.analysis_period_start, analysis.analysis_period_end,
                    analysis.total_evaluations_analyzed, json.dumps(analysis.common_artifacts),
                    json.dumps(analysis.frequent_missing_elements), json.dumps(analysis.low_score_patterns),
                    json.dumps(analysis.improvement_themes), json.dumps(analysis.prompt_type_issues),
                    analysis.confidence_score
                ))
                
                analysis_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Saved pattern analysis {analysis_id} for {self.model_name}")
                return analysis_id
                
        except Exception as e:
            self.logger.error(f"Error saving pattern analysis: {e}")
            return 0
    
    def _save_adaptive_prompt(self, adaptive_prompt: AdaptivePrompt) -> int:
        """Save adaptive prompt to database and return the ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO adaptive_prompts (
                        model_name, pattern_analysis_id, positive_prompt_additions,
                        negative_prompt_additions, target_flaws, effectiveness_score,
                        usage_count, last_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    adaptive_prompt.model_name, adaptive_prompt.pattern_analysis_id,
                    adaptive_prompt.positive_prompt_additions, adaptive_prompt.negative_prompt_additions,
                    json.dumps(adaptive_prompt.target_flaws), adaptive_prompt.effectiveness_score,
                    adaptive_prompt.usage_count, adaptive_prompt.last_used
                ))
                
                prompt_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Saved adaptive prompt {prompt_id} for {self.model_name}")
                return prompt_id
                
        except Exception as e:
            self.logger.error(f"Error saving adaptive prompt: {e}")
            return 0
    
    def add_local_data(self, data: Dict[str, Any]):
        """Add custom data to local memory (for future extensions)."""
        # This can be extended for other types of local memory data
        pass
        
    def get_local_data(self, session_id: str) -> List[Dict[str, Any]]:
        """Get local data for a session (for future extensions)."""
        # This can be extended for session-specific data retrieval
        return []

class MemoryManager:
    """Central manager for all model memory modules."""
    
    def __init__(self, db_path: str = "model_memories.db"):
        self.db_path = db_path
        self.models = {}
        
        # Initialize memory modules for each model
        self.qwen_image_memory = ModelMemoryModule("Qwen-Image", db_path)
        self.qwen_edit_memory = ModelMemoryModule("Qwen-Image-Edit", db_path)
        
        self.models["Qwen-Image"] = self.qwen_image_memory
        self.models["Qwen-Image-Edit"] = self.qwen_edit_memory
        
        logger.info("Memory manager initialized for all models")
        
    def get_model_memory(self, model_name: str) -> Optional[ModelMemoryModule]:
        """Get memory module for a specific model."""
        return self.models.get(model_name)
        
    def add_evaluation_memory(self, model_name: str, config: Any, session_id: str = None):
        """
        Add evaluation result to the appropriate model's global memory.
        
        Args:
            model_name: Name of the model ('Qwen-Image' or 'Qwen-Image-Edit')
            config: T2IConfig object containing evaluation data
            session_id: Optional session identifier
        """
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return
            
        try:
            current_config = config.get_current_config()
            
            # Create session ID if not provided
            if session_id is None:
                session_id = f"{int(time.time())}_{config.image_index or 0}"
            
            # Create evaluation memory object
            evaluation_memory = EvaluationMemory(
                model_name=model_name,
                prompt=current_config.get("generating_prompt", ""),
                original_prompt=config.prompt_understanding.get("original_prompt", ""),
                negative_prompt=current_config.get("negative_prompt", ""),
                image_path=current_config.get("gen_image_path", ""),
                evaluation_score=current_config.get("evaluation_score", 0.0),
                aesthetic_reasoning=current_config.get("aesthetic_reasoning", ""),
                alignment_reasoning=current_config.get("alignment_reasoning", ""),
                overall_reasoning=current_config.get("overall_reasoning", ""),
                improvement_suggestions=current_config.get("improvement_suggestions", ""),
                detected_artifacts=current_config.get("detected_artifacts", []),
                artifact_reasoning=current_config.get("artifact_reasoning", ""),
                main_subjects_present=current_config.get("main_subjects_present", True),
                missing_elements=current_config.get("missing_elements", []),
                timestamp=datetime.now().isoformat(),
                session_id=session_id,
                regeneration_count=config.regeneration_count,
                seed=config.seed or 0,
                confidence_score=current_config.get("confidence_score", 0.0)
            )
            
            self.models[model_name].add_evaluation_to_global_memory(evaluation_memory)
            logger.info(f"Added evaluation memory for {model_name}")
            
        except Exception as e:
            logger.error(f"Error adding evaluation memory: {e}")
            
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all models including pattern analysis."""
        summary = {}
        for model_name, memory_module in self.models.items():
            summary[model_name] = {
                "global_stats": memory_module.get_global_memory_stats(),
                "pattern_analysis": memory_module.get_pattern_analysis_summary()
            }
        return summary
    
    def get_adaptive_prompts_for_model(self, model_name: str) -> Optional[AdaptivePrompt]:
        """Get adaptive prompts for a specific model."""
        model_memory = self.get_model_memory(model_name)
        if model_memory:
            return model_memory.get_adaptive_prompts()
        return None
    
    def update_adaptive_prompt_effectiveness(self, model_name: str, evaluation_score: float, 
                                           adaptive_prompt_used: bool = True):
        """Update adaptive prompt effectiveness after evaluation."""
        model_memory = self.get_model_memory(model_name)
        if model_memory and adaptive_prompt_used:
            # For now, we'll track the latest adaptive prompt
            # In a production system, you'd want to track specific prompt IDs
            adaptive_prompts = model_memory.get_adaptive_prompts()
            if adaptive_prompts:
                # Update effectiveness (this is a simplified approach)
                model_memory.local_memory.update_adaptive_prompt_effectiveness(
                    1,  # Would be actual prompt ID in production
                    evaluation_score, 
                    adaptive_prompt_used
                )
        
    def export_memory_data(self, output_dir: str = "memory_exports"):
        """Export memory data for analysis."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for model_name, memory_module in self.models.items():
            try:
                # Get all evaluations
                evaluations = memory_module.get_recent_evaluations(limit=1000)  # Get a large number
                
                # Convert to list of dictionaries
                data = [eval_mem.to_dict() for eval_mem in evaluations]
                
                # Save as JSON
                output_file = os.path.join(output_dir, f"{model_name.replace('-', '_').lower()}_memory.json")
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                logger.info(f"Exported {len(data)} evaluations for {model_name} to {output_file}")
                
            except Exception as e:
                logger.error(f"Error exporting memory data for {model_name}: {e}")

# Global memory manager instance
memory_manager = None

def get_memory_manager(db_path: str = "model_memories.db") -> MemoryManager:
    """Get or create the global memory manager instance."""
    global memory_manager
    if memory_manager is None:
        memory_manager = MemoryManager(db_path)
    return memory_manager

def add_evaluation_to_memory(model_name: str, config: Any, session_id: str = None):
    """Convenience function to add evaluation to memory."""
    manager = get_memory_manager()
    manager.add_evaluation_memory(model_name, config, session_id)

def get_model_stats(model_name: str) -> Dict[str, Any]:
    """Convenience function to get model statistics."""
    manager = get_memory_manager()
    model_memory = manager.get_model_memory(model_name)
    if model_memory:
        return model_memory.get_global_memory_stats()
    return {}

def search_model_evaluations(model_name: str, prompt_keywords: str, limit: int = 10) -> List[EvaluationMemory]:
    """Convenience function to search model evaluations."""
    manager = get_memory_manager()
    model_memory = manager.get_model_memory(model_name)
    if model_memory:
        return model_memory.search_evaluations_by_prompt(prompt_keywords, limit)
    return []
