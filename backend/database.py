import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import json

DATABASE_PATH = "experiments.db"

def init_database():
    """Initialize the SQLite database with experiments table."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT NOT NULL,
            user_label TEXT NOT NULL,
            image_path TEXT,
            file_size INTEGER,
            image_dimensions TEXT,
            
            -- Detection Results
            verdict TEXT,
            is_ai_generated BOOLEAN,
            confidence INTEGER,
            ai_score INTEGER,
            
            -- Branch Scores
            ggd_score INTEGER,
            local_consistency_score INTEGER,
            texture_score INTEGER,
            patchcraft_score INTEGER,
            spectral_score INTEGER,
            chromatic_aberration_score INTEGER,
            kurtosis_score INTEGER,
            ssp_adjustment INTEGER,
            
            -- Key Metrics
            ggd_shape REAL,
            ggd_scale REAL,
            noise_std REAL,
            beta_gaussian REAL,
            beta_nongaussian REAL,
            texture_variance REAL,
            patchcraft_energy REAL,
            spectral_frequency REAL,
            lca_displacement REAL,
            lca_absolute REAL,
            kurtosis_value REAL,
            ssp_beta REAL,
            
            -- AI Reasoning (if available)
            ai_verdict TEXT,
            ai_sub_category TEXT,
            ai_confidence_score INTEGER,
            ai_primary_evidence TEXT,
            ai_reasoning_chain TEXT,
            ai_explanation TEXT,
            
            -- Full JSON data for reference
            full_data TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")


def save_experiment(
    image_name: str,
    user_label: str,
    file_size: int,
    image_dimensions: tuple,
    detection_result: Dict,
    image_path: Optional[str] = None
) -> int:
    """
    Save an experiment to the database.
    
    Args:
        image_name: User-provided name for the image
        user_label: User's ground truth label (e.g., "AI", "Real", "Smartphone")
        file_size: Size of the uploaded file in bytes
        image_dimensions: Tuple of (width, height)
        detection_result: Full detection result dictionary
        image_path: Path to the saved image file
        
    Returns:
        The ID of the inserted record
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    metrics = detection_result.get("metrics", {})
    ai_reasoning = detection_result.get("ai_reasoning", {})
    
    # Extract reasoning chain as JSON string
    reasoning_chain = json.dumps(ai_reasoning.get("reasoning_chain", [])) if ai_reasoning else None
    
    cursor.execute("""
        INSERT INTO experiments (
            timestamp, image_name, user_label, image_path, file_size, image_dimensions,
            verdict, is_ai_generated, confidence, ai_score,
            ggd_score, local_consistency_score, texture_score, patchcraft_score,
            spectral_score, chromatic_aberration_score, kurtosis_score, ssp_adjustment,
            ggd_shape, ggd_scale, noise_std, beta_gaussian, beta_nongaussian,
            texture_variance, patchcraft_energy, spectral_frequency,
            lca_displacement, lca_absolute, kurtosis_value, ssp_beta,
            ai_verdict, ai_sub_category, ai_confidence_score,
            ai_primary_evidence, ai_reasoning_chain, ai_explanation,
            full_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        image_name,
        user_label,
        image_path,
        file_size,
        f"{image_dimensions[0]}x{image_dimensions[1]}",
        detection_result.get("verdict"),
        detection_result.get("is_ai_generated"),
        detection_result.get("confidence"),
        metrics.get("ai_score"),
        metrics.get("ggd_score"),
        metrics.get("local_consistency_score"),
        metrics.get("texture_score"),
        metrics.get("patchcraft_score"),
        metrics.get("spectral_score"),
        metrics.get("chromatic_aberration_score"),
        metrics.get("kurtosis_score"),
        metrics.get("ssp_adjustment"),
        metrics.get("ggd_shape"),
        metrics.get("ggd_scale"),
        metrics.get("noise_std"),
        metrics.get("beta_gaussian"),
        metrics.get("beta_nongaussian"),
        metrics.get("texture_variance"),
        metrics.get("patchcraft_energy"),
        metrics.get("spectral_frequency"),
        metrics.get("lca_displacement"),
        metrics.get("lca_absolute"),
        metrics.get("kurtosis_value"),
        metrics.get("ssp_beta"),
        ai_reasoning.get("verdict") if ai_reasoning else None,
        ai_reasoning.get("sub_category") if ai_reasoning else None,
        ai_reasoning.get("confidence_score") if ai_reasoning else None,
        ai_reasoning.get("primary_smoking_gun") if ai_reasoning else None,
        reasoning_chain,
        ai_reasoning.get("human_readable_explanation") if ai_reasoning else None,
        json.dumps(detection_result)
    ))
    
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return record_id


def get_all_experiments() -> List[Dict]:
    """Retrieve all experiments from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    experiments = [dict(row) for row in rows]
    conn.close()
    
    return experiments


def get_experiment_by_id(experiment_id: int) -> Optional[Dict]:
    """Retrieve a specific experiment by ID."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = cursor.fetchone()
    
    conn.close()
    
    return dict(row) if row else None


def get_experiments_by_label(user_label: str) -> List[Dict]:
    """Retrieve all experiments with a specific label."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM experiments WHERE user_label = ? ORDER BY timestamp DESC", (user_label,))
    rows = cursor.fetchall()
    
    experiments = [dict(row) for row in rows]
    conn.close()
    
    return experiments


def get_experiment_statistics() -> Dict:
    """Get summary statistics from all experiments."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_experiments,
            COUNT(DISTINCT user_label) as unique_labels,
            AVG(ai_score) as avg_ai_score,
            AVG(confidence) as avg_confidence,
            SUM(CASE WHEN is_ai_generated = 1 THEN 1 ELSE 0 END) as detected_as_ai,
            SUM(CASE WHEN is_ai_generated = 0 THEN 1 ELSE 0 END) as detected_as_real,
            AVG(ggd_shape) as avg_ggd_shape,
            AVG(lca_displacement) as avg_lca
        FROM experiments
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    return {
        "total_experiments": row[0] or 0,
        "unique_labels": row[1] or 0,
        "avg_ai_score": round(row[2], 2) if row[2] else 0,
        "avg_confidence": round(row[3], 2) if row[3] else 0,
        "detected_as_ai": row[4] or 0,
        "detected_as_real": row[5] or 0,
        "avg_ggd_shape": round(row[6], 3) if row[6] else 0,
        "avg_lca": round(row[7], 3) if row[7] else 0
    }


# Initialize database on module import
init_database()
