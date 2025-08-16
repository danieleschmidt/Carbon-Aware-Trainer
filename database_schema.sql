
-- Carbon-Aware-Trainer Database Schema
CREATE TABLE IF NOT EXISTS carbon_intensities (
    id SERIAL PRIMARY KEY,
    region VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    carbon_intensity FLOAT NOT NULL,
    renewable_percentage FLOAT,
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_region_timestamp (region, timestamp)
);

CREATE TABLE IF NOT EXISTS training_sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    region VARCHAR(10),
    total_steps INTEGER DEFAULT 0,
    total_energy_kwh FLOAT DEFAULT 0.0,
    total_carbon_kg FLOAT DEFAULT 0.0,
    model_name VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS carbon_forecasts (
    id SERIAL PRIMARY KEY,
    region VARCHAR(10) NOT NULL,
    forecast_time TIMESTAMP NOT NULL,
    horizon_hours INTEGER NOT NULL,
    predicted_intensity FLOAT NOT NULL,
    confidence_score FLOAT,
    model_version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS optimization_events (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    details JSONB,
    carbon_saved_kg FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_training_sessions_region ON training_sessions(region);
CREATE INDEX IF NOT EXISTS idx_carbon_forecasts_region_time ON carbon_forecasts(region, forecast_time);
CREATE INDEX IF NOT EXISTS idx_optimization_events_session ON optimization_events(session_id);

-- Partitioning for carbon_intensities by date
CREATE TABLE IF NOT EXISTS carbon_intensities_2024 PARTITION OF carbon_intensities
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Views for analytics
CREATE OR REPLACE VIEW carbon_savings_summary AS
SELECT 
    region,
    DATE(start_time) as date,
    COUNT(*) as sessions,
    SUM(total_carbon_kg) as total_carbon_kg,
    AVG(total_carbon_kg) as avg_carbon_per_session,
    SUM(CASE WHEN total_carbon_kg > 0 THEN 1 ELSE 0 END) as carbon_aware_sessions
FROM training_sessions 
WHERE start_time >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY region, DATE(start_time);
