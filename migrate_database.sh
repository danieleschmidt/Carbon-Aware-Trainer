#!/bin/bash
# Database migration for Carbon-Aware-Trainer

set -e

echo "Starting database migration..."

# Check if database exists
if ! psql -lqt | cut -d \| -f 1 | grep -qw carbon_trainer; then
    echo "Creating database..."
    createdb carbon_trainer
fi

# Run schema migration
echo "Applying schema..."
psql -d carbon_trainer -f database_schema.sql

# Insert sample data
echo "Inserting sample data..."
psql -d carbon_trainer << EOF
INSERT INTO carbon_intensities (region, timestamp, carbon_intensity, renewable_percentage, data_source)
VALUES 
    ('US-CA', NOW(), 85.0, 65.0, 'sample'),
    ('US-WA', NOW(), 45.0, 85.0, 'sample'),
    ('EU-FR', NOW(), 65.0, 75.0, 'sample'),
    ('EU-NO', NOW(), 25.0, 95.0, 'sample');

INSERT INTO carbon_forecasts (region, forecast_time, horizon_hours, predicted_intensity, confidence_score, model_version)
VALUES
    ('US-CA', NOW() + INTERVAL '1 hour', 1, 80.0, 0.85, 'v1.0'),
    ('US-WA', NOW() + INTERVAL '1 hour', 1, 40.0, 0.90, 'v1.0'),
    ('EU-FR', NOW() + INTERVAL '1 hour', 1, 60.0, 0.80, 'v1.0'),
    ('EU-NO', NOW() + INTERVAL '1 hour', 1, 20.0, 0.95, 'v1.0');
EOF

echo "Database migration completed successfully!"
