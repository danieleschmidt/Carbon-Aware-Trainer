
# Disaster Recovery Plan for Carbon-Aware-Trainer

## 1. Backup Strategy
- Database backups: Every 6 hours to S3 with cross-region replication
- Configuration backups: Git-based with ArgoCD sync
- Metrics/logs: 30-day retention in each region

## 2. Failover Procedures
- Primary region failure: Automatic DNS failover within 5 minutes
- Cross-region database sync: Maximum 1-hour RPO
- State reconstruction: Training sessions resume from last checkpoint

## 3. Carbon-Aware Recovery
- Prioritize recovery in lowest carbon regions
- Temporary scaling in clean energy regions during outages
- Carbon cost tracking during DR scenarios

## 4. Testing Schedule
- Monthly DR drills in staging environment
- Quarterly cross-region failover tests
- Annual full disaster simulation

## 5. Communication Plan
- Automated alerts via PagerDuty, Slack, email
- Status page updates at status.carbon-trainer.terragonlabs.com
- Customer notifications for >15 min outages
