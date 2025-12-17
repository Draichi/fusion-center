"""
Event Correlation Engine Server - main entry point.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from src.correlation_engine.alerts import AlertGenerator
from src.correlation_engine.collector import EventCollector
from src.correlation_engine.config import ECE_DB_PATH, POLLING_INTERVALS
from src.correlation_engine.notifications import EmailNotifier
from src.correlation_engine.patterns import PatternDetector
from src.correlation_engine.rules import CorrelationRulesEngine
from src.correlation_engine.schemas import Watchlist
from src.correlation_engine.storage import EventStorage
from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


class EventCorrelationEngine:
    """Main Event Correlation Engine."""
    
    def __init__(
        self,
        mcp_server_url: str | None = None,
        db_path: Path | None = None,
        watchlist: Watchlist | None = None,
    ):
        """Initialize the Event Correlation Engine.
        
        Args:
            mcp_server_url: MCP server URL (defaults to settings)
            db_path: Database path (defaults to config)
            watchlist: Optional watchlist for filtering events
        """
        self.mcp_server_url = mcp_server_url or f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"
        self.watchlist = watchlist
        
        # Initialize components
        self.storage = EventStorage(db_path or ECE_DB_PATH)
        self.collector = EventCollector(self.mcp_server_url, watchlist)
        self.rules_engine = CorrelationRulesEngine()
        self.pattern_detector = PatternDetector()
        self.email_notifier = EmailNotifier()
        self.alert_generator = AlertGenerator(self.storage, self.email_notifier)
        
        self.running = False
        self.iteration = 0  # Track collection cycle iterations
        
        logger.info("=" * 60)
        logger.info("Event Correlation Engine initialized")
        logger.info(f"MCP Server: {self.mcp_server_url}")
        logger.info(f"Database: {self.storage.db_path}")
        if watchlist:
            logger.info(f"Watchlist: {watchlist.name}")
        logger.info("=" * 60)
    
    async def start(self):
        """Start the correlation engine (continuous operation)."""
        self.running = True
        logger.info("🚀 Starting Event Correlation Engine...")
        
        try:
            while self.running:
                self.iteration += 1
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Iteration #{self.iteration} - {datetime.now(datetime.UTC).isoformat()}")
                logger.info(f"{'=' * 60}")
                
                await self._run_collection_cycle()
                
                # Wait before next cycle (use minimum polling interval)
                min_interval_hours = min(POLLING_INTERVALS.values())
                wait_seconds = min_interval_hours * 3600
                
                logger.info(f"\n⏳ Waiting {min_interval_hours}h until next collection cycle...")
                await asyncio.sleep(wait_seconds)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Received shutdown signal")
        finally:
            self.stop()
    
    async def _run_collection_cycle(self):
        """Run a single collection and correlation cycle."""
        try:
            # Step 1: Collect events
            logger.info("📊 Step 1: Collecting events from all sources...")
            events = await self.collector.collect_all_sources()
            
            if not events:
                logger.info("No events collected this cycle")
                return
            
            # Step 2: Store events
            logger.info(f"💾 Step 2: Storing {len(events)} events...")
            for event in events:
                self.storage.insert_event(event)
            
            # Step 3: Get recent events for correlation
            logger.info("🔍 Step 3: Retrieving recent events for correlation...")
            recent_events = self.storage.get_events(hours_back=24)
            logger.info(f"Analyzing {len(recent_events)} recent events")
            
            # Step 4: Find correlations
            logger.info("🔗 Step 4: Finding correlations...")
            correlations = self.rules_engine.find_correlations(recent_events)
            logger.info(f"Found {len(correlations)} potential correlations")
            
            # Step 5: Validate correlations with ML/LLM
            logger.info("🤖 Step 5: Validating correlations with LLM...")
            validated_correlations = []
            
            for correlation in correlations:
                # Get events for this correlation
                corr_events = [
                    e for e in recent_events 
                    if e["event_id"] in correlation.event_ids
                ]
                
                # Validate with LLM
                is_valid, reason = await self.pattern_detector.validate_correlation(
                    correlation,
                    corr_events
                )
                
                if is_valid:
                    validated_correlations.append(correlation)
                else:
                    logger.debug(f"Correlation {correlation.correlation_id} rejected by LLM: {reason}")
            
            logger.info(f"✅ {len(validated_correlations)} correlations validated")
            
            # Step 6: Detect patterns
            logger.info("📈 Step 6: Detecting patterns...")
            anomalies = self.pattern_detector.detect_anomalies(recent_events)
            escalation = self.pattern_detector.detect_escalation_pattern(recent_events)
            
            if anomalies["anomalies_detected"]:
                logger.info(f"⚠️  Detected {len(anomalies['anomalies'])} anomalies")
            
            if escalation["escalation_detected"]:
                logger.warning("⚠️  Escalation pattern detected!")
            
            # Step 7: Generate alerts
            logger.info("🚨 Step 7: Generating alerts...")
            alerts_generated = 0
            
            for correlation in validated_correlations:
                if self.alert_generator.process_correlation(correlation):
                    alerts_generated += 1
            
            logger.info(f"Generated {alerts_generated} alerts")
            
            # Step 8: Cleanup old data
            if self.iteration % 24 == 0:  # Once per day
                logger.info("🧹 Step 8: Cleaning up old data...")
                self.storage.cleanup_old_data()
            
            logger.info("✨ Cycle completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in collection cycle: {e}", exc_info=True)
    
    def stop(self):
        """Stop the correlation engine."""
        self.running = False
        logger.info("🛑 Event Correlation Engine stopped")
    
    async def run_once(self):
        """Run a single collection cycle (for testing)."""
        logger.info("Running single collection cycle...")
        await self._run_collection_cycle()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Event Correlation Engine")
    parser.add_argument(
        "--mcp-url",
        type=str,
        help=f"MCP server URL (default: http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse)"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help=f"Database path (default: {ECE_DB_PATH})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create engine
    engine = EventCorrelationEngine(
        mcp_server_url=args.mcp_url,
        db_path=args.db_path
    )
    
    # Run
    if args.once:
        await engine.run_once()
    else:
        await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
