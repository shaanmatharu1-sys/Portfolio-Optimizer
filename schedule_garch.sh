#!/bin/bash
# Setup cron job to run GARCH update daily at 10 PM

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UPDATE_SCRIPT="$SCRIPT_DIR/update_garch.py"

# Make update script executable
chmod +x "$UPDATE_SCRIPT"

# Create cron job entry (runs at 10 PM daily)
CRON_ENTRY="0 22 * * * cd '$SCRIPT_DIR' && /usr/bin/python3 '$UPDATE_SCRIPT' >> '$SCRIPT_DIR/garch_cron.log' 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "update_garch.py"; then
    echo "GARCH update cron job already exists"
    echo "Current cron jobs:"
    crontab -l | grep "update_garch.py"
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "✓ GARCH update cron job installed successfully"
    echo "Runs daily at 10:00 PM"
    echo ""
    echo "To verify:"
    echo "  crontab -l"
    echo ""
    echo "To remove:"
    echo "  crontab -l | grep -v 'update_garch.py' | crontab -"
fi

echo ""
echo "Log files:"
echo "  Main log: $SCRIPT_DIR/garch_updates.log"
echo "  Cron log: $SCRIPT_DIR/garch_cron.log"
