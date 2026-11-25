"""Application entrypoint starting both the HTTP health server and Discord bot."""

from __future__ import annotations

import asyncio
import logging
import sys

from src.bot import DiscordGeminiBot
from src.config import BotConfig
from src.health import start_health_server


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


async def run() -> int:
    """Boot the health-check server and run the Discord bot until stopped."""
    config = BotConfig.from_env()
    if not config.discord_token or config.discord_token == "your_discord_bot_token_here":
        logger.error(
            "Thiếu DISCORD_TOKEN! Hãy tạo file .env từ .env.example (hoặc đặt biến môi trường DISCORD_TOKEN) trước khi chạy."
        )
        return 1

    bot = DiscordGeminiBot(config=config)
    runner = await start_health_server(port=config.port, status_provider=bot.get_status)
    logger.info("HTTP health-check server listening on 0.0.0.0:%d (/health)", config.port)

    try:
        async with bot:
            await bot.start(config.discord_token)
    finally:
        await runner.cleanup()
        logger.info("Shutdown complete.")
    return 0


def main() -> None:
    """CLI entrypoint."""
    try:
        exit_code = asyncio.run(run())
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
