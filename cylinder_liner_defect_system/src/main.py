#!/usr/bin/env python3
"""
缸套缺陷分级评估与智能分选系统 - 主程序
Cylinder Liner Defect Grading and Intelligent Sorting System - Main Application
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from api.mes_interface import MESInterface
from api.web_server import WebServer
from core.config_manager import ConfigManager
from core.system_manager import SystemManager
from utils.logger import setup_logging


class CylinderDefectSystem:
    """缸套缺陷检测系统主类"""

    def __init__(self, config_path: str = 'config/system_config.yaml'):
        """初始化系统"""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()

        # 设置日志
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)

        # 初始化系统组件
        self.system_manager = SystemManager(self.config)
        self.web_server = WebServer(self.config.get('web_server', {}))
        self.mes_interface = MESInterface(self.config.get('mes', {}))

        # 系统状态
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f'接收到信号 {signum}，开始优雅关闭...')
        self.shutdown_event.set()

    async def start(self):
        """启动系统"""
        try:
            self.logger.info('正在启动缸套缺陷检测系统...')

            # 启动系统管理器
            await self.system_manager.start()

            # 启动Web服务器
            await self.web_server.start()

            # 启动MES接口
            await self.mes_interface.start()

            self.is_running = True
            self.logger.info('系统启动完成')

            # 等待关闭信号
            await self.shutdown_event.wait()

        except Exception as e:
            self.logger.error(f'系统启动失败: {e}')
            raise

    async def stop(self):
        """停止系统"""
        if not self.is_running:
            return

        self.logger.info('正在关闭系统...')

        try:
            # 停止各个组件
            await self.mes_interface.stop()
            await self.web_server.stop()
            await self.system_manager.stop()

            self.is_running = False
            self.logger.info('系统已安全关闭')

        except Exception as e:
            self.logger.error(f'系统关闭时发生错误: {e}')

    async def run(self):
        """运行系统"""
        try:
            await self.start()
        finally:
            await self.stop()


async def main():
    """主函数"""
    # 检查配置文件
    config_path = 'config/system_config.yaml'
    if not Path(config_path).exists():
        print(f'配置文件不存在: {config_path}')
        sys.exit(1)

    # 创建并运行系统
    system = CylinderDefectSystem(config_path)

    try:
        await system.run()
    except KeyboardInterrupt:
        print('\n用户中断，正在关闭系统...')
    except Exception as e:
        print(f'系统运行错误: {e}')
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
