"""
系统管理器 - 负责协调各个子系统的运行
System Manager - Coordinates the operation of all subsystems
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from ..utils.event_bus import EventBus
from ..utils.performance_monitor import PerformanceMonitor
from .data_acquisition import DataAcquisitionSystem
from .inference_engine import InferenceEngine
from .quality_manager import QualityManager
from .sorting_controller import SortingController


class SystemManager:
    """系统管理器主类"""

    def __init__(self, config: dict[str, Any]):
        """初始化系统管理器"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化事件总线
        self.event_bus = EventBus()

        # 初始化性能监控
        self.performance_monitor = PerformanceMonitor(config.get('monitoring', {}))

        # 初始化各个子系统
        self.data_acquisition = DataAcquisitionSystem(
            config.get('hardware', {}), self.event_bus
        )

        self.inference_engine = InferenceEngine(
            config.get('ai_models', {}), config.get('inference', {}), self.event_bus
        )

        self.sorting_controller = SortingController(
            config.get('sorting_control', {}), self.event_bus
        )

        self.quality_manager = QualityManager(
            config.get('grading_rules', {}), self.event_bus
        )

        # 系统状态
        self.is_running = False
        self.current_part_id = None
        self.processing_queue = asyncio.Queue(maxsize=100)

        # 注册事件处理器
        self._register_event_handlers()

        # 统计信息
        self.stats = {
            'total_processed': 0,
            'qualified_count': 0,
            'minor_count': 0,
            'major_count': 0,
            'reject_count': 0,
            'start_time': None,
            'last_processing_time': None,
        }

    def _register_event_handlers(self):
        """注册事件处理器"""
        # 数据采集完成事件
        self.event_bus.subscribe('data_acquired', self._on_data_acquired)

        # 推理完成事件
        self.event_bus.subscribe('inference_completed', self._on_inference_completed)

        # 分选完成事件
        self.event_bus.subscribe('sorting_completed', self._on_sorting_completed)

        # 错误事件
        self.event_bus.subscribe('system_error', self._on_system_error)

        # 性能警告事件
        self.event_bus.subscribe('performance_warning', self._on_performance_warning)

    async def start(self):
        """启动系统"""
        try:
            self.logger.info('正在启动系统管理器...')

            # 启动性能监控
            await self.performance_monitor.start()

            # 启动各个子系统
            await self.data_acquisition.start()
            await self.inference_engine.start()
            await self.sorting_controller.start()
            await self.quality_manager.start()

            # 启动主处理循环
            self.processing_task = asyncio.create_task(self._main_processing_loop())

            self.is_running = True
            self.stats['start_time'] = datetime.now()

            self.logger.info('系统管理器启动完成')

        except Exception as e:
            self.logger.error(f'系统管理器启动失败: {e}')
            raise

    async def stop(self):
        """停止系统"""
        if not self.is_running:
            return

        self.logger.info('正在停止系统管理器...')

        try:
            # 停止主处理循环
            if hasattr(self, 'processing_task'):
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            # 停止各个子系统
            await self.quality_manager.stop()
            await self.sorting_controller.stop()
            await self.inference_engine.stop()
            await self.data_acquisition.stop()

            # 停止性能监控
            await self.performance_monitor.stop()

            self.is_running = False
            self.logger.info('系统管理器已停止')

        except Exception as e:
            self.logger.error(f'系统管理器停止时发生错误: {e}')

    async def _main_processing_loop(self):
        """主处理循环"""
        self.logger.info('主处理循环已启动')

        while self.is_running:
            try:
                # 等待新的处理任务
                task = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)

                # 处理任务
                await self._process_part(task)

            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                self.logger.error(f'主处理循环错误: {e}')
                await asyncio.sleep(1)

    async def _process_part(self, task: dict[str, Any]):
        """处理单个零件"""
        part_id = task['part_id']
        self.current_part_id = part_id

        start_time = datetime.now()
        self.logger.info(f'开始处理零件: {part_id}')

        try:
            # 1. 数据采集
            self.logger.debug(f'[{part_id}] 开始数据采集')
            acquisition_result = await self.data_acquisition.acquire_data(part_id)

            if not acquisition_result['success']:
                raise Exception(f'数据采集失败: {acquisition_result["error"]}')

            # 2. AI推理
            self.logger.debug(f'[{part_id}] 开始AI推理')
            inference_result = await self.inference_engine.process(
                acquisition_result['data']
            )

            if not inference_result['success']:
                raise Exception(f'AI推理失败: {inference_result["error"]}')

            # 3. 质量评估
            self.logger.debug(f'[{part_id}] 开始质量评估')
            quality_result = await self.quality_manager.evaluate_quality(
                part_id, inference_result['data']
            )

            # 4. 分选控制
            self.logger.debug(f'[{part_id}] 开始分选控制')
            await self.sorting_controller.execute_sorting(
                part_id, quality_result['grade']
            )

            # 5. 更新统计信息
            self._update_statistics(quality_result['grade'])

            # 6. 记录处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['last_processing_time'] = processing_time

            # 7. 发送完成事件
            await self.event_bus.publish(
                'part_processing_completed',
                {
                    'part_id': part_id,
                    'grade': quality_result['grade'],
                    'processing_time': processing_time,
                    'success': True,
                },
            )

            self.logger.info(
                f'零件 {part_id} 处理完成，等级: {quality_result["grade"]}, '
                f'耗时: {processing_time:.2f}秒'
            )

        except Exception as e:
            self.logger.error(f'处理零件 {part_id} 时发生错误: {e}')

            # 发送错误事件
            await self.event_bus.publish(
                'part_processing_failed',
                {
                    'part_id': part_id,
                    'error': str(e),
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                },
            )

        finally:
            self.current_part_id = None

    def _update_statistics(self, grade: str):
        """更新统计信息"""
        self.stats['total_processed'] += 1

        if grade == 'qualified':
            self.stats['qualified_count'] += 1
        elif grade == 'minor':
            self.stats['minor_count'] += 1
        elif grade == 'major':
            self.stats['major_count'] += 1
        elif grade == 'reject':
            self.stats['reject_count'] += 1

    async def _on_data_acquired(self, event_data: dict[str, Any]):
        """数据采集完成事件处理"""
        part_id = event_data['part_id']
        self.logger.debug(f'数据采集完成: {part_id}')

        # 记录性能指标
        await self.performance_monitor.record_metric(
            'data_acquisition_time', event_data.get('acquisition_time', 0)
        )

    async def _on_inference_completed(self, event_data: dict[str, Any]):
        """推理完成事件处理"""
        part_id = event_data['part_id']
        self.logger.debug(f'推理完成: {part_id}')

        # 记录性能指标
        await self.performance_monitor.record_metric(
            'inference_time', event_data.get('inference_time', 0)
        )

    async def _on_sorting_completed(self, event_data: dict[str, Any]):
        """分选完成事件处理"""
        part_id = event_data['part_id']
        grade = event_data['grade']
        self.logger.debug(f'分选完成: {part_id}, 等级: {grade}')

    async def _on_system_error(self, event_data: dict[str, Any]):
        """系统错误事件处理"""
        error_type = event_data.get('error_type', 'unknown')
        error_message = event_data.get('message', '')

        self.logger.error(f'系统错误 [{error_type}]: {error_message}')

        # 根据错误类型采取相应措施
        if error_type == 'critical':
            self.logger.critical('检测到严重错误，系统将进入安全模式')
            await self._enter_safe_mode()

    async def _on_performance_warning(self, event_data: dict[str, Any]):
        """性能警告事件处理"""
        metric = event_data.get('metric', '')
        value = event_data.get('value', 0)
        threshold = event_data.get('threshold', 0)

        self.logger.warning(f'性能警告 - {metric}: {value} 超过阈值 {threshold}')

    async def _enter_safe_mode(self):
        """进入安全模式"""
        self.logger.info('系统进入安全模式')

        # 停止数据采集
        await self.data_acquisition.pause()

        # 停止分选控制
        await self.sorting_controller.pause()

        # 发送安全模式事件
        await self.event_bus.publish(
            'safe_mode_activated',
            {'timestamp': datetime.now(), 'reason': 'critical_error'},
        )

    async def trigger_part_processing(self, part_id: str):
        """触发零件处理"""
        if not self.is_running:
            raise RuntimeError('系统未运行')

        task = {'part_id': part_id, 'timestamp': datetime.now()}

        try:
            await asyncio.wait_for(self.processing_queue.put(task), timeout=5.0)
            self.logger.info(f'零件 {part_id} 已加入处理队列')
        except asyncio.TimeoutError:
            raise RuntimeError('处理队列已满，无法添加新任务')

    def get_system_status(self) -> dict[str, Any]:
        """获取系统状态"""
        uptime = None
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()

        return {
            'is_running': self.is_running,
            'current_part_id': self.current_part_id,
            'queue_size': self.processing_queue.qsize(),
            'uptime': uptime,
            'statistics': self.stats.copy(),
            'subsystem_status': {
                'data_acquisition': self.data_acquisition.get_status(),
                'inference_engine': self.inference_engine.get_status(),
                'sorting_controller': self.sorting_controller.get_status(),
                'quality_manager': self.quality_manager.get_status(),
            },
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """获取性能指标"""
        return self.performance_monitor.get_metrics()

    async def update_grading_rules(self, new_rules: dict[str, Any]):
        """更新分级规则"""
        await self.quality_manager.update_rules(new_rules)
        self.logger.info('分级规则已更新')

    async def calibrate_system(self):
        """系统标定"""
        self.logger.info('开始系统标定...')

        # 标定数据采集系统
        await self.data_acquisition.calibrate()

        # 标定推理引擎
        await self.inference_engine.calibrate()

        self.logger.info('系统标定完成')
