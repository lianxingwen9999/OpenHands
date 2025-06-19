#!/usr/bin/env python3
"""
缸套缺陷分级评估与智能分选系统演示程序
Cylinder Liner Defect Grading and Intelligent Sorting System Demo
"""

import asyncio
import json
import logging
import time
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 模拟导入（实际使用时需要真实的模块）
class MockDataAcquisition:
    """模拟数据采集系统"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def capture_multimodal_data(self, part_id: str) -> dict[str, Any]:
        """模拟多模态数据采集"""
        self.logger.info(f'正在采集零件 {part_id} 的多模态数据...')

        # 模拟采集延时
        await asyncio.sleep(1.5)

        # 生成模拟的2D图像数据
        image_2d = self._generate_mock_image()

        # 生成模拟的3D点云数据
        pointcloud_3d = self._generate_mock_pointcloud()

        return {
            'part_id': part_id,
            'timestamp': time.time(),
            'image_2d': image_2d,
            'pointcloud_3d': pointcloud_3d,
            'acquisition_time': 1.5,
            'success': True,
        }

    def _generate_mock_image(self) -> np.ndarray:
        """生成模拟图像"""
        # 创建512x512的模拟缸套图像
        image = np.ones((512, 512, 3), dtype=np.uint8) * 200

        # 添加一些模拟缺陷
        # 划伤
        cv2.line(image, (100, 100), (200, 150), (50, 50, 50), 3)

        # 磕碰
        cv2.circle(image, (300, 200), 15, (80, 80, 80), -1)

        # 锈蚀区域
        cv2.ellipse(image, (400, 350), (20, 30), 0, 0, 360, (120, 100, 80), -1)

        return image

    def _generate_mock_pointcloud(self) -> np.ndarray:
        """生成模拟点云"""
        # 创建圆柱形点云
        n_points = 10000
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        z = np.random.uniform(0, 100, n_points)  # 高度100mm
        r = 50 + np.random.normal(0, 0.1, n_points)  # 半径50mm，加噪声

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # 添加一些缺陷（深度变化）
        defect_mask = (theta > 1.0) & (theta < 1.5) & (z > 30) & (z < 40)
        r[defect_mask] -= 2.0  # 2mm深的缺陷

        pointcloud = np.column_stack([x, y, z])
        return pointcloud


class MockInferenceEngine:
    """模拟推理引擎"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.defect_types = ['scratch', 'dent', 'corrosion', 'inclusion', 'hole']
        self.grade_types = ['qualified', 'minor', 'major', 'reject']

    async def process_multimodal_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """模拟多模态数据处理"""
        part_id = data['part_id']
        self.logger.info(f'正在处理零件 {part_id} 的AI推理...')

        # 模拟推理延时
        await asyncio.sleep(2.0)

        # 模拟检测结果
        defects = self._simulate_defect_detection(
            data['image_2d'], data['pointcloud_3d']
        )

        # 模拟等级评估
        overall_grade = self._simulate_grade_assessment(defects)

        return {
            'part_id': part_id,
            'defects': defects,
            'overall_grade': overall_grade,
            'confidence': 0.95,
            'inference_time': 2.0,
            'success': True,
        }

    def _simulate_defect_detection(
        self, image: np.ndarray, pointcloud: np.ndarray
    ) -> list[dict[str, Any]]:
        """模拟缺陷检测"""
        defects = [
            {
                'type': 'scratch',
                'location': [150, 125],
                'bbox': [100, 100, 100, 50],
                'confidence': 0.92,
                'measurements': {
                    'length': 8.5,  # mm
                    'width': 0.3,  # mm
                    'depth': 0.15,  # mm
                },
            },
            {
                'type': 'dent',
                'location': [300, 200],
                'bbox': [285, 185, 30, 30],
                'confidence': 0.88,
                'measurements': {
                    'area': 7.2,  # mm²
                    'depth': 0.8,  # mm
                },
            },
            {
                'type': 'corrosion',
                'location': [400, 350],
                'bbox': [380, 320, 40, 60],
                'confidence': 0.85,
                'measurements': {
                    'area': 12.5,  # mm²
                    'depth': 0.05,  # mm
                },
            },
        ]

        return defects

    def _simulate_grade_assessment(self, defects: list[dict[str, Any]]) -> str:
        """模拟等级评估"""
        # 简单的规则：根据缺陷类型和尺寸判断等级
        max_severity = 'qualified'

        for defect in defects:
            if defect['type'] == 'scratch':
                if defect['measurements']['length'] > 10:
                    max_severity = 'reject'
                elif defect['measurements']['length'] > 5:
                    max_severity = 'major'
                elif defect['measurements']['length'] > 2:
                    max_severity = 'minor'

            elif defect['type'] == 'dent':
                if defect['measurements']['area'] > 10:
                    max_severity = 'major'
                elif defect['measurements']['area'] > 4:
                    max_severity = 'minor'

            elif defect['type'] == 'corrosion':
                if defect['measurements']['area'] > 10:
                    max_severity = 'minor'

        return max_severity


class MockSortingController:
    """模拟分选控制器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sorting_channels = {'qualified': 1, 'minor': 2, 'major': 3, 'reject': 4}

    async def execute_sorting(self, part_id: str, grade: str) -> dict[str, Any]:
        """模拟分选执行"""
        self.logger.info(f'正在执行零件 {part_id} 的分选，等级: {grade}')

        # 模拟分选延时
        await asyncio.sleep(0.5)

        channel = self.sorting_channels.get(grade, 4)

        return {
            'part_id': part_id,
            'grade': grade,
            'channel': channel,
            'sorting_time': 0.5,
            'success': True,
        }


class SystemDemo:
    """系统演示主类"""

    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # 初始化模拟组件
        self.data_acquisition = MockDataAcquisition()
        self.inference_engine = MockInferenceEngine()
        self.sorting_controller = MockSortingController()

        # 统计信息
        self.stats = {
            'total_processed': 0,
            'qualified': 0,
            'minor': 0,
            'major': 0,
            'reject': 0,
            'total_time': 0,
        }

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('demo.log')],
        )

    async def process_single_part(self, part_id: str) -> dict[str, Any]:
        """处理单个零件的完整流程"""
        start_time = time.time()

        try:
            self.logger.info(f'开始处理零件: {part_id}')

            # 1. 数据采集
            acquisition_result = await self.data_acquisition.capture_multimodal_data(
                part_id
            )

            # 2. AI推理
            inference_result = await self.inference_engine.process_multimodal_data(
                acquisition_result
            )

            # 3. 分选控制
            sorting_result = await self.sorting_controller.execute_sorting(
                part_id, inference_result['overall_grade']
            )

            # 4. 统计更新
            processing_time = time.time() - start_time
            self._update_stats(inference_result['overall_grade'], processing_time)

            result = {
                'part_id': part_id,
                'grade': inference_result['overall_grade'],
                'defects': inference_result['defects'],
                'confidence': inference_result['confidence'],
                'processing_time': processing_time,
                'channel': sorting_result['channel'],
                'success': True,
            }

            self.logger.info(
                f'零件 {part_id} 处理完成 - 等级: {result["grade"]}, '
                f'耗时: {processing_time:.2f}秒, 通道: {result["channel"]}'
            )

            return result

        except Exception as e:
            self.logger.error(f'处理零件 {part_id} 时发生错误: {e}')
            return {'part_id': part_id, 'success': False, 'error': str(e)}

    def _update_stats(self, grade: str, processing_time: float):
        """更新统计信息"""
        self.stats['total_processed'] += 1
        self.stats[grade] += 1
        self.stats['total_time'] += processing_time

    async def run_batch_demo(self, num_parts: int = 10):
        """运行批量演示"""
        self.logger.info(f'开始批量演示，处理 {num_parts} 个零件')

        results = []

        for i in range(num_parts):
            part_id = f'CL_{int(time.time())}_{i + 1:03d}'
            result = await self.process_single_part(part_id)
            results.append(result)

            # 模拟生产间隔
            await asyncio.sleep(0.5)

        # 生成报告
        self._generate_demo_report(results)

        return results

    def _generate_demo_report(self, results: list[dict[str, Any]]):
        """生成演示报告"""
        self.logger.info('生成演示报告...')

        # 控制台报告
        print('\n' + '=' * 60)
        print('缸套缺陷检测系统演示报告')
        print('=' * 60)
        print(f'总处理数量: {self.stats["total_processed"]}')
        print(f'合格品: {self.stats["qualified"]}')
        print(f'轻微缺陷: {self.stats["minor"]}')
        print(f'严重缺陷: {self.stats["major"]}')
        print(f'报废品: {self.stats["reject"]}')
        print(
            f'平均处理时间: {self.stats["total_time"] / self.stats["total_processed"]:.2f}秒'
        )
        print(
            f'合格率: {self.stats["qualified"] / self.stats["total_processed"] * 100:.1f}%'
        )
        print('=' * 60)

        # 保存详细结果
        with open('demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(
                {'statistics': self.stats, 'detailed_results': results},
                f,
                indent=2,
                ensure_ascii=False,
            )

        # 生成可视化图表
        self._create_visualization()

    def _create_visualization(self):
        """创建可视化图表"""
        # 等级分布饼图
        grades = ['qualified', 'minor', 'major', 'reject']
        counts = [self.stats[grade] for grade in grades]
        colors = ['green', 'yellow', 'orange', 'red']

        plt.figure(figsize=(12, 5))

        # 饼图
        plt.subplot(1, 2, 1)
        plt.pie(counts, labels=grades, colors=colors, autopct='%1.1f%%')
        plt.title('缺陷等级分布')

        # 柱状图
        plt.subplot(1, 2, 2)
        plt.bar(grades, counts, color=colors)
        plt.title('各等级数量统计')
        plt.ylabel('数量')

        plt.tight_layout()
        plt.savefig('demo_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        self.logger.info('可视化图表已保存为 demo_visualization.png')

    async def run_realtime_demo(self, duration_minutes: int = 5):
        """运行实时演示"""
        self.logger.info(f'开始实时演示，持续 {duration_minutes} 分钟')

        start_time = time.time()
        end_time = start_time + duration_minutes * 60

        part_counter = 1

        while time.time() < end_time:
            part_id = f'RT_{int(time.time())}_{part_counter:03d}'

            # 处理零件
            result = await self.process_single_part(part_id)

            # 实时显示结果
            if result['success']:
                print(
                    f'[实时] {part_id}: {result["grade"]} ({result["processing_time"]:.1f}s)'
                )
            else:
                print(f'[实时] {part_id}: 处理失败')

            part_counter += 1

            # 模拟生产节拍（9秒/件）
            await asyncio.sleep(9)

        self.logger.info('实时演示结束')
        self._generate_demo_report([])


async def main():
    """主函数"""
    print('缸套缺陷分级评估与智能分选系统演示')
    print('Cylinder Liner Defect Grading and Intelligent Sorting System Demo')
    print('-' * 60)

    demo = SystemDemo()

    # 选择演示模式
    print('请选择演示模式:')
    print('1. 单个零件处理演示')
    print('2. 批量处理演示 (10个零件)')
    print('3. 实时处理演示 (5分钟)')

    try:
        choice = input('请输入选择 (1-3): ').strip()

        if choice == '1':
            # 单个零件演示
            part_id = f'DEMO_{int(time.time())}'
            result = await demo.process_single_part(part_id)
            print(f'\n处理结果: {json.dumps(result, indent=2, ensure_ascii=False)}')

        elif choice == '2':
            # 批量演示
            results = await demo.run_batch_demo(10)
            print(f'\n批量处理完成，共处理 {len(results)} 个零件')

        elif choice == '3':
            # 实时演示
            await demo.run_realtime_demo(5)

        else:
            print('无效选择，退出演示')
            return

        print('\n演示完成！')
        print('详细结果已保存到 demo_results.json')
        print('日志已保存到 demo.log')

    except KeyboardInterrupt:
        print('\n\n演示被用户中断')
    except Exception as e:
        print(f'\n演示过程中发生错误: {e}')


if __name__ == '__main__':
    asyncio.run(main())
