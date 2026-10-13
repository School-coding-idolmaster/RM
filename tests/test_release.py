"""Offline regressions. Training and LLM calls are replaced by test doubles."""
import contextlib
import csv
import io
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from compiler.compile import compile_robot, make_robot
from robomorph.evolution import loop
from robomorph.evolution.population import EvolutionUnit
from robomorph.evolution.utils import convert_dataset_to_zip, write_txt
from robomorph.main import build_parser, main
from train import parallel_train
from train.train_brax import ExperimentConfig, Hyperparameters, train_policy


class ReleaseTests(unittest.TestCase):
    def test_cli_help_does_not_import_training(self):
        script = '''
import builtins, sys
real_import = builtins.__import__
def lightweight_import(name, *args, **kwargs):
    if name.split('.')[0] in {'jax', 'brax', 'wandb', 'numpy', 'openai'}:
        raise AssertionError('Help imported ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = lightweight_import
from robomorph.main import main
sys.argv = ['robomorph', '--help']
main()
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--terrain', result.stdout)

    def test_cli_rejects_zero_budget_and_path_names(self):
        for arguments in [['--population_size', '0'], ['--timesteps', '-1'],
                          ['--num_generations', '0'], ['--top_k', '0'],
                          ['--exp_name', '../escape']]:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(['--exp_name', 'test'] + arguments)
                self.assertEqual(error.exception.code, 2)

    def test_cli_preserves_existing_results_before_gpu_or_api_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results.csv'
            output.write_text('valuable previous run')
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test'}), \
                    patch.object(sys, 'argv', ['robomorph', '--exp_name', 'test', '--results_path', str(output)]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            self.assertEqual(output.read_text(), 'valuable previous run')
            self.assertFalse((output.parent / 'arguments_test.json').exists())

    def test_initialization_preserves_existing_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results.csv'
            output.write_text('previous results')
            population = loop.create_population(1, 'Design a robot')
            with self.assertRaises(FileExistsError):
                loop.init_run(population, 'test', str(output))
            self.assertEqual(output.read_text(), 'previous results')

    def test_sorted_results_keep_design_id_and_xml_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results.csv'
            population = loop.create_population(2, 'Design a robot')
            loop.init_run(population, 'test', str(output))
            population.units = [
                EvolutionUnit(unit_id=0, fitness=1, components='slow', xml_path=str(output.parent / 'examples_test/evo0_unit0.xml')),
                EvolutionUnit(unit_id=1, fitness=2, components='fast', xml_path=str(output.parent / 'examples_test/evo0_unit1.xml')),
            ]
            with patch.object(loop, '_evaluate_fitness'):
                loop.run_for_n('test', 1, population, Mock(), 'flat', 'unused', str(output))
            with output.open() as file:
                rows = list(csv.DictReader(file, delimiter=';'))
            self.assertEqual(rows[0]['component'], 'fast')
            self.assertEqual(rows[0]['unit_id'], '1')
            self.assertEqual(rows[0]['xml_file'], 'examples_test/evo0_unit1.xml')
            self.assertEqual(rows[1]['unit_id'], '0')

    def test_few_shot_fitness_roundtrip_including_scientific_notation(self):
        records = [('U', '[body=15cm]', 1e-8), ('U', '[body=20cm]', -2.5e-7)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'population.txt'
            write_txt(output, records)
            self.assertEqual(convert_dataset_to_zip(output.read_text()), records)

    def test_empty_llm_response_fails_explicitly(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
        with self.assertRaisesRegex(ValueError, 'empty design'):
            loop.generate_grammar(client, '', '', 'unused')

    def test_no_trainable_design_aborts_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            population = loop.create_population(1, 'Design a robot')
            with patch.object(loop, 'generate_grammar', return_value='unparseable response'), \
                    patch.object(loop, 'train_multiple_models') as train:
                with self.assertRaisesRegex(RuntimeError, 'no trainable designs'):
                    loop._evaluate_fitness('test', 0, population, Mock(), 'flat', 'unused', str(Path(temporary) / 'results.csv'))
            train.assert_not_called()
            self.assertEqual((Path(temporary) / 'examples_test/evo0_unit0.txt').read_text(), 'unparseable response')

    def test_successful_negative_fitness_is_an_elite(self):
        response = 'Step 1: r0) H-U-(C-M-J-L-E)-T\n[null]-[body=15cm]-([connector]-[mount]-[knee=30deg]-[limb=10cm]-[null])-[null]'
        with tempfile.TemporaryDirectory() as temporary:
            population = loop.create_population(1, 'Design a robot')
            results_path = str(Path(temporary) / 'results.csv')
            with patch.object(loop, 'generate_grammar', return_value=response), \
                    patch.object(loop, 'train_multiple_models', side_effect=lambda xmls, **kwargs: [(xmls[0], -5.0)]):
                loop._evaluate_fitness('test', 0, population, Mock(), 'flat', 'unused', results_path)
            self.assertEqual(population.elites[0].fitness, -5.0)
            self.assertEqual(convert_dataset_to_zip((Path(temporary) / 'population_test.txt').read_text())[0][2], -5.0)

    def test_unsupported_jax_masks_fail_before_importing_jax(self):
        for setting in [{'JAX_CUDA_VISIBLE_DEVICES': '1'}, {'JAX_PLATFORM_NAME': 'cpu'}]:
            with self.subTest(setting=setting), patch.dict(os.environ, setting), self.assertRaises(ValueError):
                parallel_train.gpu_ids()

    def test_gpu_ids_honor_existing_cuda_allocation(self):
        jax = SimpleNamespace(local_devices=lambda backend: [SimpleNamespace(id=0), SimpleNamespace(id=1)])
        with patch.dict(sys.modules, {'jax': jax}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,7'}):
            self.assertEqual(parallel_train.gpu_ids(), ['3', '7'])

    def test_gpu_detection_rejects_cpu_only_backend(self):
        jax = SimpleNamespace(local_devices=Mock(side_effect=RuntimeError('no GPU')))
        with patch.dict(sys.modules, {'jax': jax}), self.assertRaisesRegex(RuntimeError, 'requires a CUDA GPU'):
            parallel_train.gpu_ids()

    def test_worker_drains_tasks_before_sentinel_without_empty_check(self):
        tasks, results = queue.Queue(), queue.Queue()
        tasks.empty = Mock(side_effect=AssertionError('Queue.empty must not be used'))
        for xml in ['evo0_unit0.xml', 'evo0_unit1.xml', None]:
            tasks.put(xml)
        trained = []
        def fake_train(config, hyperparameters):
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '7')
            trained.append(config)
            return 0.25
        module = SimpleNamespace(train_policy=fake_train, ExperimentConfig=ExperimentConfig, Hyperparameters=Hyperparameters)
        with patch.dict(sys.modules, {'train.train_brax': module}), patch.dict(os.environ):
            parallel_train._process_task(tasks, results, '7', terrain='beams', output_dir='chosen')
        self.assertEqual([config.name for config in trained], ['evo0_unit0', 'evo0_unit1'])
        self.assertTrue(all(config.output_dir == 'chosen' for config in trained))
        self.assertEqual(results.get_nowait(), ('evo0_unit0.xml', 0.25))
        self.assertEqual(results.get_nowait(), ('evo0_unit1.xml', 0.25))

    def test_timeout_reaps_workers_and_closes_queues(self):
        class TestQueue(queue.Queue):
            def cancel_join_thread(self):
                pass
            def close(self):
                self.closed = True
        tasks, results = TestQueue(), TestQueue()
        process = Mock()
        process.is_alive.return_value = True
        process.terminate.side_effect = lambda: setattr(process.is_alive, "return_value", False)
        context = SimpleNamespace(Queue=Mock(side_effect=[tasks, results]), Process=Mock(return_value=process))
        with patch.object(parallel_train, 'gpu_ids', return_value=['0']), \
                patch.object(parallel_train.multiprocessing, 'get_context', return_value=context):
            with self.assertRaises(TimeoutError):
                parallel_train.train_multiple_models(['robot.xml'], timeout=0.001)
        process.terminate.assert_called_once()
        self.assertTrue(tasks.closed and results.closed)

    def test_failed_training_is_persisted_and_cannot_beat_negative_fitness(self):
        tasks, results = queue.Queue(), queue.Queue()
        tasks.put('failed.xml')
        tasks.put(None)
        module = SimpleNamespace(train_policy=Mock(side_effect=ValueError('invalid robot')), ExperimentConfig=ExperimentConfig, Hyperparameters=Hyperparameters)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(sys.modules, {'train.train_brax': module}), patch.dict(os.environ), self.assertLogs('train.parallel_train', level='ERROR'):
                parallel_train._process_task(tasks, results, '0', output_dir=temporary)
            xml, fitness = results.get_nowait()
            self.assertEqual(xml, 'failed.xml')
            self.assertEqual(fitness, float('-inf'))
            self.assertIn('invalid robot', (Path(temporary) / 'failure_failed.json').read_text())
            units = [EvolutionUnit(fitness=fitness), EvolutionUnit(fitness=-5)]
            self.assertEqual(loop.select_top_units(units, 1)[0].fitness, -5)

    def test_all_failed_training_aborts_after_persisting_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results.csv'
            population = loop.create_population(1, 'Design a robot')
            population.units[0].error = 'invalid robot'
            loop.init_run(population, 'test', str(output))
            with patch.object(loop, '_evaluate_fitness'), self.assertRaisesRegex(RuntimeError, 'no successful training'):
                loop.run_for_n('test', 1, population, Mock(), 'flat', 'unused', str(output))
            with output.open() as file:
                rows = list(csv.DictReader(file, delimiter=';'))
            self.assertEqual(rows[0]['fitness'], '-inf')
            self.assertEqual(rows[0]['error'], 'invalid robot')

    def test_ppo_zero_minibatches_is_rejected_before_loading_rl(self):
        with self.assertRaisesRegex(ValueError, 'positive --num_minibatches'):
            train_policy(ExperimentConfig(xml='unused', name='test', algorithm='ppo'), Hyperparameters())

    def test_renderer_preserves_existing_html_before_loading_policy(self):
        from train.render_policy import render_from_path
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'trained_render.html'
            output.write_text('previous render')
            with self.assertRaises(FileExistsError):
                render_from_path(temporary)
            self.assertEqual(output.read_text(), 'previous render')

    def test_ppo_renderer_builds_inference_without_calling_trainer(self):
        from train import render_policy
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'args.pkl').write_bytes(pickle.dumps({'env': 'flat', 'xml': 'unused', 'algorithm': 'ppo'}))
            environment = SimpleNamespace(observation_size=5, action_size=2)
            with patch('brax.io.model.load_params', return_value=object()), \
                    patch('train.envs.create_env', return_value=environment), \
                    patch('brax.training.agents.ppo.train.train', side_effect=AssertionError('Rendering must not train')) as train, \
                    patch('brax.training.agents.ppo.networks.make_ppo_networks'), \
                    patch('brax.training.agents.ppo.networks.make_inference_fn') as inference, \
                    patch.object(render_policy, '_collect_and_save_rollouts', return_value='offline render'):
                self.assertEqual(render_policy.render_from_path(temporary), 'offline render')
            train.assert_not_called()
            inference.assert_called_once()

    def test_compiler_output_loads_in_mujoco_without_simulation(self):
        import mujoco
        graph = '[null]-[body=15cm]-([connector]-[mount]-[knee=30deg]-[limb=10cm]-[null])-[null]'
        model = mujoco.MjModel.from_xml_string(make_robot(compile_robot(graph)))
        self.assertEqual(model.nu, 2)


if __name__ == '__main__':
    unittest.main()
