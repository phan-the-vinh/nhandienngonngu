"""Adapter parity and safe entrypoints; no webcam, extraction or training."""
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src import ROOT
from src.model import load_configuration,load_models,load_digits,load_dynamic,load_letters
from src.recognition.alphabet import AlphabetRecognizer
from app.inference.unified_abc123_jz_auto_coordinator import UnifiedAutoCoordinator
from tests.test_unified_abc123_jz_auto import DynamicModel,mirrored
from tests.test_unified_camera_experiment import FakePredictor,synthetic_hand
from tests.test_dynamic_jz_motion_gate import articulated_hand


class AdapterTests(unittest.TestCase):
    def test_baseline_protected_models_algorithms_manifests_and_configs_unchanged(self):
        baseline=json.loads((ROOT/'results/experiments/architecture_refactor_v1/baseline.json').read_text())
        allowed={'run.py','README.md','app/camera/realtime_inference.py',
                 'app/inference/unified_abc123_jz_auto_experiment.py'}
        for name,info in baseline['files'].items():
            if name not in allowed:
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),info['sha256'],name)

    def test_canonical_functions_and_interaction_classes_are_identical(self):
        from src.preprocessing import landmarks,hand_centric,dynamic_sequence
        from src.interaction import hold,coordinator
        from ai.preprocessing.landmark_utils import normalize_landmarks
        from ai.preprocessing.hand_centric_2d import HandCentric2DAligner
        from ai.feature_extraction.dynamic_sequence import build_sequence_features,sample_valid_frame_indices
        from app.interaction.hold_confirmation import HoldConfirmation
        from app.inference.unified_static_coordinator import TextComposer
        self.assertIs(landmarks.normalize_landmarks,normalize_landmarks)
        self.assertIs(hand_centric.HandCentric2DAligner,HandCentric2DAligner)
        self.assertIs(dynamic_sequence.build_sequence_features,build_sequence_features)
        self.assertIs(dynamic_sequence.sample_valid_frame_indices,sample_valid_frame_indices)
        self.assertIs(hold.HoldConfirmation,HoldConfirmation)
        self.assertIs(coordinator.TextComposer,TextComposer)

    def test_imports_do_not_load_models_open_camera_or_train(self):
        with (patch('cv2.VideoCapture',side_effect=AssertionError('camera forbidden')),
              patch('joblib.load',side_effect=AssertionError('model load forbidden'))):
            for path in (ROOT/'src').rglob('*.py'):
                name='.'.join(path.relative_to(ROOT).with_suffix('').parts)
                if name.endswith('.__init__'):name=name[:-9]
                importlib.reload(importlib.import_module(name))

    def test_alphabet_is_coordinator_facade_not_26_class_classifier(self):
        self.assertEqual(len(AlphabetRecognizer.alphabet),26)
        self.assertEqual(len(AlphabetRecognizer.static_labels),24)
        self.assertFalse(AlphabetRecognizer.is_26_class_classifier)
        self.assertIs(AlphabetRecognizer.update,UnifiedAutoCoordinator.update)
        self.assertIs(AlphabetRecognizer.process_pending,UnifiedAutoCoordinator.process_pending)
        self.assertIs(AlphabetRecognizer.consume_dynamic_attempt,UnifiedAutoCoordinator.consume_dynamic_attempt)

    def test_facade_and_old_coordinator_match_every_frame_including_shadow_and_switch(self):
        models=[DynamicModel('Z'),DynamicModel('Z')]
        constructors=(UnifiedAutoCoordinator,AlphabetRecognizer)
        cs=[constructor(FakePredictor('I',0.),FakePredictor('0',.5),models[n],
                enable_experimental_jz=True,z_start_pose_shadow=True) for n,constructor in enumerate(constructors)]
        for i in range(320):
            hand=articulated_hand(min(i,65)) if i<220 else synthetic_hand()
            if i==220:
                for c in cs:c.switch_mode('123')
            if i==275:
                for c in cs:c.switch_mode('ABC')
            outputs=[c.update(i,mirrored(hand),hand,(480,640,3),i/30,raw_hand_count=1) for c in cs]
            self.assertEqual(outputs[0][1],outputs[1][1])
            for attr in ('display_label','candidate','blocked','status','reason','z_start_pose_report'):
                self.assertEqual(getattr(cs[0],attr),getattr(cs[1],attr))
            self.assertEqual(cs[0].hold.state,cs[1].hold.state)
            self.assertEqual(cs[0].composer.text,cs[1].composer.text)
        self.assertEqual(len(models[0].inputs),len(models[1].inputs))
        for a,b in zip(models[0].inputs,models[1].inputs):np.testing.assert_array_equal(a,b)

    def test_default_facade_does_not_enable_jz_commits(self):
        c=AlphabetRecognizer(FakePredictor('I',0.),FakePredictor('0',.5),DynamicModel())
        self.assertFalse(c.enable_experimental_jz)

    def test_ui_alias_preserves_mirror_and_buttons(self):
        from src.ui import camera_interface as new
        from app.inference import unified_abc123_jz_auto_experiment as old
        from app.inference.unified_camera_experiment import prepare_inference_frame
        self.assertIs(new.render,old.render)
        self.assertIs(new.mode_from_click,old.mode_from_click)
        self.assertIs(new.prepare_inference_frame,prepare_inference_frame)


class RealModelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=load_models()

    def test_letter_labels_scaler_and_probabilities_equal_original_loader(self):
        old=load_letters()
        self.assertEqual(self.bundle.letters.labels,old.labels)
        self.assertEqual(len(old.labels),24)
        from src.recognition.letters import features_from_landmarks
        features=features_from_landmarks(mirrored(synthetic_hand())).reshape(1,63)
        np.testing.assert_array_equal(self.bundle.letters.scaler.transform(features),old.scaler.transform(features))
        np.testing.assert_array_equal(self.bundle.letters.model.predict_proba(old.scaler.transform(features)),
                                      old.model.predict_proba(old.scaler.transform(features)))

    def test_digit_pipeline_deserialization_and_probabilities_equal_original(self):
        old=load_digits();new=self.bundle.digits
        from ai.preprocessing.landmark_utils import normalize_landmarks
        features=np.asarray(normalize_landmarks(synthetic_hand())).reshape(1,63)
        self.assertEqual(list(new.named_steps),['orientation_2d','scaler','mlp'])
        self.assertEqual(new.named_steps['orientation_2d'].__class__.__module__,'hand_centric_2d')
        np.testing.assert_array_equal(new.predict_proba(features),old.predict_proba(features))
        np.testing.assert_array_equal(new.classes_,old.classes_)

    def test_dynamic_builder_and_model_outputs_equal_original(self):
        from src.recognition.dynamic_jz import predict_attempt
        from app.inference.dynamic_jz_camera_test import predict_attempt as old_predict
        old,digest=load_dynamic()
        entries=[(i,articulated_hand(i),(640,480)) for i in range(40)]
        self.assertEqual(predict_attempt(entries,40,self.bundle.dynamic),old_predict(entries,40,old))
        self.assertEqual(digest,self.bundle.dynamic_sha256)
        self.assertEqual(list(old.classes_),['J','Z'])
        self.assertEqual(old.n_features_in_,1104)

    def test_prediction_service_routes_models_without_ui(self):
        from src.predict import PredictionService
        service=PredictionService(self.bundle)
        self.assertIn(service.letters(mirrored(synthetic_hand())).instant_label,self.bundle.letters.labels)
        self.assertIn(service.digits(synthetic_hand()).instant_label,tuple('0123456789'))
        result=service.dynamic_jz([(i,articulated_hand(i),(640,480)) for i in range(20)],20)
        self.assertEqual(result['status'],'PREDICTED')
        self.assertIsInstance(service.alphabet(),AlphabetRecognizer)

    def test_mediapipe_task_is_resolved_from_configuration_without_detection(self):
        from src.model import project_file
        from app.inference.letter_inference import HAND_MODEL_PATH
        from app.inference.jz_attempt_capture import sha256
        path=project_file(self.bundle.config['mediapipe_task'])
        self.assertEqual(path,HAND_MODEL_PATH.resolve())
        self.assertEqual(sha256(path),'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1')

    def test_mediapipe_task_loads_from_config_without_camera_or_frame_processing(self):
        from src.model import project_file
        from src.preprocessing.landmarks import create_hand_landmarker
        path=project_file(self.bundle.config['mediapipe_task'])
        with patch('cv2.VideoCapture',side_effect=AssertionError('camera forbidden')):
            for mode in ('IMAGE','VIDEO'):
                with create_hand_landmarker(path,running_mode=mode,num_hands=1) as task:
                    self.assertIsNotNone(task)


class EntrypointTests(unittest.TestCase):
    def test_default_mode_forwards_check_only_to_unified_with_explicit_jz_enable(self):
        from app.camera import realtime_inference as dispatcher
        import sys
        with patch.object(dispatcher.runpy,'run_path') as launch,patch.object(sys,'argv',['run.py']):
            self.assertEqual(dispatcher.main(['--check-only']),0)
            launch.assert_called_once_with(str(ROOT/'src/main.py'),run_name='__main__')
            self.assertEqual(sys.argv[1:],['--check-only','--enable-experimental-jz'])

    def test_run_python_file_without_arguments_routes_to_existing_unified(self):
        import runpy
        import sys
        from app.camera import realtime_inference as dispatcher
        execute_entry=runpy.run_path
        with (patch.object(dispatcher.runpy,'run_path') as launch,
              patch.object(sys,'argv',[str(ROOT/'run.py')]),
              patch('cv2.VideoCapture',side_effect=AssertionError('camera forbidden'))):
            with self.assertRaises(SystemExit) as exit_code:
                execute_entry(str(ROOT/'run.py'),run_name='__main__')
            self.assertEqual(exit_code.exception.code,0)
            launch.assert_called_once_with(str(ROOT/'src/main.py'),run_name='__main__')
            self.assertEqual(sys.argv[1:],['--enable-experimental-jz'])

    def test_run_default_check_only_enables_jz_commit_and_never_opens_camera(self):
        import runpy
        import sys
        from app.inference import unified_abc123_jz_auto_experiment as old
        with (patch.object(sys,'argv',[str(ROOT/'run.py'),'--check-only']),
              patch('cv2.VideoCapture',side_effect=AssertionError('camera forbidden')),
              patch.object(old,'create_hand_landmarker',side_effect=AssertionError('task execution forbidden')),
              contextlib.redirect_stdout(io.StringIO()) as output):
            with self.assertRaises(SystemExit) as exit_code:
                runpy.run_path(str(ROOT/'run.py'),run_name='__main__')
        self.assertEqual(exit_code.exception.code,0)
        report=json.loads(output.getvalue())
        self.assertEqual(report['modes'],['ABC','123'])
        self.assertTrue(report['jz_commits_enabled'])
        self.assertFalse(report['camera_opened'])
        self.assertEqual(report['files_written'],0)

    def test_old_modes_forward_argv_unchanged(self):
        from app.camera import realtime_inference as dispatcher
        import sys
        for mode in ('letters','digits'):
            with patch.object(dispatcher.runpy,'run_path') as run,patch.object(sys,'argv',['test']):
                self.assertEqual(dispatcher.main(['--mode',mode,'--model-path','example.joblib']),0)
                run.assert_called_once_with(str(dispatcher.PROGRAMS[mode]),run_name='__main__')
                self.assertEqual(sys.argv[1:],['--model-path','example.joblib'])

    def test_explicit_unified_has_same_jz_launch_policy_as_default(self):
        from app.camera import realtime_inference as dispatcher
        import sys
        with patch.object(dispatcher.runpy,'run_path') as run,patch.object(sys,'argv',['test']):
            dispatcher.main(['--mode','unified','--check-only'])
            run.assert_called_once_with(str(ROOT/'src/main.py'),run_name='__main__')
            self.assertEqual(sys.argv[1:],['--check-only','--enable-experimental-jz'])

    def test_disable_switch_is_consumed_and_never_reaches_unsupported_inner_parser(self):
        from app.camera import realtime_inference as dispatcher
        import sys
        for args in (['--disable-experimental-jz'],['--mode','unified','--disable-experimental-jz']):
            with patch.object(dispatcher.runpy,'run_path'),patch.object(sys,'argv',['run.py']):
                self.assertEqual(dispatcher.main([*args,'--check-only']),0)
                self.assertEqual(sys.argv[1:],['--check-only'])

    def test_explicit_enable_is_forwarded_exactly_once(self):
        from app.camera import realtime_inference as dispatcher
        import sys
        with patch.object(dispatcher.runpy,'run_path'),patch.object(sys,'argv',['run.py']):
            dispatcher.main(['--enable-experimental-jz','--check-only'])
            self.assertEqual(sys.argv[1:].count('--enable-experimental-jz'),1)

    def test_conflicting_jz_flags_and_jz_flags_in_static_modes_are_rejected(self):
        from app.camera import realtime_inference as dispatcher
        cases=[['--enable-experimental-jz','--disable-experimental-jz']]
        cases.extend(['--mode',mode,'--disable-experimental-jz'] for mode in ('letters','digits'))
        for args in cases:
            with (patch.object(dispatcher.runpy,'run_path') as launch,
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as result:dispatcher.main(args)
                self.assertEqual(result.exception.code,2)
                launch.assert_not_called()

    def test_real_default_disable_check_only_reports_commit_off(self):
        from app.camera.realtime_inference import main
        import sys
        with (patch.object(sys,'argv',['run.py']),
              patch('cv2.VideoCapture',side_effect=AssertionError('camera forbidden')),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(main(['--check-only','--disable-experimental-jz']),0)
        self.assertFalse(json.loads(output.getvalue())['jz_commits_enabled'])

    def test_new_camera_injects_facade_and_loader_not_new_camera_loop(self):
        from src.camera import realtime
        from app.inference import unified_abc123_jz_auto_experiment as old
        with patch.object(old,'main',return_value=0) as run:
            self.assertEqual(realtime.main(['--check-only','--z-start-pose-shadow']),0)
            self.assertEqual(run.call_args.args[0],['--check-only','--z-start-pose-shadow'])
            self.assertIs(run.call_args.kwargs['coordinator_factory'],AlphabetRecognizer)
            self.assertTrue(callable(run.call_args.kwargs['artifact_loader']))

    def test_real_unified_check_only_never_opens_camera_or_writes_diagnostics(self):
        from src.main import main
        from app.inference import unified_abc123_jz_auto_experiment as old
        with (patch('cv2.VideoCapture',side_effect=AssertionError('camera forbidden')),
              patch.object(old,'create_hand_landmarker',side_effect=AssertionError('task execution forbidden')),
              patch.object(old,'AttemptCapture',side_effect=AssertionError('capture write forbidden')),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(main(['--check-only','--z-start-pose-shadow','--capture-jz-diagnostics']),0)
        report=json.loads(output.getvalue())
        self.assertFalse(report['jz_commits_enabled'])
        self.assertFalse(report['z_start_pose_acceptance_gate'])
        self.assertEqual(report['files_written'],0)
        self.assertEqual(report['dynamic_feature_shape'],[16,69])

    def test_contract_override_fails_instead_of_being_silently_ignored(self):
        import src.model as models
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            config=json.loads((ROOT/'configs/application.json').read_text())
            config['jz_commits_default']=True
            path.write_text(json.dumps(config))
            with patch.object(models,'ROOT',Path(temp)):
                with self.assertRaisesRegex(ValueError,'Unsupported'):load_configuration(path)

    def test_train_check_only_never_imports_training_or_samples(self):
        from src.train import main
        with patch('src._workflow.importlib.import_module',side_effect=AssertionError('training import forbidden')):
            for module in ('digits','dynamic'):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main(['--module',module,'--check-only']),0)
                self.assertFalse(json.loads(out.getvalue())['execution_performed'])

    def test_letters_and_production_digits_evaluation_are_explicit_blockers(self):
        from src.train import main as train
        from src.evaluate import main as evaluate
        for call,args in ((train,['--module','letters','--check-only']),
                          (evaluate,['--module','digits','--check-only']),
                          (evaluate,['--module','letters','--execute'])):
            with (patch('src._workflow.importlib.import_module',side_effect=AssertionError('unsafe implementation import')),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(call(args),2)

    def test_user_execution_delegates_without_rewriting_training(self):
        from src.train import main
        fake=SimpleNamespace(main=lambda argv:argv)
        with patch('src._workflow.importlib.import_module',return_value=fake):
            self.assertEqual(main(['--module','dynamic','--execute','--config','example.json']),
                             ['--train','--config','example.json'])


class DataTests(unittest.TestCase):
    def test_static_csv_adapter_preserves_metadata_and_feature_order(self):
        from src.data import read_static_csv,FEATURE_COLUMNS
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.csv'
            path.write_text(','.join(('sample_id','label',*FEATURE_COLUMNS))+'\n'+
                            ','.join(('00001','A',*(str(i) for i in range(63))))+'\n')
            rows,features=read_static_csv(path,labels=('A','B'))
            self.assertEqual(rows[0]['sample_id'],'00001')
            np.testing.assert_array_equal(features,np.arange(63).reshape(1,63))
            with self.assertRaises(ValueError):read_static_csv(path,labels=('Z',))

    def test_sequence_reader_keeps_shape_order_and_metadata(self):
        from src.data import read_sequences
        record={'source_path':'J/example.avi','label':'J','features':np.zeros((16,69)).tolist(),
                'sequence_length':16,'feature_dim':69,'selected_original_frame_indices':list(range(16))}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'sequences.jsonl';path.write_text(json.dumps(record)+'\n')
            self.assertEqual(list(read_sequences(path)),[record])
            path.write_text(json.dumps(record)+'\n'+json.dumps(record)+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):list(read_sequences(path))

    def test_invalid_sequence_or_labels_not_silently_repaired(self):
        from src.data import validate_sequence_record
        with self.assertRaises(ValueError):validate_sequence_record({'features':np.zeros((15,69)).tolist(),'label':'NONE'})

    def test_target_dataset_is_not_assumed_migrated(self):
        from src.data import dataset_path,PLANNED_DATASET_ROOT
        self.assertEqual(PLANNED_DATASET_ROOT,ROOT/'dataset')
        self.assertIn('data/processed/digits',dataset_path('digits_frozen_train').as_posix())


if __name__=='__main__':
    unittest.main()
