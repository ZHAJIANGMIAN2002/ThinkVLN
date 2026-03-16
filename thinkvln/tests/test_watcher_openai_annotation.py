from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import thinkvln.datagen.generation.watcher_openai_annotation as watcher_openai_annotation
from thinkvln.datagen.generation.watcher_openai_annotation import (
    annotate_sample,
    annotate_manifest,
    build_gt_rgb_dir_path,
    build_memory_start_messages,
    build_rollout_messages,
    build_reasoning_config,
    parse_args,
    request_json_completion,
    write_debug_html,
    select_stride_paths,
)
from thinkvln.datagen.generation.watcher_utils import (
    resolve_bundle_image_path,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(response)


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class TestWatcherOpenAIAnnotation:
    def test_parse_args_supports_max_samples_without_model_flag(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "watcher_openai_annotation.py",
                "--gt_image_root", "/tmp/gt",
                "--bundle_root", "/tmp/bundle",
                "--manifest_file", "/tmp/manifest.jsonl",
                "--output_file", "/tmp/output.jsonl",
                "--image_stride", "5",
                "--max_samples", "10",
                "--debug_html_dir", "/tmp/debug",
                "--reasoning_effort", "none",
            ],
        )

        args = parse_args()

        assert args.gt_image_root == Path("/tmp/gt")
        assert args.bundle_root == Path("/tmp/bundle")
        assert args.image_stride == 5
        assert args.max_samples == 10
        assert args.debug_html_dir == Path("/tmp/debug")
        assert args.reasoning_effort == "none"
        assert not hasattr(args, "openai_model")

    def test_build_reasoning_config_disables_reasoning_by_default(self):
        assert build_reasoning_config("none") == {"effort": "none"}
        assert build_reasoning_config("low") == {"effort": "low", "exclude": True}

    def test_prompt_messages_are_structured_and_concise(self):
        image_path = Path("/tmp/test_watcher_prompt_image.jpg")
        Image.new("RGB", (4, 4), color=(1, 2, 3)).save(image_path)
        record = {
            "plan": ["Leave the hall.", "Enter the room.", "Stop near the sink."],
            "subtask_id": 2,
            "subtask_text": "Enter the room.",
            "actions": ["forward", "turn_right"],
            "pivot_frame": 10,
        }
        history_images = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}]

        memory_messages = build_memory_start_messages(record, history_images)
        memory_text = memory_messages[1]["content"][0]["text"]
        assert "Keep only useful progress that still matters" in memory_text
        assert "Make the current pivot state explicit" in memory_text
        assert "Do not narrate frame by frame" in memory_text
        assert "Instruction:" not in memory_text
        assert "Current subtask:" not in memory_text

        rollout_messages = build_rollout_messages(
            record,
            [image_path],
            memory_start="The agent has left the room and faces the doorway.",
        )
        rollout_system = rollout_messages[0]["content"]
        rollout_text = rollout_messages[1]["content"][0]["text"]
        assert '{"done":true,"next_subtask":"...","memory_end":"..."}' in rollout_system
        assert "Done flag rules:" in rollout_system
        assert "Next subtask rules:" in rollout_system
        assert "Memory_end rules:" in rollout_system
        assert (
            '{"done":false,"next_subtask":"continue walking into the bathroom","memory_end":"The agent has left the hallway and is now entering the bathroom toward the sink."}'
            in rollout_system
        )
        assert "Plan state" in rollout_text
        assert "Done" in rollout_text
        assert "- Leave the hall." in rollout_text
        assert "Active" in rollout_text
        assert "- Enter the room." in rollout_text
        assert "Pending" in rollout_text
        assert "- Stop near the sink." in rollout_text
        assert "Current step" not in rollout_text
        assert "Current plan step:" not in rollout_text
        assert "If the active step reaches a natural handoff, set done=true" in rollout_text
        assert "If the active step is still the right step, set done=false" in rollout_text
        assert "Rewrite memory_start into a new cumulative watcher memory" in rollout_text
        assert "Keep only the still-relevant part of memory_start" in rollout_text
        assert "Keep memory_end to exactly 1 short sentence" in rollout_text

    def test_select_stride_paths_uses_stride_and_keeps_final(self):
        relpaths = select_stride_paths(
            [
                "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000001_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000002_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000003_rgb.jpg",
            ],
            stride=2,
        )

        assert relpaths == [
            "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
            "rollout/ep/pivot_000010/rollout_01/000002_rgb.jpg",
            "rollout/ep/pivot_000010/rollout_01/000003_rgb.jpg",
        ]

    def test_build_gt_rgb_dir_path_uses_scene_and_episode(self, tmp_path: Path):
        record = {
            "scene_id": "17DRP5sb8fy",
            "episode_id": 10154,
        }

        rgb_dir = build_gt_rgb_dir_path(tmp_path, record)

        assert rgb_dir == tmp_path / "17DRP5sb8fy_r2r_010154"

    def test_request_json_completion_retries_after_bad_payload(self):
        client = _FakeClient(["not json", '{"ok": true}'])

        payload = request_json_completion(
            client,
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            max_retries=2,
            request_timeout=123.0,
            log_prefix="test",
            reasoning_effort="none",
        )

        assert payload == {"ok": True}
        assert len(client.chat.completions.calls) == 2
        assert client.chat.completions.calls[0]["timeout"] == 123.0
        assert client.chat.completions.calls[0]["extra_body"] == {"reasoning": {"effort": "none"}}

    def test_annotate_sample_uses_gt_history_and_rollout_stride(self, tmp_path: Path, monkeypatch):
        bundle_root = tmp_path / "bundle"
        image_root = bundle_root / "images"
        pivot_path = image_root / "pivot" / "ep" / "pivot_000010_rgb.jpg"
        rollout_paths = [
            image_root / "rollout" / "ep" / "pivot_000010" / "rollout_01" / f"{idx:06d}_rgb.jpg"
            for idx in range(4)
        ]
        for image_path in [pivot_path, *rollout_paths]:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(idx := len(str(image_path)) % 255, 0, 0)).save(image_path)

        record = {
            "sample_id": "ep_p000010_r01",
            "episode_key": "ep",
            "scene_id": "scene",
            "episode_id": 10,
            "pivot_frame": 10,
            "instruction": "Go to the room.",
            "plan": ["Leave the hall.", "Enter the room.", "Stop near the sink."],
            "subtask_id": 2,
            "subtask_text": "Enter the room.",
            "base_image_path": "images",
            "pivot_image_relpath": "pivot/ep/pivot_000010_rgb.jpg",
            "rollout_image_relpaths": [
                "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000001_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000002_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000003_rgb.jpg",
            ],
            "actions": ["forward", "turn_right"],
        }
        client = _FakeClient(
            [
                '{"memory_start": "The agent is near the doorway."}',
                '{"done": false, "next_subtask": "continue walking through the doorway", "memory_end": "The agent is near the doorway and still aligned to enter the room."}',
            ]
        )
        history_images = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,bbb"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ccc"}},
        ]
        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            lambda gt_image_root, record, stride: history_images,
        )

        annotation = annotate_sample(
            client=client,
            model="gpt-test",
            gt_image_root=tmp_path / "gt_images",
            bundle_root=bundle_root,
            record=record,
            image_stride=2,
        )

        assert annotation["done"] is False
        assert annotation["next_subtask"] == "continue walking through the doorway"
        assert annotation["memory_start"] == "The agent is near the doorway."
        assert annotation["memory_end"] == "The agent is near the doorway and still aligned to enter the room."

        first_call = client.chat.completions.calls[0]
        first_content = first_call["messages"][1]["content"]
        first_images = [item for item in first_content if item.get("type") == "image_url"]
        assert first_images == history_images

        second_call = client.chat.completions.calls[1]
        content = second_call["messages"][1]["content"]
        image_items = [item for item in content if item.get("type") == "image_url"]
        assert len(image_items) == 3
        assert "Plan state" in content[0]["text"]
        assert "Done" in content[0]["text"]
        assert "- Leave the hall." in content[0]["text"]
        assert "Active" in content[0]["text"]
        assert "- Enter the room." in content[0]["text"]
        assert "Pending" in content[0]["text"]
        assert "- Stop near the sink." in content[0]["text"]
        assert "p=" not in content[0]["text"]
        assert "The first image is the pivot image" not in content[0]["text"]

        resolved = resolve_bundle_image_path(
            bundle_root,
            record["base_image_path"],
            record["pivot_image_relpath"],
        )
        assert resolved == pivot_path

    def test_annotate_sample_accepts_handoff_next_subtask_from_plan(self, tmp_path: Path, monkeypatch):
        bundle_root = tmp_path / "bundle"
        image_root = bundle_root / "images"
        rollout_path = image_root / "rollout" / "ep" / "pivot_000010" / "rollout_01" / "000000_rgb.jpg"
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(7, 0, 0)).save(rollout_path)
        record = {
            "sample_id": "ep_p000010_r01",
            "episode_key": "ep",
            "scene_id": "scene",
            "episode_id": 10,
            "pivot_frame": 10,
            "plan": ["Leave the hall.", "Enter the room.", "Stop near the sink."],
            "subtask_id": 2,
            "subtask_text": "Enter the room.",
            "base_image_path": "images",
            "pivot_image_relpath": "pivot/ep/pivot_000010_rgb.jpg",
            "rollout_image_relpaths": [
                "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
            ],
            "actions": ["forward"],
        }
        client = _FakeClient(
            [
                '{"memory_start": "The agent is just outside the room."}',
                '{"done": true, "next_subtask": "Stop near the sink.", "memory_end": "The agent has entered the room and is now near the sink area."}',
            ]
        )
        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            lambda gt_image_root, record, stride: [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}
            ],
        )

        annotation = annotate_sample(
            client=client,
            model="gpt-test",
            gt_image_root=tmp_path / "gt_images",
            bundle_root=bundle_root,
            record=record,
            image_stride=2,
        )

        assert annotation["done"] is True
        assert annotation["next_subtask"] == "Stop near the sink."

    def test_annotate_sample_accepts_recovery_next_subtask(self, tmp_path: Path, monkeypatch):
        bundle_root = tmp_path / "bundle"
        image_root = bundle_root / "images"
        rollout_path = image_root / "rollout" / "ep" / "pivot_000010" / "rollout_01" / "000000_rgb.jpg"
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(7, 0, 0)).save(rollout_path)
        record = {
            "sample_id": "ep_p000010_r01",
            "episode_key": "ep",
            "scene_id": "scene",
            "episode_id": 10,
            "pivot_frame": 10,
            "plan": ["Leave the hall.", "Enter the room.", "Stop near the sink."],
            "subtask_id": 2,
            "subtask_text": "Enter the room.",
            "base_image_path": "images",
            "pivot_image_relpath": "pivot/ep/pivot_000010_rgb.jpg",
            "rollout_image_relpaths": [
                "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
            ],
            "actions": ["turn_left"],
        }
        client = _FakeClient(
            [
                '{"memory_start": "The agent is at the wrong doorway."}',
                '{"done": false, "next_subtask": "turn left and move back to the correct doorway", "memory_end": "The agent is still at the wrong doorway facing away from the room entrance."}',
            ]
        )
        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            lambda gt_image_root, record, stride: [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}
            ],
        )

        annotation = annotate_sample(
            client=client,
            model="gpt-test",
            gt_image_root=tmp_path / "gt_images",
            bundle_root=bundle_root,
            record=record,
            image_stride=2,
        )

        assert annotation["done"] is False
        assert annotation["next_subtask"] == "turn left and move back to the correct doorway"

    def test_annotate_manifest_uses_module_client(self, tmp_path: Path, monkeypatch):
        bundle_root = tmp_path / "bundle"
        gt_image_root = tmp_path / "gt_images"
        image_root = bundle_root / "images"
        output_file = tmp_path / "annotations.jsonl"
        manifest_file = tmp_path / "manifest.jsonl"
        pivot_path = image_root / "pivot" / "ep" / "pivot_000010_rgb.jpg"
        rollout_path = image_root / "rollout" / "ep" / "pivot_000010" / "rollout_01" / "000000_rgb.jpg"
        for image_path in [pivot_path, rollout_path]:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)

        manifest_file.write_text(
            '{"sample_id":"ep_p000010_r01","episode_key":"ep","scene_id":"scene","episode_id":10,"pivot_frame":10,'
            '"instruction":"Go to the room.","subtask_id":2,'
            '"subtask_text":"Enter the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],"base_image_path":"images",'
            '"pivot_image_relpath":"pivot/ep/pivot_000010_rgb.jpg",'
            '"rollout_image_relpaths":["rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg"],'
            '"actions":["forward"]}\n'
        )
        fake_client = _FakeClient(
            [
                '{"ok": true}',
                '{"ok": true}',
                '{"memory_start": "The agent is near the doorway."}',
                '{"done": true, "next_subtask": "stop", "memory_end": "The agent entered the room."}',
            ]
        )
        monkeypatch.setattr(watcher_openai_annotation, "client", fake_client)
        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            lambda gt_image_root, record, stride: [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}
            ],
        )

        written = annotate_manifest(
            SimpleNamespace(
                gt_image_root=gt_image_root,
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                image_stride=3,
                max_samples=None,
                resume=False,
            )
        )

        assert written == 1
        rows = output_file.read_text().strip().splitlines()
        assert len(rows) == 1
        assert '"done": true' in rows[0]
        assert len(fake_client.chat.completions.calls) == 4

    def test_annotate_manifest_uses_summary_full_plan_for_prompt_and_debug(self, tmp_path: Path, monkeypatch):
        bundle_root = tmp_path / "bundle"
        gt_image_root = tmp_path / "gt_images"
        debug_html_dir = tmp_path / "debug_html"
        summary_full_path = tmp_path / "summary_full.jsonl"
        image_root = bundle_root / "images"
        output_file = tmp_path / "annotations.jsonl"
        manifest_file = tmp_path / "manifest.jsonl"
        pivot_path = image_root / "pivot" / "ep" / "pivot_000010_rgb.jpg"
        rollout_path = image_root / "rollout" / "ep" / "pivot_000010" / "rollout_01" / "000000_rgb.jpg"
        for image_path in [pivot_path, rollout_path]:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)

        summary_full_path.write_text(
            '{"episode_key":"ep","scene_id":"scene","episode_id":10,"instruction":"Go to the room.",'
            '"plan":["Walk around and turn to and open the left-side door.","Walk through the doorway.","Continue straight past the sinks.","Stop in the doorway immediately after the sinks."]}\n'
        )
        manifest_file.write_text(
            '{"sample_id":"ep_p000010_r01","episode_key":"ep","scene_id":"scene","episode_id":10,"pivot_frame":10,'
            '"instruction":"Go to the room.","subtask_id":2,"subtask_text":"Walk through the doorway.","plan":["Walk through the doorway."],"base_image_path":"images",'
            '"pivot_image_relpath":"pivot/ep/pivot_000010_rgb.jpg","rollout_image_relpaths":["rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg"],'
            '"actions":["forward"]}\n'
        )
        fake_client = _FakeClient(
            [
                '{"ok": true}',
                '{"ok": true}',
                '{"memory_start": "The agent is near the doorway."}',
                '{"done": false, "next_subtask": "continue walking through the doorway", "memory_end": "The agent is near the doorway and still aligned to enter the room."}',
            ]
        )
        monkeypatch.setattr(watcher_openai_annotation, "client", fake_client)
        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            lambda gt_image_root, record, stride: [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}
            ],
        )

        written = annotate_manifest(
            SimpleNamespace(
                gt_image_root=gt_image_root,
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                image_stride=3,
                max_samples=None,
                debug_html_dir=debug_html_dir,
                summary_full_path=summary_full_path,
                resume=False,
            )
        )

        assert written == 1
        rollout_text = fake_client.chat.completions.calls[3]["messages"][1]["content"][0]["text"]
        assert "Done\n- Walk around and turn to and open the left-side door." in rollout_text
        assert "Active\n- Walk through the doorway." in rollout_text
        assert "Pending\n- Continue straight past the sinks.\n- Stop in the doorway immediately after the sinks." in rollout_text

        html = (debug_html_dir / "index.html").read_text()
        assert "Walk around and turn to and open the left-side door." in html
        assert "Walk through the doorway." in html
        assert "Continue straight past the sinks." in html
        assert "Stop in the doorway immediately after the sinks." in html

    def test_annotate_manifest_respects_max_samples(self, tmp_path: Path, monkeypatch):
        bundle_root = tmp_path / "bundle"
        gt_image_root = tmp_path / "gt_images"
        image_root = bundle_root / "images"
        output_file = tmp_path / "annotations.jsonl"
        manifest_file = tmp_path / "manifest.jsonl"
        for pivot_name in ["pivot_000010", "pivot_000011"]:
            pivot_path = image_root / "pivot" / "ep" / f"{pivot_name}_rgb.jpg"
            rollout_path = image_root / "rollout" / "ep" / pivot_name / "rollout_01" / "000000_rgb.jpg"
            for image_path in [pivot_path, rollout_path]:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)
        manifest_file.write_text(
            '{"sample_id":"ep_p000010_r01","episode_key":"ep","scene_id":"scene","episode_id":10,"pivot_frame":10,'
            '"instruction":"Go to the room.","subtask_id":2,"subtask_text":"Enter the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],"base_image_path":"images",'
            '"pivot_image_relpath":"pivot/ep/pivot_000010_rgb.jpg","rollout_image_relpaths":["rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg"],'
            '"actions":["forward"]}\n'
            '{"sample_id":"ep_p000011_r01","episode_key":"ep","scene_id":"scene","episode_id":11,"pivot_frame":11,'
            '"instruction":"Go to the room.","subtask_id":2,"subtask_text":"Enter the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],"base_image_path":"images",'
            '"pivot_image_relpath":"pivot/ep/pivot_000011_rgb.jpg","rollout_image_relpaths":["rollout/ep/pivot_000011/rollout_01/000000_rgb.jpg"],'
            '"actions":["forward"]}\n'
        )
        fake_client = _FakeClient(
            [
                '{"ok": true}',
                '{"ok": true}',
                '{"memory_start": "The agent is near the doorway."}',
                '{"done": true, "next_subtask": "stop", "memory_end": "The agent entered the room."}',
            ]
        )
        monkeypatch.setattr(watcher_openai_annotation, "client", fake_client)
        monkeypatch.setattr(
            watcher_openai_annotation,
            "MODEL_NAME",
            "gpt-test",
        )
        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            lambda gt_image_root, record, stride: [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}
            ],
        )

        written = annotate_manifest(
            SimpleNamespace(
                gt_image_root=gt_image_root,
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                image_stride=3,
                max_samples=1,
                resume=False,
            )
        )

        assert written == 1
        assert len(output_file.read_text().strip().splitlines()) == 1

    def test_annotate_manifest_skips_missing_gt_when_enabled(self, tmp_path: Path, monkeypatch):
        bundle_root = tmp_path / "bundle"
        gt_image_root = tmp_path / "gt_images"
        image_root = bundle_root / "images"
        output_file = tmp_path / "annotations.jsonl"
        manifest_file = tmp_path / "manifest.jsonl"
        for pivot_name in ["pivot_000010", "pivot_000011"]:
            pivot_path = image_root / "pivot" / "ep" / f"{pivot_name}_rgb.jpg"
            rollout_path = image_root / "rollout" / "ep" / pivot_name / "rollout_01" / "000000_rgb.jpg"
            for image_path in [pivot_path, rollout_path]:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)
        manifest_file.write_text(
            '{"sample_id":"missing_p000010_r01","episode_key":"ep_missing","scene_id":"scene","episode_id":10,"pivot_frame":10,'
            '"instruction":"Go to the room.","subtask_id":2,"subtask_text":"Enter the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],"base_image_path":"images",'
            '"pivot_image_relpath":"pivot/ep/pivot_000010_rgb.jpg","rollout_image_relpaths":["rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg"],'
            '"actions":["forward"]}\n'
            '{"sample_id":"valid_p000011_r01","episode_key":"ep_valid","scene_id":"scene","episode_id":11,"pivot_frame":11,'
            '"instruction":"Go to the room.","subtask_id":2,"subtask_text":"Enter the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],"base_image_path":"images",'
            '"pivot_image_relpath":"pivot/ep/pivot_000011_rgb.jpg","rollout_image_relpaths":["rollout/ep/pivot_000011/rollout_01/000000_rgb.jpg"],'
            '"actions":["forward"]}\n'
        )
        fake_client = _FakeClient(
            [
                '{"ok": true}',
                '{"ok": true}',
                '{"memory_start": "The agent is near the doorway."}',
                '{"done": true, "next_subtask": "stop", "memory_end": "The agent entered the room."}',
            ]
        )
        monkeypatch.setattr(watcher_openai_annotation, "client", fake_client)
        monkeypatch.setattr(watcher_openai_annotation, "MODEL_NAME", "gpt-test")

        def fake_history(gt_root, record, stride):
            if record["sample_id"] == "missing_p000010_r01":
                raise FileNotFoundError("missing trajectory")
            return [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}]

        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            fake_history,
        )

        written = annotate_manifest(
            SimpleNamespace(
                gt_image_root=gt_image_root,
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                image_stride=3,
                max_samples=None,
                skip_missing_gt=True,
                resume=False,
            )
        )

        rows = output_file.read_text().strip().splitlines()
        assert written == 1
        assert len(rows) == 1
        assert "valid_p000011_r01" in rows[0]

    def test_write_debug_html_renders_sample_context(self, tmp_path: Path):
        debug_dir = tmp_path / "debug"
        records = [
            {
                "sample_id": "ep_p000010_r01",
                "instruction": "Go to the room.",
                "plan": ["Leave the hall.", "Enter the room.", "Stop near the sink."],
                "subtask_id": 2,
                "subtask_text": "Enter the room.",
                "actions": ["forward", "turn_right"],
                "memory_start": "The agent has crossed the hall.",
                "done": False,
                "next_subtask": "continue walking through the doorway",
                "memory_end": "The agent is still approaching the doorway.",
                "gt_history_images": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}
                ],
                "rollout_images": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,bbb"}}
                ],
            }
        ]

        write_debug_html(debug_dir, records)

        html = (debug_dir / "index.html").read_text()
        assert "ep_p000010_r01" in html
        assert "The agent has crossed the hall." in html
        assert "False" in html
        assert "<strong>Subtask:</strong>" not in html
        assert "Plan State:" in html
        assert "Done Plan:" in html
        assert "Leave the hall." in html
        assert "Active Plan:" in html
        assert "Enter the room." in html
        assert "Pending Plan:" in html
        assert "Stop near the sink." in html
        assert "continue walking through the doorway" in html
        assert "The agent is still approaching the doorway." in html
        assert "data:image/jpeg;base64,aaa" in html
        assert "data:image/jpeg;base64,bbb" in html

    def test_annotate_manifest_debug_html_does_not_persist_image_payloads(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        bundle_root = tmp_path / "bundle"
        gt_image_root = tmp_path / "gt_images"
        debug_html_dir = tmp_path / "debug_html"
        image_root = bundle_root / "images"
        output_file = tmp_path / "annotations.jsonl"
        manifest_file = tmp_path / "manifest.jsonl"
        pivot_path = image_root / "pivot" / "ep" / "pivot_000010_rgb.jpg"
        rollout_path = image_root / "rollout" / "ep" / "pivot_000010" / "rollout_01" / "000000_rgb.jpg"
        for image_path in [pivot_path, rollout_path]:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)

        manifest_file.write_text(
            '{"sample_id":"ep_p000010_r01","episode_key":"ep","scene_id":"scene","episode_id":10,"pivot_frame":10,'
            '"instruction":"Go to the room.","subtask_id":2,"subtask_text":"Enter the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],"base_image_path":"images",'
            '"pivot_image_relpath":"pivot/ep/pivot_000010_rgb.jpg","rollout_image_relpaths":["rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg"],'
            '"actions":["forward"]}\n'
        )
        fake_client = _FakeClient(
            [
                '{"ok": true}',
                '{"ok": true}',
                '{"memory_start": "The agent is near the doorway."}',
                '{"done": true, "next_subtask": "stop", "memory_end": "The agent entered the room."}',
            ]
        )
        monkeypatch.setattr(watcher_openai_annotation, "client", fake_client)
        monkeypatch.setattr(
            watcher_openai_annotation,
            "extract_gt_history_image_contents",
            lambda gt_image_root, record, stride: [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,aaa"}}
            ],
        )

        written = annotate_manifest(
            SimpleNamespace(
                gt_image_root=gt_image_root,
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                image_stride=3,
                max_samples=None,
                debug_html_dir=debug_html_dir,
                resume=False,
            )
        )

        assert written == 1
        line = output_file.read_text().strip()
        assert '"gt_history_images"' not in line
        assert '"rollout_images"' not in line
        assert (debug_html_dir / "index.html").exists()
