from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from thinkvln.datagen.generation.watcher_openai_annotation import (
    annotate_sample,
    request_json_completion,
)
from thinkvln.datagen.generation.watcher_utils import (
    resolve_bundle_image_path,
    select_label_image_relpaths,
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
    def test_select_label_image_relpaths_uses_stride_and_keeps_final(self):
        relpaths = select_label_image_relpaths(
            "pivot/ep/pivot_000010_rgb.jpg",
            [
                "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000001_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000002_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000003_rgb.jpg",
            ],
            stride=2,
        )

        assert relpaths == [
            "pivot/ep/pivot_000010_rgb.jpg",
            "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
            "rollout/ep/pivot_000010/rollout_01/000002_rgb.jpg",
            "rollout/ep/pivot_000010/rollout_01/000003_rgb.jpg",
        ]

    def test_request_json_completion_retries_after_bad_payload(self):
        client = _FakeClient(["not json", '{"ok": true}'])

        payload = request_json_completion(
            client,
            model="gpt-test",
            messages=[{"role": "user", "content": "hello"}],
            max_retries=2,
        )

        assert payload == {"ok": True}
        assert len(client.chat.completions.calls) == 2

    def test_annotate_sample_resolves_paths_and_uses_stride(self, tmp_path: Path):
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
            "instruction": "Go to the room.",
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
                '{"label": "RESUME", "memory_end": "The agent still needs to enter the room."}',
            ]
        )

        annotation = annotate_sample(
            client=client,
            model="gpt-test",
            bundle_root=bundle_root,
            record=record,
            label_image_stride=2,
        )

        assert annotation["label"] == "RESUME"
        assert annotation["memory_start"] == "The agent is near the doorway."
        assert annotation["memory_end"] == "The agent still needs to enter the room."

        second_call = client.chat.completions.calls[1]
        content = second_call["messages"][1]["content"]
        image_items = [item for item in content if item.get("type") == "image_url"]
        assert len(image_items) == 4
        assert "Trajectory summary: a=['forward', 'turn_right']" in content[0]["text"]
        assert "p=" not in content[0]["text"]

        resolved = resolve_bundle_image_path(
            bundle_root,
            record["base_image_path"],
            record["pivot_image_relpath"],
        )
        assert resolved == pivot_path
