import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image

from thinkvln.datagen.generation.watcher_manual_switch_annotation import parse_args
from thinkvln.datagen.manual_switch_web.app import APP_DIR, ManualSwitchConfig, build_store, prepare_sample


class TestWatcherManualSwitchAnnotation:
    def test_parse_args_supports_service_inputs(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "watcher_manual_switch_annotation.py",
                "--bundle_root", "/tmp/bundle",
                "--manifest_file", "/tmp/manifest.jsonl",
                "--output_file", "/tmp/human_switch.jsonl",
                "--page_size", "25",
                "--port", "9100",
                "--resume",
            ],
        )

        args = parse_args()

        assert args.bundle_root == Path("/tmp/bundle")
        assert args.manifest_file == Path("/tmp/manifest.jsonl")
        assert args.output_file == Path("/tmp/human_switch.jsonl")
        assert args.page_size == 25
        assert args.port == 9100
        assert args.resume is True

    def test_prepare_sample_builds_asset_urls(self, tmp_path: Path):
        bundle_root = tmp_path / "bundle"
        image_root = bundle_root / "images"
        pivot_path = image_root / "pivot" / "ep" / "pivot_000010_rgb.jpg"
        rollout_paths = [
            image_root / "rollout" / "ep" / "pivot_000010" / "rollout_01" / f"{idx:06d}_rgb.jpg"
            for idx in range(3)
        ]
        for image_path in [pivot_path, *rollout_paths]:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(17, 0, 0)).save(image_path)

        record = {
            "sample_id": "ep_p000010_r01",
            "episode_key": "ep",
            "scene_id": "scene",
            "episode_id": 10,
            "pivot_frame": 10,
            "rollout_id": 1,
            "subtask_id": 2,
            "instruction": "Go to the room.",
            "plan": ["Leave the hall.", "Enter the room.", "Stop near the sink."],
            "base_image_path": "images",
            "pivot_image_relpath": "pivot/ep/pivot_000010_rgb.jpg",
            "rollout_image_relpaths": [
                "rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000001_rgb.jpg",
                "rollout/ep/pivot_000010/rollout_01/000002_rgb.jpg",
            ],
        }

        sample = prepare_sample(bundle_root=bundle_root, record=record, image_stride=2)

        assert sample["sample_id"] == "ep_p000010_r01"
        assert sample["active_subtask"] == "Enter the room."
        assert sample["next_subtask"] == "Stop near the sink."
        assert sample["done_steps"] == ["Leave the hall."]
        assert sample["pending_steps"] == ["Stop near the sink."]
        assert sample["pivot_image_url"] == "/assets/images/pivot/ep/pivot_000010_rgb.jpg"
        assert sample["rollout_frame_urls"] == [
            "/assets/images/rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg",
            "/assets/images/rollout/ep/pivot_000010/rollout_01/000002_rgb.jpg",
        ]

    def test_store_supports_revisit_and_edit_own_annotations(self, tmp_path: Path):
        bundle_root = tmp_path / "bundle"
        image_root = bundle_root / "images"
        manifest_file = tmp_path / "manifest.jsonl"
        output_file = tmp_path / "human_switch.jsonl"
        for pivot_name in ["pivot_000010", "pivot_000011"]:
            pivot_path = image_root / "pivot" / "ep" / f"{pivot_name}_rgb.jpg"
            rollout_dir = image_root / "rollout" / "ep" / pivot_name / "rollout_01"
            rollout_path = rollout_dir / "000000_rgb.jpg"
            for image_path in [pivot_path, rollout_path]:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)

        manifest_file.write_text(
            '{"sample_id":"done_p000010_r01","episode_key":"ep","scene_id":"scene","episode_id":10,"pivot_frame":10,'
            '"rollout_id":1,"subtask_id":2,"instruction":"Go to the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],'
            '"base_image_path":"images","pivot_image_relpath":"pivot/ep/pivot_000010_rgb.jpg",'
            '"rollout_image_relpaths":["rollout/ep/pivot_000010/rollout_01/000000_rgb.jpg"]}\n'
            '{"sample_id":"todo_p000011_r01","episode_key":"ep","scene_id":"scene","episode_id":11,"pivot_frame":11,'
            '"rollout_id":1,"subtask_id":2,"instruction":"Go to the room.","plan":["Leave the hall.","Enter the room.","Stop near the sink."],'
            '"base_image_path":"images","pivot_image_relpath":"pivot/ep/pivot_000011_rgb.jpg",'
            '"rollout_image_relpaths":["rollout/ep/pivot_000011/rollout_01/000000_rgb.jpg"]}\n',
            encoding="utf-8",
        )
        output_file.write_text('{"sample_id":"done_p000010_r01","should_switch":true}\n', encoding="utf-8")

        store = build_store(
            ManualSwitchConfig(
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                page_size=1,
                max_samples=2,
                bootstrap_admin_password="secret",
            )
        )

        assert store.status()["total"] == 2
        assert store.status()["labeled"] == 1
        assert store.status()["remaining"] == 1

        claimed = store.claim_next(user_id=11)
        assert claimed["sample"]["sample_id"] == "todo_p000011_r01"

        saved = store.submit_annotation(
            user={"id": 11, "username": "alice"},
            sample_id="todo_p000011_r01",
            should_switch=False,
        )
        assert saved["annotation"]["should_switch"] is False
        assert store.status()["labeled"] == 2
        assert store.status()["remaining"] == 0

        edited = store.submit_annotation(
            user={"id": 11, "username": "alice"},
            sample_id="todo_p000011_r01",
            should_switch=True,
        )
        assert edited["annotation"]["should_switch"] is True

        mine = store.list_user_annotations(user_id=11, limit=50)
        assert [row["sample_id"] for row in mine] == ["todo_p000011_r01"]
        assert mine[0]["should_switch"] is True

        loaded = store.get_user_sample(user_id=11, sample_id="todo_p000011_r01")
        assert loaded["sample"]["sample_id"] == "todo_p000011_r01"
        assert loaded["sample"]["annotation"]["should_switch"] is True

        rows = [json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 3
        assert rows[-1]["sample_id"] == "todo_p000011_r01"
        assert rows[-1]["should_switch"] is True

    def test_build_store_defaults_to_full_manifest(self, tmp_path: Path):
        bundle_root = tmp_path / "bundle"
        image_root = bundle_root / "images"
        manifest_file = tmp_path / "manifest.jsonl"
        output_file = tmp_path / "human_switch.jsonl"
        lines = []
        for idx in range(20):
            pivot_path = image_root / "pivot" / "ep" / f"pivot_{idx:06d}_rgb.jpg"
            rollout_path = image_root / "rollout" / "ep" / f"pivot_{idx:06d}" / "rollout_01" / "000000_rgb.jpg"
            for image_path in [pivot_path, rollout_path]:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)
            lines.append(
                json.dumps(
                    {
                        "sample_id": f"sample_{idx:02d}",
                        "episode_key": f"ep_{idx:02d}",
                        "scene_id": "scene",
                        "episode_id": idx,
                        "pivot_frame": idx,
                        "rollout_id": 1,
                        "subtask_id": 1,
                        "instruction": "Go to the room.",
                        "plan": ["Enter the room."],
                        "base_image_path": "images",
                        "pivot_image_relpath": f"pivot/ep/pivot_{idx:06d}_rgb.jpg",
                        "rollout_image_relpaths": [f"rollout/ep/pivot_{idx:06d}/rollout_01/000000_rgb.jpg"],
                    }
                )
            )
        manifest_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        store = build_store(
            ManualSwitchConfig(
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                bootstrap_admin_password="secret",
            )
        )

        assert len(store.samples) == 20
        assert [sample["sample_id"] for sample in store.samples] == [f"sample_{idx:02d}" for idx in range(20)]

    def test_build_store_reconciles_stale_tasks_when_sample_set_changes(self, tmp_path: Path):
        bundle_root = tmp_path / "bundle"
        image_root = bundle_root / "images"
        manifest_file = tmp_path / "manifest.jsonl"
        output_file = tmp_path / "human_switch.jsonl"
        database_file = tmp_path / "manual_switch.sqlite3"
        lines = []
        for idx in range(3):
            pivot_path = image_root / "pivot" / "ep" / f"pivot_{idx:06d}_rgb.jpg"
            rollout_path = image_root / "rollout" / "ep" / f"pivot_{idx:06d}" / "rollout_01" / "000000_rgb.jpg"
            for image_path in [pivot_path, rollout_path]:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(7, 0, 0)).save(image_path)
            lines.append(
                json.dumps(
                    {
                        "sample_id": f"sample_{idx:02d}",
                        "episode_key": f"ep_{idx:02d}",
                        "scene_id": "scene",
                        "episode_id": idx,
                        "pivot_frame": idx,
                        "rollout_id": 1,
                        "subtask_id": 1,
                        "instruction": "Go to the room.",
                        "plan": ["Enter the room."],
                        "base_image_path": "images",
                        "pivot_image_relpath": f"pivot/ep/pivot_{idx:06d}_rgb.jpg",
                        "rollout_image_relpaths": [f"rollout/ep/pivot_{idx:06d}/rollout_01/000000_rgb.jpg"],
                    }
                )
            )
        manifest_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        build_store(
            ManualSwitchConfig(
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                database_file=database_file,
                max_samples=3,
                bootstrap_admin_password="secret",
            )
        )

        store = build_store(
            ManualSwitchConfig(
                bundle_root=bundle_root,
                manifest_file=manifest_file,
                output_file=output_file,
                database_file=database_file,
                max_samples=1,
                bootstrap_admin_password="secret",
            )
        )

        assert len(store.samples) == 1
        assert store.status()["total"] == 1
        assert list(store.sample_by_id) == ["sample_00"]

    def test_frontend_assets_drop_download_flow_and_keep_replay_logic(self):
        page = (APP_DIR / "templates" / "index.html").read_text(encoding="utf-8")
        script = (APP_DIR / "static" / "app.js").read_text(encoding="utf-8")

        assert "Download JSONL" not in page
        assert "{{ title }}" in page
        assert "Claim Next Task" in page
        assert "Refresh Status" not in page
        assert "Reload Mine" in page
        assert "Download JSONL" not in script
        assert "/api/tasks/mine" in script
        assert "async function loadSampleById" in script
        assert "data-value=\"clear\"" not in script
        assert "Current Task" in script
        assert "Next Task If Switched" in script
        assert "if (frameIndex >= frames.length - 1)" in script
        assert "renderFrame(0);" in script
        assert "const wasInMine = state.mineRows.some((row) => String(row.sample_id) === String(sample.sample_id));" in script
        assert "if (wasInMine) {" in script
        assert "setCurrentSample(sample);" in script
        assert "await claimNext();" in script

    def test_frontend_prev_next_follows_my_labels_dropdown_order(self):
        script_path = APP_DIR / "static" / "app.js"
        node_script = textwrap.dedent(
            f"""
            (async () => {{
            const fs = require("node:fs");
            const vm = require("node:vm");
            const assert = require("node:assert/strict");

            function createElement(id) {{
              return {{
                id,
                hidden: false,
                textContent: "",
                innerHTML: "",
                value: "",
                disabled: false,
                dataset: {{}},
                classList: {{ add() {{}}, remove() {{}} }},
                addEventListener() {{}},
                querySelector() {{ return null; }},
                querySelectorAll() {{ return []; }},
              }};
            }}

            const elements = new Proxy({{}}, {{
              get(target, key) {{
                if (!target[key]) {{
                  target[key] = createElement(String(key));
                }}
                return target[key];
              }},
            }});

            function samplePayload(sampleId) {{
              return {{
                sample: {{
                  sample_id: sampleId,
                  episode_id: 1,
                  pivot_frame: 1,
                  rollout_id: 1,
                  active_subtask: "current",
                  next_subtask: "next",
                  pivot_image_url: "/assets/pivot.jpg",
                  rollout_frame_urls: ["/assets/frame.jpg"],
                  annotation: {{ sample_id: sampleId, should_switch: true }},
                }},
                status: {{}},
              }};
            }}

            const fetchCalls = [];
            const payloads = new Map([
              ["/api/me", {{ user: {{ id: 7, username: "alice", role: "annotator" }}, status: {{ total: 4, labeled: 3, remaining: 1, mine_done: 3 }} }}],
              ["/api/tasks/mine?limit=500", {{
                samples: [
                  {{ sample_id: "sample_03", updated_at: 300, should_switch: true }},
                  {{ sample_id: "sample_02", updated_at: 200, should_switch: false }},
                  {{ sample_id: "sample_01", updated_at: 100, should_switch: true }},
                ],
              }}],
              ["/api/tasks/claim", {{ sample: {{
                sample_id: "claim_99",
                episode_id: 99,
                pivot_frame: 99,
                rollout_id: 1,
                active_subtask: "claim current",
                next_subtask: "claim next",
                pivot_image_url: "/assets/pivot.jpg",
                rollout_frame_urls: ["/assets/frame.jpg"],
              }}, status: {{ total: 4, labeled: 3, remaining: 1, mine_done: 3 }} }}],
              ["/api/tasks/sample_03", samplePayload("sample_03")],
              ["/api/tasks/sample_02", samplePayload("sample_02")],
              ["/api/tasks/sample_01", samplePayload("sample_01")],
            ]);

            async function fetch(url) {{
              fetchCalls.push(url);
              if (!payloads.has(url)) {{
                throw new Error(`Unexpected fetch: ${{url}}`);
              }}
              const payload = payloads.get(url);
              return {{
                ok: true,
                async json() {{ return payload; }},
                async text() {{ return JSON.stringify(payload); }},
              }};
            }}

            const context = {{
              console,
              fetch,
              window: {{
                setInterval() {{ return 1; }},
                clearInterval() {{}},
              }},
              document: {{
                getElementById(id) {{
                  return elements[id];
                }},
              }},
              CSS: {{
                escape(value) {{
                  return String(value);
                }},
              }},
              localStorage: {{
                getItem() {{ return "test-token"; }},
                setItem() {{}},
                removeItem() {{}},
              }},
              Image: function() {{
                this.decode = async () => {{}};
              }},
              setTimeout,
              clearTimeout,
              Promise,
            }};
            vm.createContext(context);
            const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
            vm.runInContext(
              source + "\\nthis.__test__ = {{ navigateStrip, loadSampleById }};",
              context,
            );
            await new Promise((resolve) => setTimeout(resolve, 0));

            fetchCalls.length = 0;
            await context.__test__.loadSampleById("sample_02");
            fetchCalls.length = 0;
            await context.__test__.navigateStrip(-1);
            assert.deepStrictEqual(fetchCalls, ["/api/tasks/sample_03"]);

            fetchCalls.length = 0;
            await context.__test__.loadSampleById("sample_02");
            fetchCalls.length = 0;
            await context.__test__.navigateStrip(1);
            assert.deepStrictEqual(fetchCalls, ["/api/tasks/sample_01"]);

            fetchCalls.length = 0;
            await context.__test__.loadSampleById("sample_01");
            fetchCalls.length = 0;
            await context.__test__.navigateStrip(1);
            assert.deepStrictEqual(fetchCalls, []);
            }})().catch((error) => {{
              console.error(error);
              process.exit(1);
            }});
            """
        )

        completed = subprocess.run(
            ["node", "--eval", node_script],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr or completed.stdout
