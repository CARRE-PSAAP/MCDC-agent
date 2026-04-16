import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mcdc_agent.v2.services.visualization import VisualizationProgram, VisualizationService


class FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, payload):
        return {"content": self.content}


class VisualizationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = VisualizationService()

    def test_plan_plot_uses_direct_llm_code(self):
        self.service._llm = FakeLLM(
            """```python
import numpy as np
fig, ax = plt.subplots()
ax.plot(np.asarray(handle["k_cycle"][:]))
fig.savefig("should_not_exist.png")
```"""
        )

        program = self.service.plan_plot(self._sample_summary(), request="Plot the k-effective convergence.")

        self.assertEqual(program.mode, "llm_direct")
        self.assertIn('ax.plot(np.asarray(handle["k_cycle"][:]))', program.code)
        self.assertNotIn("import numpy", program.code)
        self.assertNotIn("savefig", program.code)

    def test_plan_plot_raises_when_llm_is_unavailable(self):
        self.service._get_llm = lambda: (_ for _ in ()).throw(RuntimeError("LLM unavailable"))

        with self.assertRaisesRegex(RuntimeError, "LLM unavailable"):
            self.service.plan_plot(self._sample_summary(), request="")

    def test_render_plot_executes_generated_code(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_h5 = Path(tmp_dir) / "output.h5"
            plot_path = Path(tmp_dir) / "plot.png"
            self._write_output(output_h5)

            program = VisualizationProgram(
                code=(
                    'fig, ax = plt.subplots(figsize=(6, 4))\n'
                    'ax.plot(np.asarray(handle["k_cycle"][:]), marker="o")\n'
                    'ax.set_title("k-effective convergence")\n'
                    'ax.set_xlabel("Cycle index")\n'
                    'ax.set_ylabel("k-effective")'
                ),
                mode="llm_direct",
                request="Plot the k-effective convergence.",
            )

            rendered = self.service.render_plot(output_h5, program, output_path=plot_path)

            self.assertEqual(rendered, plot_path.resolve())
            self.assertTrue(rendered.exists())
            self.assertGreater(rendered.stat().st_size, 0)

    @staticmethod
    def _sample_summary():
        return {
            "has_eigenvalue": True,
            "has_runtime": True,
            "has_tallies": True,
            "tallies": [
                {
                    "name": "mesh_tally_0",
                    "is_mesh": True,
                    "scores": {"flux": {"mean_shape": [2, 2]}},
                }
            ],
        }

    @staticmethod
    def _write_output(path: Path) -> None:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("k_cycle", data=np.array([1.0, 1.02, 0.99, 1.01]))


if __name__ == "__main__":
    unittest.main()
