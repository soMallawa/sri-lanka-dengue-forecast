"""Keep research artifact locations portable across developer environments."""
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _constant(relative_path, name):
    tree = ast.parse((ROOT / 'src' / 'dengue_forecast' / relative_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return eval(compile(ast.Expression(node.value), '<constant>', 'eval'),
                        {'core': SimpleNamespace(REPO_ROOT=Path('/project')), 'Path': Path})
    raise AssertionError(f'Missing constant: {name}')


def test_trusted_registry_uses_research_directory():
    assert _constant('milestone3/final_evaluation.py', 'TRUSTED_REGISTRY_ROOT') == Path('/project/.research/m3-final-trusted-registry')


def test_fixture_outputs_use_research_directory():
    assert Path('/project/.research/qa-temp') in _constant('milestone3/development.py', 'FIXTURE_OUTPUT_ROOTS')


def test_report_archive_reference_is_location_independent():
    text = (ROOT / 'src/dengue_forecast/reports/milestone2.py').read_text()
    assert '`m2-source-before-first-cv.tar.gz`' in text
