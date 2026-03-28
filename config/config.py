from pathlib import Path
project_root_path = Path(__file__).resolve().parents[1]
res_dir = project_root_path.joinpath('results')


config = {
    'root_path': project_root_path,


}
