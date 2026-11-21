# Borehole and coal-seam analysis

Python scripts for reading drilling logs from a close-grid grouting project over
shallow coal workings, and for plotting the boreholes, cross-sections, coal-seam
contours and faults.

**This repository holds the code only.** The borehole data it was written for is
not included. To run the scripts, put these files in the same folder (the format
of each is described in `project_notes.txt`):

| File | Contents |
|---|---|
| `borehole_locations.csv` | `hole_id, plot, hole_no, hole_type, x_m, y_m, ground_level_m` |
| `drilling_logs.xlsx` | the paper drilling-log form, one block per borehole, one sheet per plot |
| `faults.csv` | `fault_id, vertex, x_m, y_m` (one polyline per fault) |
| `fault_throws.csv` | `fault_id, throw_as_marked, throw_ft, figure_uncertain, length_in_area_m, remarks` |

Coordinates are local x (east) and y (north) in metres; ground level is in metres
above a site datum.

## Scripts

| Script | What it does |
|---|---|
| `read_drilling_logs.py` | reads `drilling_logs.xlsx` into one row per layer (`strata_layers.csv`) |
| `site_data.py` | shared loading of holes, strata and faults; the setting-out grid; the rules that turn a driller's description into a stratum class |
| `plot_borehole_layout.py` | plan of the boreholes coloured by plot, with the faults and their throws |
| `plot_all_holes.py` | plan of every hole by type, with plot outlines and section lines; index map of all grid lines |
| `plot_section.py` | cross-sections along lines of grid holes, by depth or by level (`python plot_section.py all` for every line) |
| `plot_fault_section.py` | a level section with the faults marked where they cross it |
| `plot_3d.py` | interactive 3D view (HTML) of the strata near a line |
| `plot_seam_contours.py` | depth and level contours of the worked-seam horizon |
| `seam_model.py` | assigns every coal layer to the upper, worked or lower seam (plane fits and nearest-neighbour rules) |

Run any of them with `python <script>.py`; the docstring at the top of each file
gives the options. Figures and tables are written to `output/`.

## Requirements

```
pip install pandas numpy scipy matplotlib openpyxl plotly
```

(`plotly` is needed only for `plot_3d.py`.)

## Licence

MIT - see [LICENSE](LICENSE).
