# Steps for feature extraction:

1. Put your files (DEF, other reports from Innovus) in directory under ./data. 

   Put LEF files in ./LEF.

2. Modify the arguments in process_data.py to match your path.

   Modify the function read to call functions and get features.


<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Supported Features</th>
    <th class="tg-0pky">Function</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" colspan="2">Routability Features</td>
  </tr>
  <tr>
    <td class="tg-0pky">macro_region</td>
    <td class="tg-0pky">read_place_def</td>
  </tr>
  <tr>
    <td class="tg-0pky">congestion_early_global_routing</td>
    <td class="tg-0pky">read_eGR_congestion</td>
  </tr>
  <tr>
    <td class="tg-0pky">congestion_global_routing</td>
    <td class="tg-0pky">read_route_congestion</td>
  </tr>
  <tr>
    <td class="tg-0pky">cell_density</td>
    <td class="tg-0pky">compute_cell_density</td>
  </tr>
  <tr>
    <td class="tg-0pky">RUDY/RUDY_long/RUDY_short/pin_RUDY/pin_RUDY_long</td>
    <td class="tg-0pky">get_RUDY</td>
  </tr>
  <tr>
    <td class="tg-0pky">DRC</td>
    <td class="tg-0pky">get_DRC</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="2">IR Drop Features</td>
  </tr>
  <tr>
    <td class="tg-0pky">power_i/power_s/power_sca/power_all/power_t/IR_drop</td>
    <td class="tg-0pky">get_IR_features</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="2">Graph Features</td>
  </tr>
  <tr>
    <td class="tg-0pky">pin_positions</td>
    <td class="tg-0pky">read_route_pin_position</td>
  </tr>
  <tr>
    <td class="tg-0pky">instance_placement_micron/gcell</td>
    <td class="tg-0pky">read_instance_placement</td>
  </tr>
</tbody>
</table>

3. Start feature extraction

```python
python process_data.py
```

The results are in ./out

4. Visualization (optional)

Modify the arguments in vis.py to match your path, and

```python
python vis.py
```

The results are in ./images
