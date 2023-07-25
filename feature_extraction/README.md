# Steps for feature extraction:

1. Put your files (DEF, other reports from Innovs) in directory under ./data. 

   Put LEF files in ./LEF.

2. Modify the arguments in process_data.py to math your path.

3. Start feature extraction

```python
python process_data.py
```

The results are in ./out

4. Visualization

Modify the arguments in vis.py to match your path, and

```python
python vis.py
```

The results are in ./images
