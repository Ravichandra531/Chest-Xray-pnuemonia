"""Execute plain Python notebook cells locally without Jupyter server sockets.

Captures genuine cell stdout and display outputs; stops and saves on failure.
Notebook contains no magics. Colab uses its own standard kernel instead.
"""
import base64
import contextlib
import io
import json
import traceback
import sys
from pathlib import Path

root=Path(__file__).resolve().parent
path=root/(sys.argv[1] if len(sys.argv)>1 else 'Notebooks/Chest_Xray_Pneumonia_Phase1.ipynb')
nb=json.loads(path.read_text())
namespace={'__name__':'__notebook__'}
from IPython.core.interactiveshell import InteractiveShell
shell=InteractiveShell.instance()
from IPython.utils.capture import capture_output

count=0
for cell in nb['cells']:
    if cell['cell_type']!='code': continue
    count+=1
    print(f'Executing cell {count}',flush=True)
    cell['execution_count']=count
    cell['outputs']=[]
    try:
        with capture_output(display=True) as captured:
            exec(compile(cell['source'],f'<notebook-cell-{count}>','exec'),namespace)
        if captured.stdout: cell['outputs'].append(dict(output_type='stream',name='stdout',text=captured.stdout))
        if captured.stderr: cell['outputs'].append(dict(output_type='stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            data={k:base64.b64encode(v).decode() if isinstance(v,bytes) else v for k,v in output.data.items()}
            cell['outputs'].append(dict(output_type='display_data',data=data,metadata=output.metadata))
    except Exception as exc:
        cell['outputs'].append(dict(output_type='error',ename=type(exc).__name__,evalue=str(exc),traceback=traceback.format_exc().splitlines()))
        path.write_text(json.dumps(nb,indent=1))
        raise
    path.write_text(json.dumps(nb,indent=1))
print('Executed and saved all cells:',path,flush=True)
