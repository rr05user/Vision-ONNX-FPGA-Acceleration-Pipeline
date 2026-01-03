
import onnx
from collections import Counter

MODEL = "yolov8n.onnx"

m = onnx.load(MODEL)
ops = [n.op_type for n in m.graph.node]
c = Counter(ops)

print("Model:", MODEL)
print("\nTop ops:")
for op, cnt in c.most_common():
    print(f"  {op:25s} {cnt}")

print("\nTotal nodes:", len(ops))
print("Inputs:")
for i in m.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else "?" for d in i.type.tensor_type.shape.dim]
    print(f"  {i.name}: {shape}")

print("Outputs:")
for o in m.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else "?" for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name}: {shape}")

