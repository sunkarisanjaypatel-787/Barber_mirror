# Save as: modularize_engine.py
with open("GoldenVectorEngine.kt", "r") as f:
    content = f.read()

# Split the content by "// Tree X"
trees = content.split("// Tree ")

header = trees[0]
tree_data = trees[1:]

new_kotlin = header + "\n    fun predictShape(...) {\n"
for i in range(len(tree_data)):
    new_kotlin += f"        votes[tree{i}(h_w, j_w, f_j, c_j)]++\n"
new_kotlin += "    }\n\n"

for i, tree in enumerate(tree_data):
    new_kotlin += f"    private fun tree{i}(h_w: Float, j_w: Float, f_j: Float, c_j: Float): Int {{\n"
    # Logic to extract the shape index from the "votes[X]++" line
    # ...
    new_kotlin += "    }\n\n"

with open("GoldenVectorEngine_Fixed.kt", "w") as f:
    f.write(new_kotlin)
