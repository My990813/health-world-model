"""
将 jepa_weights.json 嵌入 index.html 的 <script> 标签中，
解决 file:// 协议下 fetch() 被 CORS 阻止的问题。
"""
import json, os, re

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
WEIGHTS_FILE = os.path.join(BASE, 'frontend', 'jepa_weights.json')
HTML_FILE = os.path.join(BASE, 'frontend', 'index.html')

def main():
    with open(WEIGHTS_FILE) as f:
        data = json.load(f)

    # 序列化 JSON，用紧凑格式减少体积
    weights_json = json.dumps(data, separators=(',', ':'))

    # 构建 script 标签内容
    script_content = f"// JEPA 权重（自动嵌入，{len(weights_json)//1024}KB）\n"
    script_content += f"const JEPA_BUILTIN_WEIGHTS = {weights_json};\n"

    # 读取 HTML
    with open(HTML_FILE) as f:
        html = f.read()

    # 替换 loadJEPAWeights 函数
    old_func = """// ---- 加载权重 ----
async function loadJEPAWeights() {
  if (JEPA_WEIGHTS) return JEPA_WEIGHTS;
  try {
    const resp = await fetch('jepa_weights.json');
    if (!resp.ok) throw new Error('无法加载 jepa_weights.json');
    const d = await resp.json();
    JEPA_WEIGHTS = d.networks;
    JEPA_STATS   = d.stats;
    console.log('[JEPA] 权重加载成功，参数量:', d.version);
    return JEPA_WEIGHTS;
  } catch(e) {
    console.warn('[JEPA] 权重加载失败，使用物理模型:', e.message);
    return null;
  }
}"""

    new_func = """// ---- 加载权重（内嵌模式，无需 fetch）----
async function loadJEPAWeights() {
  if (JEPA_WEIGHTS) return JEPA_WEIGHTS;
  try {
    if (typeof JEPA_BUILTIN_WEIGHTS !== 'undefined') {
      JEPA_WEIGHTS = JEPA_BUILTIN_WEIGHTS.networks;
      JEPA_STATS   = JEPA_BUILTIN_WEIGHTS.stats;
      console.log('[JEPA] 内嵌权重加载成功，版本:', JEPA_BUILTIN_WEIGHTS.version);
      return JEPA_WEIGHTS;
    }
  } catch(e) {
    console.warn('[JEPA] 权重加载失败:', e.message);
  }
  return null;
}"""

    if old_func in html:
        html = html.replace(old_func, new_func)
    else:
        print("WARNING: 未找到原始 loadJEPAWeights 函数，尝试备用替换")
        # 备用: 查找并替换
        pattern = r'async function loadJEPAWeights\(\)[^}]+\}'
        html = re.sub(pattern, new_func.strip(), html, count=1)

    # 在 <script> 标签后面插入内嵌权重
    insert_point = '<script>\n'
    if insert_point in html:
        html = html.replace(insert_point, insert_point + script_content + '\n', 1)
    else:
        print("WARNING: 未找到 <script> 标签")

    with open(HTML_FILE, 'w') as f:
        f.write(html)

    print(f"嵌入成功!")
    print(f"  权重 JSON: {len(weights_json)//1024} KB")
    print(f"  HTML 文件: {os.path.getsize(HTML_FILE)//1024} KB")


if __name__ == '__main__':
    main()
