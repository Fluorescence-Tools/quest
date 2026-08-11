export function getNestedValue(obj: Record<string, any>, path: string): any {
  if (!obj || !path) return undefined;
  // Handle direct flat alias keys first
  if (path in obj) return obj[path];

  const parts = path.split(".");
  let curr: any = obj;
  for (const part of parts) {
    if (curr === null || curr === undefined) return undefined;
    curr = curr[part];
  }
  return curr;
}

export function setNestedValue(obj: Record<string, any>, path: string, value: any): Record<string, any> {
  const copy = JSON.parse(JSON.stringify(obj || {}));
  const parts = path.split(".");
  let curr = copy;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (!(part in curr) || typeof curr[part] !== "object") {
      curr[part] = {};
    }
    curr = curr[part];
  }
  curr[parts[parts.length - 1]] = value;
  return copy;
}
