export function compactNumber(value: unknown) {
  const number = Number(value ?? 0);
  return Intl.NumberFormat("en", { notation: number > 9999 ? "compact" : "standard" }).format(number);
}

export function titleCase(value: string) {
  return value.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
