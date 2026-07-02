import ReactECharts from "echarts-for-react";
import { motion } from "framer-motion";
import { Activity, Gauge } from "lucide-react";
import type { ChartSpec } from "../api/client";

const matrixTypes = new Set(["heatmap", "correlation_matrix", "correlation_heatmap", "confusion_matrix", "matrix"]);

function labelText(value: unknown) {
  return String(value ?? "");
}

function longestLabel(values: unknown[]) {
  return values.reduce<number>((max, value) => Math.max(max, labelText(value).length), 0);
}

function isMatrixChart(chart: ChartSpec) {
  return matrixTypes.has(chart.chart_type);
}

function abbreviateLabel(value: unknown, max = 16) {
  const text = labelText(value);
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(4, max - 1))}…`;
}

function matrixLabels(chart: ChartSpec) {
  const stats = chart.computed_stats as { x_labels?: string[]; y_labels?: string[] } | undefined;
  const xLabels: string[] = stats?.x_labels ?? Array.from(new Set(chart.data.map((d) => String(d.x_label ?? d.x))));
  const yLabels: string[] = stats?.y_labels ?? Array.from(new Set(chart.data.map((d) => String(d.y_label ?? d.y))));
  return { xLabels, yLabels };
}

function matrixCellSize(xCount: number, yCount: number) {
  const maxCount = Math.max(xCount, yCount);
  if (maxCount <= 7) return 46;
  if (maxCount <= 12) return 40;
  if (maxCount <= 18) return 34;
  if (maxCount <= 26) return 29;
  return 25;
}

function matrixLayout(chart: ChartSpec) {
  const { xLabels, yLabels } = matrixLabels(chart);
  const cell = matrixCellSize(xLabels.length, yLabels.length);
  const rotateX = longestLabel(xLabels) > 8 || xLabels.length > 7;
  const fontSize = Math.max(10, Math.min(12, cell * 0.34));
  const left = Math.min(260, Math.max(86, longestLabel(yLabels) * 7 + 28));
  const right = 44;
  const top = 34;
  const bottom = Math.min(170, Math.max(74, rotateX ? longestLabel(xLabels) * 4.8 + 64 : 76));
  const plotWidth = Math.max(1, xLabels.length) * cell;
  const plotHeight = Math.max(1, yLabels.length) * cell;
  const totalWidth = Math.ceil(left + plotWidth + right);
  const totalHeight = Math.ceil(top + plotHeight + bottom);
  return { xLabels, yLabels, cell, rotateX, fontSize, left, right, top, bottom, plotWidth, plotHeight, totalWidth, totalHeight };
}

function optionFor(chart: ChartSpec) {
  const xKey = chart.encoding.x;
  const yKey = chart.encoding.y;
  const text = "#dbeafe";
  const baseTooltip = { backgroundColor: "rgba(5,10,24,.96)", borderColor: "rgba(103,232,249,.22)", textStyle: { color: "#e2e8f0" }, extraCssText: "box-shadow:0 18px 50px rgba(0,0,0,.35);border-radius:10px;" };
  const axis = { axisLabel: { color: "#94a3b8", hideOverlap: true }, axisLine: { lineStyle: { color: "rgba(148,163,184,.25)" } }, splitLine: { lineStyle: { color: "rgba(148,163,184,.12)" } } };
  const rawCategories = chart.data.map((d) => d[xKey] ?? d.value ?? d.category ?? d.pair ?? d.metric);
  const longX = longestLabel(rawCategories);
  const grid = { top: 30, left: 62, right: 34, bottom: longX > 18 ? 108 : longX > 12 ? 84 : 62, containLabel: true };
  const renderAsHorizontal = ["bar", "category_bar"].includes(chart.chart_type) && longX > 16;

  if (chart.chart_type === "scatter") {
    return {
      backgroundColor: "transparent",
      tooltip: { ...baseTooltip, trigger: "item" },
      grid: { ...grid, left: 70, bottom: 64 },
      xAxis: { ...axis, type: "value" },
      yAxis: { ...axis, type: "value" },
      series: [{ type: "scatter", data: chart.data.map((d) => [d[xKey] ?? d.x, d[yKey] ?? d.y]), symbolSize: 8, itemStyle: { color: "#67e8f9", opacity: 0.82 }, emphasis: { scale: 1.8 } }],
    };
  }
  if (chart.chart_type === "line" || chart.chart_type === "area" || chart.chart_type === "density") {
    const isArea = chart.chart_type === "area";
    return {
      backgroundColor: "transparent",
      tooltip: { ...baseTooltip, trigger: "axis" },
      grid: { ...grid, left: 64, right: 36 },
      xAxis: { ...axis, type: "category", data: chart.data.map((d) => d[xKey]), axisLabel: { ...axis.axisLabel, rotate: longX > 12 ? 28 : 0 } },
      yAxis: { ...axis, type: "value" },
      series: [{ type: "line", smooth: true, data: chart.data.map((d) => d[yKey]), lineStyle: { color: isArea ? "#22d3ee" : "#a78bfa", width: 3 }, areaStyle: isArea ? { color: "rgba(34,211,238,.16)" } : undefined }],
    };
  }
  if (chart.chart_type === "boxplot") {
    return {
      backgroundColor: "transparent",
      tooltip: { ...baseTooltip, trigger: "item" },
      grid: { ...grid, bottom: longestLabel(chart.data.map((d) => d.label)) > 12 ? 88 : 58 },
      xAxis: { ...axis, type: "category", data: chart.data.map((d) => d.label), axisLabel: { ...axis.axisLabel, rotate: longestLabel(chart.data.map((d) => d.label)) > 12 ? 24 : 0 } },
      yAxis: { ...axis, type: "value" },
      series: [{ type: "boxplot", data: chart.data.map((d) => [d.min, d.q1, d.median, d.q3, d.max]), itemStyle: { color: "rgba(34,211,238,.18)", borderColor: "#67e8f9" } }],
    };
  }
  if (chart.chart_type === "donut") {
    const x = chart.encoding.x;
    const y = chart.encoding.y;
    return {
      backgroundColor: "transparent",
      tooltip: { ...baseTooltip, trigger: "item" },
      legend: { bottom: 0, type: "scroll", textStyle: { color: "#94a3b8" }, pageIconColor: "#67e8f9", pageTextStyle: { color: "#94a3b8" } },
      series: [{
        type: "pie",
        radius: ["50%", "72%"],
        center: ["50%", "44%"],
        data: chart.data.map((d) => ({ name: String(d[x] ?? d.value), value: d[y] ?? d.count })),
        label: { color: "#dbeafe", overflow: "truncate", width: 100 },
        itemStyle: { borderColor: "#08111f", borderWidth: 2 },
      }],
    };
  }
  if (isMatrixChart(chart)) {
    const { xLabels, yLabels, rotateX, fontSize, left, right, top, bottom } = matrixLayout(chart);
    const values = chart.data.map((d) => Number(d.value ?? 0));
    const maxAbs = Math.max(0.1, ...values.map((v) => Math.abs(v)));
    const labelIndex = (value: unknown, labels: string[]) => {
      if (typeof value === "number") return value;
      const found = labels.indexOf(String(value));
      return found >= 0 ? found : 0;
    };
    return {
      backgroundColor: "transparent",
      tooltip: { ...baseTooltip, position: "top", formatter: (p: { data: [number | string, number | string, number] }) => {
        const x = labelIndex(p.data[0], xLabels);
        const y = labelIndex(p.data[1], yLabels);
        return `${xLabels[x] ?? p.data[0]} / ${yLabels[y] ?? p.data[1]}: ${p.data[2]}`;
      } },
      grid: { top, left, right, bottom, containLabel: true },
      xAxis: {
        ...axis,
        type: "category",
        data: xLabels,
        axisLabel: {
          color: "#94a3b8",
          interval: 0,
          hideOverlap: false,
          rotate: rotateX ? 38 : 0,
          fontSize,
          overflow: "truncate",
          width: rotateX ? 76 : 92,
          formatter: (value: string) => abbreviateLabel(value, rotateX ? 12 : 16),
        },
      },
      yAxis: {
        ...axis,
        type: "category",
        data: yLabels,
        axisLabel: {
          color: "#94a3b8",
          interval: 0,
          hideOverlap: false,
          fontSize,
          overflow: "truncate",
          width: Math.max(70, left - 36),
          formatter: (value: string) => abbreviateLabel(value, 20),
        },
      },
      visualMap: { min: -maxAbs, max: maxAbs, calculable: false, orient: "horizontal", left: "center", bottom: 4, textStyle: { color: "#94a3b8", fontSize: 10 }, inRange: { color: ["#7c3aed", "#111827", "#22d3ee"] } },
      series: [{
        type: "heatmap",
        data: chart.data.map((d) => [labelIndex(d.x, xLabels), labelIndex(d.y, yLabels), d.value]),
        label: { show: false },
        emphasis: { itemStyle: { borderColor: "#dbeafe", borderWidth: 1 } },
      }],
    };
  }
  if (chart.chart_type === "stacked_bar") {
    const x = chart.encoding.x;
    const categories = chart.data.map((d) => String(d[x] ?? d.category));
    const seriesKeys = Object.keys(chart.data[0] ?? {}).filter((key) => !["category", x].includes(key) && !key.endsWith("_percent"));
    const colors = ["#22d3ee", "#a78bfa", "#34d399", "#f59e0b", "#f472b6", "#60a5fa", "#f87171", "#c084fc"];
    return {
      backgroundColor: "transparent",
      tooltip: { ...baseTooltip, trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { bottom: 0, textStyle: { color: "#94a3b8" } },
      grid: { top: 30, left: 66, right: 28, bottom: longestLabel(categories) > 14 ? 108 : 74, containLabel: true },
      xAxis: { ...axis, type: "category", data: categories, axisLabel: { color: "#94a3b8", rotate: categories.length > 5 || longestLabel(categories) > 12 ? 28 : 0, hideOverlap: true } },
      yAxis: { ...axis, type: "value" },
      series: seriesKeys.map((key, index) => ({ name: key, type: "bar", stack: "total", data: chart.data.map((d) => d[key]), itemStyle: { color: colors[index % colors.length] } })),
    };
  }
  const categories = chart.data.map((d) => d[xKey] ?? d.value ?? d.category ?? d.pair ?? d.metric);
  const values = chart.data.map((d) => d[yKey] ?? d.count ?? d.mean ?? d.correlation ?? d.value);
  if (chart.chart_type === "horizontal_bar" || renderAsHorizontal) {
    return {
      backgroundColor: "transparent",
      tooltip: { ...baseTooltip, trigger: "axis" },
      grid: { top: 24, left: Math.min(280, 92 + longestLabel(categories) * 6), right: 36, bottom: 42, containLabel: true },
      xAxis: { ...axis, type: "value" },
      yAxis: { ...axis, type: "category", data: categories, axisLabel: { color: "#94a3b8", overflow: "truncate", width: 190 } },
      series: [{ type: "bar", data: values, barMaxWidth: 22, itemStyle: { borderRadius: [0, 6, 6, 0], color: "#22d3ee" } }],
    };
  }
  return {
    backgroundColor: "transparent",
    textStyle: { color: text },
    tooltip: { ...baseTooltip, trigger: "axis" },
    grid,
    xAxis: { ...axis, type: "category", data: categories, axisLabel: { ...axis.axisLabel, rotate: categories.length > 6 || longX > 12 ? 28 : 0, hideOverlap: true, overflow: "truncate", width: longX > 12 ? 118 : undefined } },
    yAxis: { ...axis, type: "value" },
    series: [{ type: "bar", data: values, barMaxWidth: 38, itemStyle: { borderRadius: [6, 6, 0, 0], color: chart.chart_type === "correlation_bar" ? "#a78bfa" : "#22d3ee" } }],
  };
}

export function chartGridClass(chart: ChartSpec) {
  const type = chart.chart_type;
  const dataLen = chart.data.length;
  if (isMatrixChart(chart)) return "lg:col-span-2 2xl:col-span-2";
  if (type === "scatter") return "xl:col-span-2 2xl:col-span-2";
  if (type === "line" || type === "area" || type === "stacked_bar") return "xl:col-span-2 2xl:col-span-2";
  if (type === "horizontal_bar" && dataLen > 8) return "xl:col-span-2";
  if (type === "donut" || type === "pie") return "";
  if (type === "boxplot" || type === "histogram" || type === "density") return "";
  return dataLen > 10 ? "xl:col-span-2 2xl:col-span-1" : "";
}

function chartHeight(chart: ChartSpec) {
  if (isMatrixChart(chart)) return "100%";
  if (chart.chart_type === "scatter") return "100%";
  if (chart.chart_type === "donut" || chart.chart_type === "pie") return "100%";
  if (chart.chart_type === "boxplot") return "100%";
  if (chart.chart_type === "line" || chart.chart_type === "area" || chart.chart_type === "stacked_bar") return "100%";
  if (chart.chart_type === "horizontal_bar" && chart.data.length > 8) return Math.min(390, 230 + chart.data.length * 15);
  return "100%";
}

function chartFrameClass(chart: ChartSpec) {
  if (isMatrixChart(chart)) return "chart-frame chart-frame-matrix";
  if (chart.chart_type === "scatter") return "chart-frame chart-frame-scatter";
  if (chart.chart_type === "donut" || chart.chart_type === "pie") return "chart-frame chart-frame-pie";
  if (chart.chart_type === "boxplot") return "chart-frame chart-frame-boxplot";
  if (chart.chart_type === "line" || chart.chart_type === "area" || chart.chart_type === "stacked_bar") return "chart-frame chart-frame-bar";
  if (chart.chart_type === "horizontal_bar" && chart.data.length > 8) return "chart-frame chart-frame-bar";
  return "chart-frame chart-frame-bar";
}

function chartFrameStyle(chart: ChartSpec) {
  if (!isMatrixChart(chart)) return undefined;
  const layout = matrixLayout(chart);
  return {
    width: `${layout.totalWidth}px`,
    height: `${layout.totalHeight}px`,
  };
}

export default function ChartCard({ chart, index = 0 }: { chart: ChartSpec; index?: number }) {
  return (
    <motion.article initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(index * 0.045, 0.35), duration: 0.38 }} className="premium-card flex flex-col p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">{chart.title}</h3>
          <p className="mt-1 text-sm text-slate-400">{chart.why}</p>
        </div>
        <span className="flex items-center gap-1 rounded-full border border-cyan-300/20 px-2 py-1 text-xs text-cyan-100">
          <Gauge size={13} />
          {Math.round(chart.usefulness_score * 100)}%
        </span>
      </div>
      <div className={isMatrixChart(chart) ? "chart-shell chart-shell-matrix" : "chart-shell"}>
        <div className={chartFrameClass(chart)} style={chartFrameStyle(chart)}>
          <ReactECharts option={optionFor(chart)} style={{ height: chartHeight(chart), width: "100%" }} opts={{ renderer: "canvas" }} notMerge lazyUpdate />
        </div>
      </div>
      <div className="mt-4 rounded-xl border border-white/8 bg-white/[0.035] p-4">
        <div className="mb-2 flex items-center gap-2 text-sm font-medium text-violet-100">
          <Activity size={15} />
          AI Insight
        </div>
        <p className="text-sm leading-6 text-slate-300">{chart.insight}</p>
        {chart.notes?.map((note) => <p key={note} className="mt-2 text-xs text-cyan-100/70">{note}</p>)}
        {chart.caveat && <p className="mt-2 text-xs text-slate-500">{chart.caveat}</p>}
      </div>
      <details className="mt-3 text-xs text-slate-500">
        <summary className="cursor-pointer">Dev render diagnostics</summary>
        <pre className="mt-2 overflow-auto rounded-md bg-black/30 p-3">{JSON.stringify(chart.render_diagnostics, null, 2)}</pre>
      </details>
    </motion.article>
  );
}
