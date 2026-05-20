"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MODEL_COMPARISON } from "@/lib/constants";
import { formatTry } from "@/lib/utils";

const chartPalette = ["#94A3B8", "#2563EB", "#0F766E", "#111827"];

export function ModelMaeBarChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={[...MODEL_COMPARISON]} barGap={18}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
        <XAxis dataKey="name" tickLine={false} axisLine={false} fontSize={12} />
        <YAxis tickLine={false} axisLine={false} fontSize={12} tickFormatter={(value: number) => `${Math.round(value / 1000)}k`} />
        <Tooltip
          cursor={{ fill: "rgba(148, 163, 184, 0.12)" }}
          formatter={(value: number) => [formatTry(value), "MAE"]}
          contentStyle={{
            borderRadius: 18,
            border: "1px solid #E5E7EB",
            background: "#FFFFFF",
            boxShadow: "0 14px 36px -22px rgba(15,23,42,0.18)",
          }}
        />
        <Bar dataKey="mae" radius={[16, 16, 8, 8]}>
          {MODEL_COMPARISON.map((entry, index) => (
            <Cell key={entry.name} fill={chartPalette[index]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
