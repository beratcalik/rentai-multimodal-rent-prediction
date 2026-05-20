"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { MODEL_COMPARISON } from "@/lib/constants";
import { formatTry } from "@/lib/utils";

export function ModelMaeLineChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={[...MODEL_COMPARISON]}>
        <CartesianGrid strokeDasharray="4 4" stroke="#E5E7EB" vertical={false} />
        <XAxis dataKey="name" tickLine={false} axisLine={false} fontSize={12} />
        <YAxis tickLine={false} axisLine={false} fontSize={12} tickFormatter={(value: number) => `${Math.round(value / 1000)}k`} />
        <Tooltip
          formatter={(value: number) => [formatTry(value), "MAE"]}
          contentStyle={{
            borderRadius: 18,
            border: "1px solid #E5E7EB",
            background: "#FFFFFF",
            boxShadow: "0 14px 36px -22px rgba(15,23,42,0.18)",
          }}
        />
        <Line
          type="monotone"
          dataKey="mae"
          stroke="#1D4ED8"
          strokeWidth={3}
          dot={{ r: 5, fill: "#1D4ED8" }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
