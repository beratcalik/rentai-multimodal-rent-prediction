"use client";

import { CheckCircle2 } from "lucide-react";

import { PREDICTION_STEPS } from "@/lib/constants";
import { cn } from "@/lib/utils";

type StepState = "pending" | "active" | "complete";

type PredictProgressProps = {
  steps: Record<string, StepState>;
};

export function PredictProgress({ steps }: PredictProgressProps) {
  return (
    <div className="grid gap-3 md:grid-cols-5">
      {PREDICTION_STEPS.map((step, index) => {
        const status = steps[step.id] ?? "pending";
        const isComplete = status === "complete";
        const isActive = status === "active";

        return (
          <div
            key={step.id}
            className={cn(
              "rounded-[22px] border px-4 py-4 transition-colors",
              isComplete && "border-success/20 bg-success/10",
              isActive && "border-primary/20 bg-primary/8",
              status === "pending" && "border-border bg-white/82",
            )}
          >
            <div className="mb-3 flex items-center justify-between gap-3">
              <div
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-2xl",
                  isComplete && "bg-success text-white",
                  isActive && "bg-primary text-white",
                  status === "pending" && "bg-muted text-muted-foreground",
                )}
              >
                {isComplete ? <CheckCircle2 className="h-4 w-4" /> : <step.icon className="h-4 w-4" />}
              </div>
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                {String(index + 1).padStart(2, "0")}
              </div>
            </div>
            <div className="text-sm font-semibold text-foreground">{step.title}</div>
          </div>
        );
      })}
    </div>
  );
}
