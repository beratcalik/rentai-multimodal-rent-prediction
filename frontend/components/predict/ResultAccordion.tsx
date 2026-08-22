"use client";

import { ChevronDown } from "lucide-react";
import { type ReactNode, useState } from "react";

import { cn } from "@/lib/utils";

type ResultAccordionProps = {
  title: string;
  count?: number;
  children: ReactNode;
  defaultOpen?: boolean;
};

export function ResultAccordion({ title, count, children, defaultOpen = false }: ResultAccordionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="rounded-xl border border-border bg-white">
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
        aria-expanded={open}
      >
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-sm font-semibold text-foreground">{title}</span>
          {typeof count === "number" ? (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-muted-foreground">
              {count}
            </span>
          ) : null}
        </div>

        <ChevronDown
          className={cn("h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")}
        />
      </button>

      {open ? <div className="border-t border-border px-4 py-4">{children}</div> : null}
    </div>
  );
}
