import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type SectionHeadingProps = {
  eyebrow?: string;
  title: string;
  description: string;
  align?: "left" | "center";
  action?: ReactNode;
  className?: string;
};

export function SectionHeading({
  eyebrow,
  title,
  description,
  align = "left",
  action,
  className,
}: SectionHeadingProps) {
  return (
    <div
      className={cn(
        "flex flex-col gap-4 md:flex-row md:items-end md:justify-between",
        align === "center" && "items-center text-center md:flex-col md:items-center",
        className,
      )}
    >
      <div className="max-w-2xl space-y-2.5">
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <div className="space-y-2">
          <h2 className="text-[28px] font-semibold tracking-[-0.04em] text-foreground md:text-[34px]">{title}</h2>
          <p className="text-[15px] leading-6 text-muted-foreground">{description}</p>
        </div>
      </div>
      {action ? <div>{action}</div> : null}
    </div>
  );
}
