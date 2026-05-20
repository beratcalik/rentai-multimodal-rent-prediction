import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type CompactSectionProps = {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
  first?: boolean;
};

export function CompactSection({ title, description, children, className, first = false }: CompactSectionProps) {
  return (
    <section className={cn(!first && "border-t border-border pt-5", className)}>
      <div className="mb-4 space-y-1">
        <h2 className="portal-section-title">{title}</h2>
        {description ? <p className="text-sm text-muted-foreground">{description}</p> : null}
      </div>
      {children}
    </section>
  );
}
