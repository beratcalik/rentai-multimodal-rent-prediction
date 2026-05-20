import * as React from "react";

import { cn } from "@/lib/utils";

const NativeSelect = React.forwardRef<HTMLSelectElement, React.ComponentProps<"select">>(
  ({ className, children, ...props }, ref) => {
    return (
      <select
        ref={ref}
        className={cn(
          "flex h-10 w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground shadow-none transition-colors focus-visible:border-[#0057B8] disabled:cursor-not-allowed disabled:bg-slate-50 disabled:opacity-70",
          className,
        )}
        {...props}
      >
        {children}
      </select>
    );
  },
);

NativeSelect.displayName = "NativeSelect";

export { NativeSelect };
