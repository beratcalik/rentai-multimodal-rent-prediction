import * as React from "react";

import { cn } from "@/lib/utils";

const Textarea = React.forwardRef<HTMLTextAreaElement, React.ComponentProps<"textarea">>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[108px] w-full rounded-lg border border-border bg-surface px-3 py-2.5 text-sm text-foreground shadow-none placeholder:text-muted-foreground transition-colors focus-visible:border-[#0057B8] disabled:cursor-not-allowed disabled:bg-slate-50 disabled:opacity-70",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Textarea.displayName = "Textarea";

export { Textarea };
