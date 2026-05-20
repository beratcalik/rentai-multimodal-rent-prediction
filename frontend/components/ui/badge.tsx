import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center rounded-md border px-2.5 py-1 text-[11px] font-semibold", {
  variants: {
    variant: {
      default: "border-border bg-muted text-foreground",
      primary: "border-[#0B3A75]/12 bg-[#F8FBFF] text-[#0B3A75]",
      secondary: "border-[#FFD200]/70 bg-[#FFF8CC] text-[#5B4A00]",
      success: "border-success/15 bg-success/10 text-success",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
