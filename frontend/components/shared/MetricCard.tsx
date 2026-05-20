import type { LucideIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type MetricCardProps = {
  label: string;
  value: string;
  caption: string;
  icon: LucideIcon;
  tone?: "default" | "primary" | "secondary" | "success";
};

export function MetricCard({ label, value, caption, icon: Icon, tone = "default" }: MetricCardProps) {
  return (
    <Card className="h-full border-white/70 bg-white/90">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between gap-3">
          <Badge variant={tone}>{label}</Badge>
          <div className="rounded-2xl border border-primary/10 bg-primary/5 p-2 text-primary">
            <Icon className="h-4 w-4" aria-hidden="true" />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-3xl font-semibold tracking-[-0.05em] text-foreground">{value}</div>
        <p className="text-sm leading-6 text-muted-foreground">{caption}</p>
      </CardContent>
    </Card>
  );
}
