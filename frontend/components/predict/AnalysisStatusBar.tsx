"use client";

import { Camera, FileText, MapPinned, SlidersHorizontal } from "lucide-react";

import { cn } from "@/lib/utils";

export type AnalysisStatus = "ready" | "missing" | "optional";

export type PredictionAnalysisItem = {
  label: string;
  status: AnalysisStatus;
};

type AnalysisStatusBarProps = {
  items: PredictionAnalysisItem[];
};

const STATUS_COPY: Record<AnalysisStatus, { label: string; className: string }> = {
  ready: {
    label: "Hazır",
    className: "border border-success/15 bg-success/10 text-success",
  },
  missing: {
    label: "Eksik",
    className: "border border-warning/15 bg-warning/10 text-warning",
  },
  optional: {
    label: "Opsiyonel",
    className: "border border-slate-200 bg-slate-100 text-muted-foreground",
  },
};

const ITEM_META = {
  "Konum bilgileri": {
    description: "Şehir, ilçe, mahalle",
    icon: MapPinned,
  },
  "Konut özellikleri": {
    description: "Oda, m², kat, ısıtma, vb.",
    icon: SlidersHorizontal,
  },
  "Konut fotoğraflarını yükle": {
    description: "En fazla 16 fotoğraf",
    icon: Camera,
  },
  "İlan açıklaması": {
    description: "Başlık ve açıklama metni",
    icon: FileText,
  },
} as const;

function AnalysisConnector() {
  return (
    <div
      aria-hidden="true"
      className="hidden min-w-[34px] shrink-0 items-center justify-center gap-1.5 xl:flex"
    >
      <span className="h-1 w-1 rounded-full bg-[#C8D5E5]" />
      <span className="h-px w-6 bg-[#D8E1EC]" />
      <span className="h-1 w-1 rounded-full bg-[#C8D5E5]" />
    </div>
  );
}

export function AnalysisStatusBar({ items }: AnalysisStatusBarProps) {
  return (
    <section className="rounded-[18px] border border-[#DDE7F0] bg-white px-6 py-5 shadow-[0_14px_36px_rgba(14,42,89,0.06)]">
      <div className="mb-5 text-[11px] font-semibold uppercase tracking-[0.2em] text-primary">
        Analizde kullanılacaklar
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:flex xl:items-stretch xl:gap-0">
        {items.map((item, index) => {
          const meta = ITEM_META[item.label as keyof typeof ITEM_META] ?? ITEM_META["Konut özellikleri"];
          const statusCopy = STATUS_COPY[item.status];
          const Icon = meta.icon;

          return (
            <div key={item.label} className="contents">
              <div className="min-w-0 rounded-[16px] border border-[#E8EEF5] bg-[#FBFDFF] px-4 py-4 xl:flex-1 xl:border-none xl:bg-transparent xl:px-2 xl:py-0">
                <div className="grid min-w-0 grid-cols-[56px_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2">
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-[#F4F8FF] text-primary shadow-[inset_0_0_0_1px_rgba(11,58,117,0.06)]">
                    <Icon className="h-5 w-5" />
                  </div>

                  <div className="min-w-0">
                    <div className="truncate text-[15px] font-semibold text-foreground">{item.label}</div>
                    <div className="truncate text-[13px] leading-5 text-muted-foreground">{meta.description}</div>
                  </div>

                  <div className="justify-self-start xl:justify-self-end">
                    <span
                      className={cn(
                        "inline-flex whitespace-nowrap rounded-full px-2.5 py-1 text-[11px] font-semibold",
                        statusCopy.className,
                      )}
                    >
                      {statusCopy.label}
                    </span>
                  </div>
                </div>
              </div>

              {index < items.length - 1 ? <AnalysisConnector /> : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}
