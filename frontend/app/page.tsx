import Link from "next/link";
import { ArrowRight, MapPinned } from "lucide-react";

import { HOME_STEPS, HOME_TRUST_NOTES } from "@/lib/constants";
import { Button } from "@/components/ui/button";

export default function HomePage() {
  return (
    <div className="pb-10">
      <section className="container py-10 md:py-12">
        <div className="max-w-3xl space-y-5">
          <div className="eyebrow">Ankara kiralık konutlar için</div>
          <div className="space-y-3">
            <h1 className="max-w-3xl text-[32px] font-semibold leading-tight tracking-[-0.04em] text-foreground md:text-[38px]">
              Ankara&apos;da eviniz için beklenen kira aralığını öğrenin
            </h1>
            <p className="max-w-2xl text-[15px] leading-7 text-muted-foreground">
              Konum, konut özellikleri, ilan açıklaması ve fotoğrafları birlikte değerlendirerek kira tahmini üretir.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <Button asChild variant="secondary" size="lg">
              <Link href="/predict">
                Kira Tahmini Yap
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Link href="/methodology" className="text-sm font-medium text-primary hover:text-[#0057B8]">
              Nasıl çalışır?
            </Link>
          </div>
        </div>
      </section>

      <section className="container">
        <div className="grid gap-3 border-y border-border py-4 md:grid-cols-3">
          {HOME_STEPS.map((item) => (
            <div key={item.title} className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#FFF8CC] text-primary">
                <item.icon className="h-4 w-4" />
              </div>
              <div>
                <div className="text-sm font-semibold text-foreground">{item.title}</div>
                <div className="text-xs text-muted-foreground">{item.description}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="container py-5">
        <div className="flex flex-col gap-2 text-sm text-muted-foreground md:flex-row md:flex-wrap md:items-center md:gap-6">
          {HOME_TRUST_NOTES.map((item) => (
            <div key={item} className="flex items-center gap-2">
              <MapPinned className="h-4 w-4 text-[#0057B8]" />
              <span>{item}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
