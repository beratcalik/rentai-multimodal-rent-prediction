import Link from "next/link";
import { ArrowRight, MapPinned } from "lucide-react";

import { HOME_STEPS, HOME_TRUST_NOTES } from "@/lib/constants";
import { Button } from "@/components/ui/button";

export default function HomePage() {
  return (
    <div className="overflow-hidden">
      <section className="mx-auto flex min-h-[calc(100vh-7.75rem)] w-[calc(100%_-_32px)] max-w-[1500px] flex-col justify-center py-8 sm:w-[calc(100%_-_48px)] md:min-h-[calc(100vh-8.5rem)] md:py-10 lg:w-[calc(100%_-_96px)]">
        <div className="grid items-center gap-8 lg:grid-cols-[minmax(0,1.08fr)_440px] lg:gap-10">
          <div className="space-y-5">
            <div className="eyebrow">Ankara kiralık konutlar için</div>
            <div className="space-y-3">
              <h1 className="max-w-3xl text-[34px] font-semibold leading-tight tracking-[-0.05em] text-foreground md:text-[46px]">
                Ankara&apos;da eviniz için beklenen kira aralığını öğrenin
              </h1>
              <p className="max-w-2xl text-[15px] leading-7 text-muted-foreground md:text-base">
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

          <div className="relative hidden h-[280px] rounded-[28px] border border-border bg-white lg:block">
            <div className="absolute left-10 top-10 h-12 w-12 rounded-2xl bg-[#EAF2FF]" />
            <div className="absolute left-16 top-16 h-20 w-28 rounded-xl bg-[#F5F8FC]" />
            <div className="absolute left-20 top-12 h-8 w-32 -skew-x-12 rounded-md bg-[#87AEDD]" />
            <div className="absolute left-24 top-24 h-8 w-8 rounded-sm bg-white shadow-sm" />
            <div className="absolute left-36 top-24 h-8 w-8 rounded-sm bg-white shadow-sm" />
            <div className="absolute left-36 top-36 h-12 w-5 rounded-sm bg-[#D2B06A]" />
            <div className="absolute left-8 top-28 h-16 w-10 rounded-t-full bg-[#DDE8F5]" />
            <div className="absolute left-11 top-36 h-10 w-4 rounded-t-full bg-[#6B9AA6]" />
            <div className="absolute left-[3.125rem] top-44 h-8 w-1 rounded-full bg-[#7C6551]" />
            <div className="absolute right-10 top-8 h-8 w-8 rounded-full bg-[#FFD86B]" />
            <div className="absolute bottom-10 right-12 left-[48%] h-px bg-[#D8E1EE]" />
            <div className="absolute bottom-10 right-16 h-20 w-8 rounded-t-lg bg-[#D9E7F6]" />
            <div className="absolute bottom-10 right-28 h-14 w-7 rounded-t-lg bg-[#E5EEF8]" />
            <div className="absolute bottom-10 right-40 h-10 w-6 rounded-t-lg bg-[#D9E7F6]" />
            <div className="absolute bottom-10 right-52 h-16 w-7 rounded-t-lg bg-[#E5EEF8]" />
            <div className="absolute bottom-24 right-20 flex h-9 w-9 items-center justify-center rounded-full bg-[#FFD86B] text-primary shadow-sm">
              <MapPinned className="h-4 w-4" />
            </div>
          </div>
        </div>

        <div className="mt-7 rounded-2xl border border-border bg-white px-4 py-4 shadow-[0_14px_36px_rgba(14,42,89,0.05)] sm:px-5">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {HOME_STEPS.map((item, index) => (
              <div key={item.title} className="flex items-center gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#FFF8CC] text-primary">
                  <item.icon className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-[13px] font-semibold text-foreground">
                    {index + 1}. {item.title}
                  </div>
                  <div className="text-xs leading-5 text-muted-foreground">{item.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-2 text-sm text-muted-foreground md:flex-row md:flex-wrap md:items-center md:gap-6">
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
