import { METHODOLOGY_DATA_TYPES, METHODOLOGY_LIMITATIONS, METHODOLOGY_STEPS, METHODOLOGY_TECHNICAL_NOTES } from "@/lib/constants";
import { SectionHeading } from "@/components/shared/SectionHeading";
import { Card, CardContent } from "@/components/ui/card";

export default function MethodologyPage() {
  return (
    <div className="container py-10 md:py-12">
      <div className="space-y-8">
        <SectionHeading
          eyebrow="Nasıl Çalışır"
          title="Tahmin akışı kısa ve anlaşılır bir karar desteği mantığıyla çalışır"
          description="Sistem, ilanın konumunu, konut özelliklerini, açıklama metnini ve fotoğraflarını birlikte değerlendirerek kira tahmini üretir."
        />

        <section className="grid gap-4 md:grid-cols-3">
          {METHODOLOGY_STEPS.map((item, index) => (
            <Card key={item.title} className="bg-white">
              <CardContent className="flex gap-3 p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#FFF8CC] text-primary">
                  <item.icon className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Adım {index + 1}</div>
                  <div className="text-[15px] font-semibold text-foreground">{item.title}</div>
                  <p className="text-sm leading-6 text-muted-foreground">{item.description}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </section>

        <section className="space-y-4">
          <div className="space-y-1">
            <h2 className="text-[24px] font-semibold tracking-[-0.03em] text-foreground">Kullanılan veri türleri</h2>
            <p className="text-[15px] leading-6 text-muted-foreground">Ürün, tek bir veri alanına değil; birden fazla bilgi kaynağına birlikte bakar.</p>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {METHODOLOGY_DATA_TYPES.map((item) => (
              <Card key={item.title} className="bg-white">
                <CardContent className="space-y-3 p-5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-primary">
                    <item.icon className="h-4 w-4" />
                  </div>
                  <div className="text-[15px] font-semibold text-foreground">{item.title}</div>
                  <p className="text-sm leading-6 text-muted-foreground">{item.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="space-y-4">
          <div className="space-y-1">
            <h2 className="text-[24px] font-semibold tracking-[-0.03em] text-foreground">Sınırlamalar</h2>
            <p className="text-[15px] leading-6 text-muted-foreground">Çıktı, hızlı ön değerlendirme ve karar desteği içindir; resmi ekspertiz yerine geçmez.</p>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            {METHODOLOGY_LIMITATIONS.map((item) => (
              <div key={item} className="rounded-xl border border-border bg-white px-4 py-4 text-sm leading-6 text-muted-foreground">
                {item}
              </div>
            ))}
          </div>
        </section>

        <section className="space-y-4">
          <div className="space-y-1">
            <h2 className="text-[24px] font-semibold tracking-[-0.03em] text-foreground">Teknik notlar</h2>
            <p className="text-[15px] leading-6 text-muted-foreground">Daha teknik detaylar aşağıda özetlenmiştir. Ana ürün deneyiminde bu bilgiler geri planda tutulur.</p>
          </div>

          <div className="space-y-3">
            {METHODOLOGY_TECHNICAL_NOTES.map((item) => (
              <details key={item.title} className="rounded-xl border border-border bg-white">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-5 py-4 text-sm font-semibold text-foreground">
                  <span>{item.title}</span>
                  <span className="text-xs font-medium text-muted-foreground">{item.summary}</span>
                </summary>
                <div className="border-t border-border px-5 py-4">
                  <div className="space-y-2">
                    {item.details.map((detail) => (
                      <p key={detail} className="text-sm leading-6 text-muted-foreground">
                        {detail}
                      </p>
                    ))}
                  </div>
                </div>
              </details>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
