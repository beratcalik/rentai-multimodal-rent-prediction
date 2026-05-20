import { PredictWorkspace } from "@/components/predict/PredictWorkspace";

export default function PredictPage() {
  return (
    <div className="container py-8 md:py-10">
      <div className="mb-5 space-y-1.5">
        <h1 className="text-[28px] font-semibold tracking-[-0.03em] text-foreground">Kira tahmini</h1>
        <p className="text-[15px] leading-6 text-muted-foreground">Bilgileri girin, beklenen kira aralığını görün.</p>
      </div>

      <PredictWorkspace />
    </div>
  );
}
