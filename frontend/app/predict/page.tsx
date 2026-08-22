import { PredictHero } from "@/components/predict/PredictHero";
import { PredictWorkspace } from "@/components/predict/PredictWorkspace";

export default function PredictPage() {
  return (
    <div className="mx-auto w-[calc(100%_-_32px)] max-w-[1500px] py-4 sm:w-[calc(100%_-_48px)] sm:py-5 lg:w-[calc(100%_-_96px)] lg:py-6">
      <div className="space-y-4">
        <PredictHero />
        <PredictWorkspace />
      </div>
    </div>
  );
}
