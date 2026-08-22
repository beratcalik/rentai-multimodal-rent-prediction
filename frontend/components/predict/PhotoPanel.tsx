"use client";

import { Camera, LoaderCircle, Lock } from "lucide-react";

import { ImageUploader } from "@/components/predict/ImageUploader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type PhotoPanelProps = {
  files: File[];
  onChange: (files: File[]) => void;
  errorMessage?: string;
  disabled?: boolean;
  pending?: boolean;
};

export function PhotoPanel({ files, onChange, errorMessage, disabled = false, pending = false }: PhotoPanelProps) {
  return (
    <Card className="overflow-hidden rounded-[18px] border-[#DDE7F0] bg-white shadow-[0_14px_36px_rgba(14,42,89,0.06)] lg:sticky lg:top-20">
      <CardHeader className="border-b border-border px-6 py-5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#F4F8FF] text-primary">
              <Camera className="h-4.5 w-4.5" />
            </div>
            <div className="space-y-1">
              <CardTitle className="text-[18px]">Konut fotoğrafları</CardTitle>
              <CardDescription>Net ve güncel fotoğraflar tahmin kalitesini artırır.</CardDescription>
            </div>
          </div>

          <div className="rounded-full bg-[#EEF3FA] px-3 py-1 text-xs font-semibold text-muted-foreground">{files.length}/16</div>
        </div>
      </CardHeader>

      <CardContent className="space-y-5 px-6 py-5">
        <ImageUploader files={files} onChange={onChange} errorMessage={errorMessage} disabled={disabled} compact />

        <div className="border-t border-border pt-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex min-w-0 items-start gap-3">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#F4F8FF] text-primary">
                <Lock className="h-4 w-4" />
              </div>
              <div className="space-y-1">
                <div className="text-sm font-medium text-foreground">Girdiğiniz bilgiler gizli ve güvendedir.</div>
                <div className="text-xs leading-5 text-muted-foreground">Fotoğraf eklemek zorunlu değildir, ancak tahmin kalitesini artırabilir.</div>
              </div>
            </div>

            <Button type="submit" form="prediction-form" variant="default" size="lg" disabled={pending} className="hidden min-w-[188px] md:inline-flex">
              {pending ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Tahmin hazırlanıyor
                </>
              ) : (
                "Tahmini Al"
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
