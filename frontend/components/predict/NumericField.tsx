"use client";

import type { NumericFieldMetadata } from "@/lib/meta/load-metadata";
import { cn, formatNumberTr } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type NumericFieldProps = {
  id: string;
  label: string;
  value: string;
  onValueChange: (value: string) => void;
  metadata: NumericFieldMetadata;
  errorMessage?: string;
  disabled?: boolean;
  required?: boolean;
  unitLabel?: string;
};

function formatRounded(value: number) {
  return formatNumberTr(Math.round(value));
}

export function NumericField({
  id,
  label,
  value,
  onValueChange,
  metadata,
  errorMessage,
  disabled = false,
  required = false,
  unitLabel,
}: NumericFieldProps) {
  const parsedValue = value.trim() ? Number(value) : Number.NaN;
  const isOutOfTypicalRange = Number.isFinite(parsedValue) && (parsedValue < metadata.p05 || parsedValue > metadata.p95);

  const placeholderUnit = unitLabel ? ` ${unitLabel}` : "";
  const typicalRange = `${formatRounded(metadata.p05)}–${formatRounded(metadata.p95)}${placeholderUnit}`;

  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>
        {label}
        {required ? <span className="ml-1 text-error">*</span> : null}
      </Label>

      <Input
        id={id}
        type="number"
        min={0}
        inputMode="decimal"
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        placeholder={`Örn. ${formatRounded(metadata.median)}${placeholderUnit}`}
        disabled={disabled}
        aria-invalid={Boolean(errorMessage)}
        className={cn(errorMessage && "border-error/40 focus-visible:border-error")}
      />

      {errorMessage ? <p className="text-xs text-error">{errorMessage}</p> : null}
      {!errorMessage ? <p className="text-xs text-muted-foreground">Veri setinde tipik aralık: {typicalRange}</p> : null}

      {isOutOfTypicalRange ? (
        <p className="text-xs text-warning">Bu değer tipik aralığın dışında görünüyor, yine de tahmin üretilebilir.</p>
      ) : !errorMessage ? (
        <p className="text-xs text-transparent">.</p>
      ) : null}
    </div>
  );
}
