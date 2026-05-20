"use client";

import { Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { MetadataOption } from "@/lib/meta/load-metadata";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect } from "@/components/ui/native-select";

type MetadataSelectProps = {
  id: string;
  label: string;
  value: string;
  onValueChange: (value: string) => void;
  options: MetadataOption[];
  placeholder?: string;
  helperText?: string;
  errorMessage?: string;
  disabled?: boolean;
  searchable?: boolean;
  searchPlaceholder?: string;
  emptyMessage?: string;
  required?: boolean;
};

export function MetadataSelect({
  id,
  label,
  value,
  onValueChange,
  options,
  placeholder,
  helperText,
  errorMessage,
  disabled = false,
  searchable = false,
  searchPlaceholder = "Seçenek ara",
  emptyMessage = "Eşleşen seçenek bulunamadı.",
  required = false,
}: MetadataSelectProps) {
  const [query, setQuery] = useState("");

  useEffect(() => {
    setQuery("");
  }, [options]);

  const filteredOptions = useMemo(() => {
    if (!query.trim()) {
      return options;
    }

    const normalizedQuery = query.trim().toLocaleLowerCase("tr-TR");
    return options.filter((option) => option.label.toLocaleLowerCase("tr-TR").includes(normalizedQuery));
  }, [options, query]);

  const helperNode = filteredOptions.length === 0
    ? <p className="text-xs text-muted-foreground">{emptyMessage}</p>
    : errorMessage
      ? <p className="text-xs text-error">{errorMessage}</p>
      : helperText
        ? <p className="text-xs text-muted-foreground">{helperText}</p>
        : <p className="text-xs text-transparent">.</p>;

  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>
        {label}
        {required ? <span className="ml-1 text-error">*</span> : null}
      </Label>

      {searchable ? (
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={searchPlaceholder}
            className="pl-9"
            aria-label={`${label} seçeneklerinde ara`}
            disabled={disabled}
          />
        </div>
      ) : null}

      <NativeSelect
        id={id}
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        disabled={disabled || filteredOptions.length === 0}
        aria-invalid={Boolean(errorMessage)}
        className={cn(errorMessage && "border-error/40 focus-visible:border-error")}
      >
        <option value="">{placeholder ?? `${label} seçin`}</option>
        {filteredOptions.map((option) => (
          <option key={`${option.label}-${option.value}`} value={option.value}>
            {option.label}
          </option>
        ))}
      </NativeSelect>

      {helperNode}
    </div>
  );
}
