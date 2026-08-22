"use client";

import imageCompression from "browser-image-compression";
import { AlertCircle, ImagePlus, UploadCloud, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { type FileRejection, useDropzone } from "react-dropzone";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ALLOWED_PREDICTION_IMAGE_EXTENSIONS,
  ALLOWED_PREDICTION_IMAGE_TYPES,
  MAX_PREDICTION_IMAGES,
  MAX_PREDICTION_IMAGE_SIZE_BYTES,
} from "@/lib/validation/prediction-schema";

type ImageUploaderProps = {
  files: File[];
  onChange: (files: File[]) => void;
  errorMessage?: string;
  disabled?: boolean;
  compact?: boolean;
};

type PreviewItem = {
  id: string;
  name: string;
  sizeLabel: string;
  previewUrl: string;
};

function formatKb(size: number) {
  return `${Math.round(size / 1024)} KB`;
}

async function optimizeFile(file: File) {
  try {
    const compressed = await imageCompression(file, {
      maxSizeMB: 2.8,
      maxWidthOrHeight: 1800,
      useWebWorker: true,
      initialQuality: 0.84,
    });

    if (compressed instanceof File) {
      return compressed;
    }

    const compressedBlob = compressed as Blob & { type?: string };
    return new File([compressedBlob], file.name, {
      type: compressedBlob.type || file.type,
      lastModified: Date.now(),
    });
  } catch {
    return file;
  }
}

function dedupeFiles(files: File[]) {
  const seen = new Set<string>();
  const unique: File[] = [];

  for (const file of files) {
    const key = `${file.name}-${file.size}-${file.lastModified}`;
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    unique.push(file);
  }

  return unique;
}

export function ImageUploader({ files, onChange, errorMessage, disabled = false, compact = false }: ImageUploaderProps) {
  const [localNotices, setLocalNotices] = useState<string[]>([]);

  const previews = useMemo<PreviewItem[]>(
    () =>
      files.map((file, index) => ({
        id: `${file.name}-${file.size}-${file.lastModified}-${index}`,
        name: file.name,
        sizeLabel: formatKb(file.size),
        previewUrl: URL.createObjectURL(file),
      })),
    [files],
  );

  useEffect(() => {
    return () => {
      for (const preview of previews) {
        URL.revokeObjectURL(preview.previewUrl);
      }
    };
  }, [previews]);

  const onDrop = useCallback(
    async (acceptedFiles: File[], fileRejections: FileRejection[]) => {
      const notices: string[] = [];

      if (fileRejections.length > 0) {
        notices.push(
          ...fileRejections.map(({ file, errors }) => `${file.name}: ${errors[0]?.message ?? "Dosya kabul edilmedi."}`),
        );
      }

      const remainingCapacity = Math.max(0, MAX_PREDICTION_IMAGES - files.length);
      const usableFiles = acceptedFiles.slice(0, remainingCapacity);

      if (acceptedFiles.length > remainingCapacity) {
        notices.push(`En fazla ${MAX_PREDICTION_IMAGES} görsel eklenebilir. Fazla dosyalar kullanılmadı.`);
      }

      const optimizedFiles = await Promise.all(usableFiles.map((file) => optimizeFile(file)));
      const nextFiles = dedupeFiles([...files, ...optimizedFiles]).slice(0, MAX_PREDICTION_IMAGES);

      onChange(nextFiles);
      setLocalNotices(notices);
    },
    [files, onChange],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    disabled,
    noClick: true,
    maxFiles: MAX_PREDICTION_IMAGES,
    maxSize: MAX_PREDICTION_IMAGE_SIZE_BYTES,
    accept: {
      [ALLOWED_PREDICTION_IMAGE_TYPES[0]]: [ALLOWED_PREDICTION_IMAGE_EXTENSIONS[0], ALLOWED_PREDICTION_IMAGE_EXTENSIONS[1]],
      [ALLOWED_PREDICTION_IMAGE_TYPES[1]]: [ALLOWED_PREDICTION_IMAGE_EXTENSIONS[2]],
    },
  });

  const removeItem = useCallback(
    (index: number) => {
      const nextFiles = files.filter((_, fileIndex) => fileIndex !== index);
      onChange(nextFiles);
      setLocalNotices([]);
    },
    [files, onChange],
  );

  return (
    <div id="images" className="space-y-3">
      <div
        {...getRootProps()}
        className={cn(
          "rounded-2xl border border-dashed px-4 py-5 transition-colors",
          disabled && "cursor-not-allowed opacity-60",
          isDragActive ? "border-[#0057B8] bg-[#F8FBFF]" : "border-border bg-slate-50/70",
          compact && "px-4 py-5",
        )}
      >
        <input {...getInputProps()} aria-label="Konut fotoğraflarını yükleyin" />
        <div className="flex flex-col items-center text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-primary shadow-sm">
            {isDragActive ? <UploadCloud className="h-5 w-5" /> : <ImagePlus className="h-5 w-5" />}
          </div>
          <div className="mt-3 text-sm font-semibold text-foreground">Fotoğrafları buraya sürükleyin</div>
          <div className="mt-1 text-xs leading-5 text-muted-foreground">veya</div>
          <Button type="button" variant="outline" size="sm" onClick={open} disabled={disabled} className="mt-3">
            <ImagePlus className="h-4 w-4" />
            Dosya seç
          </Button>
          <div className="mt-3 text-xs leading-5 text-muted-foreground">
            PNG, JPG, JPEG dosyaları (Maks. 10 MB / fotoğraf)
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between gap-3">
          <div className="text-sm font-semibold text-foreground">Eklenen fotoğraflar</div>
          <div className="text-xs font-medium text-muted-foreground">{previews.length}/{MAX_PREDICTION_IMAGES}</div>
        </div>

        {previews.length > 0 ? (
          <div className="rounded-2xl border border-border bg-slate-50/60 p-3">
            <div className="grid max-h-[248px] grid-cols-3 gap-2 overflow-y-auto pr-1 md:grid-cols-3 xl:grid-cols-4">
              {previews.map((preview, index) => (
                <div
                  key={preview.id}
                  className="group relative overflow-hidden rounded-xl border border-border bg-white transition duration-200 hover:-translate-y-0.5 hover:shadow-[0_10px_22px_rgba(15,23,42,0.10)]"
                >
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => removeItem(index)}
                    className="absolute right-2 top-2 z-10 h-7 w-7 rounded-full border border-[#E7D7D7] bg-white/92 text-[#7F1D1D] shadow-sm transition hover:bg-[#FFF1F1] hover:text-[#991B1B]"
                    aria-label={`${preview.name} görselini sil`}
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>

                  <div className="aspect-square overflow-hidden bg-muted">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={preview.previewUrl}
                      alt={preview.name}
                      className="h-full w-full object-cover transition duration-200 group-hover:scale-[1.04]"
                    />
                  </div>

                  <div className="space-y-1.5 p-2.5">
                    <div className="truncate text-[11px] font-medium text-foreground">{preview.name}</div>
                    <div className="text-[10px] text-muted-foreground">{preview.sizeLabel}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="rounded-2xl border border-border bg-slate-50/60 px-4 py-5 text-center">
            <div className="text-sm font-medium text-foreground">Henüz fotoğraf eklemediniz.</div>
            <div className="mt-1 text-xs leading-5 text-muted-foreground">En fazla 16 fotoğraf yükleyebilirsiniz.</div>
          </div>
        )}
      </div>

      {(localNotices.length > 0 || errorMessage) && (
        <div className="space-y-2">
          {localNotices.map((notice) => (
            <div
              key={notice}
              className="flex items-start gap-2 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning"
            >
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{notice}</span>
            </div>
          ))}
          {errorMessage ? <p className="text-xs text-error">{errorMessage}</p> : null}
        </div>
      )}
    </div>
  );
}
