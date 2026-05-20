"use client";

import imageCompression from "browser-image-compression";
import { AlertCircle, ArrowLeft, ArrowRight, Camera, ImagePlus, Trash2, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { type FileRejection, useDropzone } from "react-dropzone";

import { Badge } from "@/components/ui/badge";
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

export function ImageUploader({ files, onChange, errorMessage, disabled = false }: ImageUploaderProps) {
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
        notices.push(...fileRejections.map(({ file, errors }) => `${file.name}: ${errors[0]?.message ?? "Dosya kabul edilmedi."}`));
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

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled,
    maxFiles: MAX_PREDICTION_IMAGES,
    maxSize: MAX_PREDICTION_IMAGE_SIZE_BYTES,
    accept: {
      [ALLOWED_PREDICTION_IMAGE_TYPES[0]]: [ALLOWED_PREDICTION_IMAGE_EXTENSIONS[0], ALLOWED_PREDICTION_IMAGE_EXTENSIONS[1]],
      [ALLOWED_PREDICTION_IMAGE_TYPES[1]]: [ALLOWED_PREDICTION_IMAGE_EXTENSIONS[2]],
    },
  });

  const moveItem = (index: number, direction: "left" | "right") => {
    const nextIndex = direction === "left" ? index - 1 : index + 1;
    if (nextIndex < 0 || nextIndex >= files.length) {
      return;
    }

    const updated = [...files];
    const [item] = updated.splice(index, 1);
    updated.splice(nextIndex, 0, item);
    onChange(updated);
  };

  const removeItem = (index: number) => {
    onChange(files.filter((_, fileIndex) => fileIndex !== index));
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[13px] font-medium text-foreground">Fotoğraflar</div>
          <div className="text-xs text-muted-foreground">En fazla 16 fotoğraf analiz edilir.</div>
        </div>
        <Badge variant="secondary">
          {files.length}/{MAX_PREDICTION_IMAGES}
        </Badge>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div
          {...getRootProps()}
          className={cn(
            "flex min-h-[158px] cursor-pointer flex-col justify-center rounded-xl border border-dashed px-4 py-4 transition-colors",
            disabled && "cursor-not-allowed opacity-60",
            isDragActive ? "border-[#0057B8] bg-[#F8FBFF]" : "border-border bg-slate-50/70",
          )}
        >
          <input {...getInputProps()} aria-label="İlan fotoğraflarını yükleyin" />
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-primary">
              {isDragActive ? <UploadCloud className="h-4 w-4" /> : <ImagePlus className="h-4 w-4" />}
            </div>
            <div className="space-y-2">
              <div className="text-sm font-semibold text-foreground">Fotoğraf seçin veya sürükleyin</div>
              <div className="text-sm leading-6 text-muted-foreground">JPG, JPEG veya PNG · maksimum 10 MB / görsel</div>
              <div className="text-xs text-muted-foreground">Salon, mutfak, banyo ve cephe fotoğrafları daha iyi tahmin sağlar.</div>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-slate-50/60 p-3">
          {previews.length > 0 ? (
            <div className="grid max-h-[320px] grid-cols-2 gap-3 overflow-y-auto pr-1">
              {previews.map((preview, index) => (
                <div key={preview.id} className="overflow-hidden rounded-lg border border-border bg-white">
                  <div className="aspect-[4/3] bg-muted">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={preview.previewUrl} alt={preview.name} className="h-full w-full object-cover" />
                  </div>
                  <div className="space-y-2 p-2.5">
                    <div className="min-w-0">
                      <div className="truncate text-xs font-medium text-foreground">{preview.name}</div>
                      <div className="text-[11px] text-muted-foreground">{preview.sizeLabel}</div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button type="button" variant="outline" size="icon" onClick={() => moveItem(index, "left")} disabled={index === 0} className="h-8 w-8">
                        <ArrowLeft className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        onClick={() => moveItem(index, "right")}
                        disabled={index === previews.length - 1}
                        className="h-8 w-8"
                      >
                        <ArrowRight className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => removeItem(index)}
                        className="ml-auto h-8 w-8 text-error hover:bg-error/10 hover:text-error"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex h-full min-h-[158px] items-center gap-3 rounded-lg border border-border bg-white px-4 py-4">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-muted-foreground">
                <Camera className="h-4 w-4" />
              </div>
              <div>
                <div className="text-sm font-medium text-foreground">Henüz fotoğraf eklenmedi</div>
                <div className="text-xs leading-5 text-muted-foreground">Fotoğraf eklemek zorunlu değildir, ancak tahmin kalitesini artırabilir.</div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="text-xs text-muted-foreground">İlk 16 fotoğraf analiz edilir. En temsil edici fotoğrafları öne alın.</div>

      {(localNotices.length > 0 || errorMessage) && (
        <div className="space-y-2">
          {localNotices.map((notice) => (
            <div key={notice} className="flex items-start gap-2 rounded-lg border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
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
