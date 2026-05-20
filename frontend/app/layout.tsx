import type { Metadata } from "next";
import type { ReactNode } from "react";

import { AppShell } from "@/components/layout/AppShell";
import { Providers } from "@/components/providers/Providers";

import "./globals.css";

export const metadata: Metadata = {
  title: "Rent Agent | Ankara Kira Tahmini",
  description:
    "Ankara'daki kiralık konutlar için konum, konut özellikleri, ilan açıklaması ve fotoğrafları birlikte değerlendiren kira tahmin arayüzü.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="tr">
      <body>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
