"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Building2 } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { NAV_LINKS } from "@/lib/constants";
import { cn } from "@/lib/utils";

type AppShellProps = {
  children: ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const showPrimaryCta = pathname !== "/predict";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b border-border bg-white">
        <div className="container flex h-16 items-center justify-between gap-4">
          <Link href="/" className="flex items-center gap-3" aria-label="Rent Agent ana sayfa">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-white">
              <Building2 className="h-5 w-5" aria-hidden="true" />
              <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-[#FFD200]" />
            </div>
            <div className="leading-tight">
              <div className="text-sm font-semibold text-foreground">Rent Agent</div>
              <div className="text-xs text-muted-foreground">Kira Asistanı</div>
            </div>
          </Link>

          <nav className="hidden items-center gap-6 md:flex" aria-label="Ana navigasyon">
            {NAV_LINKS.map((item) => {
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "text-sm transition-colors",
                    isActive ? "font-semibold text-primary" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          {showPrimaryCta ? (
            <Button asChild variant="secondary" className="hidden md:inline-flex">
              <Link href="/predict">Kira Tahmini Yap</Link>
            </Button>
          ) : (
            <div className="hidden text-sm text-muted-foreground md:block">Tahmin formu</div>
          )}
        </div>

        <div className="container pb-3 md:hidden">
          <nav className="flex gap-4 overflow-x-auto whitespace-nowrap text-sm" aria-label="Mobil navigasyon">
            {NAV_LINKS.map((item) => {
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "border-b-2 pb-2 pt-1 transition-colors",
                    isActive ? "border-[#FFD200] font-semibold text-primary" : "border-transparent text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      <main>{children}</main>

      <footer className="border-t border-border bg-white">
        <div className="container flex flex-col gap-2 py-6 text-sm text-muted-foreground md:flex-row md:items-center md:justify-between">
          <div>Rent Agent · Ankara için kira değerleme arayüzü</div>
          <div>Karar destek amaçlıdır.</div>
        </div>
      </footer>
    </div>
  );
}
