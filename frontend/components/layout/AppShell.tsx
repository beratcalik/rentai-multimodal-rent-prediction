"use client";

import { Building2, Menu, Sparkles, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { NAV_LINKS } from "@/lib/constants";
import { cn } from "@/lib/utils";

type AppShellProps = {
  children: ReactNode;
};

const SHELL_WIDTH = "mx-auto w-[calc(100%_-_32px)] max-w-[1500px] sm:w-[calc(100%_-_48px)] lg:w-[calc(100%_-_96px)]";

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  return (
    <div className="flex min-h-screen flex-col overflow-x-clip bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b border-[#DDE7F0] bg-white">
        <div className={cn(SHELL_WIDTH, "flex h-16 items-center justify-between gap-4")}>
          <Link href="/" className="flex min-w-0 items-center gap-3" aria-label="RentAI ana sayfa">
            <div className="relative flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary text-white shadow-[0_8px_18px_rgba(11,58,117,0.18)]">
              <Building2 className="h-5 w-5" aria-hidden="true" />
              <span className="absolute -right-1 -top-1 h-3.5 w-3.5 rounded-full bg-[#FFD200]" />
            </div>
            <div className="min-w-0 leading-tight">
              <div className="truncate text-[15px] font-semibold text-foreground">RentAI</div>
              <div className="truncate text-xs text-muted-foreground">Kira Asistanı</div>
            </div>
          </Link>

          <nav className="hidden items-center gap-9 md:flex" aria-label="Ana navigasyon">
            {NAV_LINKS.map((item) => {
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "relative pb-1 text-[15px] transition-colors",
                    isActive ? "font-semibold text-primary" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.label}
                  {isActive ? <span className="absolute inset-x-0 -bottom-[18px] h-0.5 rounded-full bg-primary" /> : null}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-2">
            <Button asChild variant="secondary" className="hidden min-w-[164px] md:inline-flex">
              <Link href="/predict">
                <Sparkles className="h-4 w-4" />
                Kira Tahmini Yap
              </Link>
            </Button>

            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="md:hidden"
              aria-label={mobileMenuOpen ? "Menüyü kapat" : "Menüyü aç"}
              aria-expanded={mobileMenuOpen}
              aria-controls="mobile-site-menu"
              onClick={() => setMobileMenuOpen((current) => !current)}
            >
              {mobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {mobileMenuOpen ? (
          <div id="mobile-site-menu" className="border-t border-border bg-white md:hidden">
            <div className={cn(SHELL_WIDTH, "space-y-2 py-3")}>
              {NAV_LINKS.map((item) => {
                const isActive = pathname === item.href;

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center justify-between rounded-xl border px-3 py-2.5 text-sm transition-colors",
                      isActive ? "border-[#FFD200] bg-[#FFF9D6] font-semibold text-primary" : "border-border text-foreground",
                    )}
                  >
                    <span>{item.label}</span>
                    {isActive ? <span className="text-[11px] text-muted-foreground">Aktif</span> : null}
                  </Link>
                );
              })}

              <Button asChild variant="secondary" className="mt-2 w-full">
                <Link href="/predict">
                  <Sparkles className="h-4 w-4" />
                  Kira Tahmini Yap
                </Link>
              </Button>
            </div>
          </div>
        ) : null}
      </header>

      <main className="flex-1 overflow-x-clip">{children}</main>

      <footer className="border-t border-border bg-white">
        <div className={cn(SHELL_WIDTH, "flex flex-col gap-2 py-4 text-sm text-muted-foreground md:flex-row md:items-center md:justify-between")}>
          <div>RentAI</div>
          <div>Karar destek amaçlıdır.</div>
        </div>
      </footer>
    </div>
  );
}
