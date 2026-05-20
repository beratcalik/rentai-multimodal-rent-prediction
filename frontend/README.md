# Rent Agent Frontend Foundation

## Secilen Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- shadcn/ui uyumlu component yapisi
- Radix UI
- React Hook Form
- Zod
- TanStack Query
- Recharts
- react-dropzone
- browser-image-compression
- lucide-react

## Tasarim Kararlari

- Dil: modern, premium, guven veren SaaS
- Renk paleti:
  - background: `#F8FAFC`
  - surface: `#FFFFFF`
  - primary: `#1D4ED8`
  - secondary: `#0F766E`
  - text: `#111827`
  - muted text: `#6B7280`
  - border: `#E5E7EB`
  - success: `#15803D`
  - warning: `#B45309`
  - error: `#B91C1C`
- Border radius: agirlikli olarak `16px` ve `24px`
- Kartlar: soft shadow, genis spacing, acik arka planlar
- Erisilebilirlik: gorunur label, focus ring, klavye odak destegi, rengin tek basina anlam tasimadigi durum etiketleri

## Route Yapisi

- `/`
- `/predict`
- `/methodology`

## Kurulum

```bash
cd frontend
npm install
npm run dev
npm run build
```

## Calisma Notlari

- `App Router` kullanilir.
- `shadcn/ui` icin `components.json`, `lib/utils.ts` ve `components/ui` tabani hazirdir.
- `Predict` sayfasi backend olmadan da UX akisini gosteren profesyonel placeholder arayuzu sunar.
- Sonraki adimda `/predict` sayfasi Python inference API'sine baglanabilir.
