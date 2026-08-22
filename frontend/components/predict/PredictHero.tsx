function HouseScene() {
  return (
    <div className="relative h-[118px] w-[220px]">
      <div className="absolute left-1 top-1 h-3 w-8 rounded-full bg-[#E7EEF8]" />
      <div className="absolute left-9 top-0 h-3.5 w-10 rounded-full bg-[#EEF3FA]" />
      <div className="absolute left-4 top-8 h-12 w-24 rounded-[8px] bg-[#E8F1FF]" />
      <div className="absolute left-3 top-5 h-5 w-[6.7rem] -skew-x-12 rounded-sm bg-[#3F73B8]" />
      <div className="absolute left-[6.2rem] top-[2.55rem] h-10 w-8 rounded-[6px] bg-[#D9E6F8]" />
      <div className="absolute left-[7.3rem] top-[1.9rem] h-3 w-10 -skew-x-12 rounded-sm bg-[#5A87C4]" />
      <div className="absolute left-7 top-[2.65rem] h-5 w-5 rounded-[4px] bg-white shadow-sm" />
      <div className="absolute left-[3.55rem] top-[2.65rem] h-5 w-5 rounded-[4px] bg-white shadow-sm" />
      <div className="absolute left-[3.6rem] top-[3.7rem] h-8 w-3.5 rounded-[4px] bg-[#EDC46C]" />
      <div className="absolute left-[5.2rem] top-[4.7rem] h-[3px] w-[4.5rem] rounded-full bg-[#D8E4F5]" />
      <div className="absolute left-[8.95rem] top-[4.95rem] h-[3px] w-[2.8rem] rounded-full bg-[#D8E4F5]" />
      <div className="absolute left-0 top-[3.2rem] h-14 w-9 rounded-t-full bg-[#DCE7F6]" />
      <div className="absolute left-[0.45rem] top-[4rem] h-8 w-5 rounded-t-full bg-[#4F8A8B]" />
      <div className="absolute left-[1rem] top-[5.2rem] h-6 w-1 rounded-full bg-[#7A6555]" />
      <div className="absolute left-[9.7rem] top-2 h-7 w-7 rounded-full bg-[#FFD764]" />
      <div className="absolute left-0 right-0 top-[5.35rem] h-px bg-[#DCE5F2]" />
    </div>
  );
}

function CityScene() {
  return (
    <div className="relative h-[110px] w-[320px]">
      <div className="absolute inset-x-0 bottom-[1.3rem] h-px bg-[#D7E1EF]" />
      <div className="absolute bottom-[1.3rem] left-10 h-11 w-6 rounded-t-[8px] bg-[#E7EEF8]" />
      <div className="absolute bottom-[1.3rem] left-20 h-14 w-7 rounded-t-[8px] bg-[#D7E6F8]" />
      <div className="absolute bottom-[1.3rem] left-[6.2rem] h-12 w-7 rounded-t-[8px] bg-[#E8EEF7]" />
      <div className="absolute bottom-[1.3rem] left-[8.7rem] h-17 w-8 rounded-t-[8px] bg-[#D5E3F5]" />
      <div className="absolute bottom-[1.3rem] left-[11.8rem] h-9 w-6 rounded-t-[8px] bg-[#E8EEF7]" />
      <div className="absolute bottom-[1.3rem] left-[14rem] h-12 w-7 rounded-t-[8px] bg-[#D9E6F6]" />
      <div className="absolute bottom-[1.3rem] left-[16.4rem] h-16 w-8 rounded-t-[8px] bg-[#D5E3F5]" />
      <div className="absolute bottom-[1.3rem] left-[19.4rem] h-10 w-6 rounded-t-[8px] bg-[#E8EEF7]" />
      <div className="absolute bottom-[1.75rem] left-[12.6rem] h-10 w-5 rounded-t-full bg-[#DCE7F6]" />
      <div className="absolute bottom-[1.75rem] left-[12.95rem] h-7 w-3 rounded-t-full bg-[#6E9CAE]" />
      <div className="absolute bottom-[1.75rem] right-[3.75rem] h-12 w-6 rounded-t-full bg-[#DCE7F6]" />
      <div className="absolute bottom-[1.75rem] right-[4.1rem] h-8 w-3.5 rounded-t-full bg-[#6E9CAE]" />
      <div className="absolute right-[5.6rem] top-[0.3rem] flex h-9 w-9 items-center justify-center rounded-full bg-[#FFD764] text-primary shadow-sm">
        <div className="h-2.5 w-2.5 rounded-full border-2 border-current" />
      </div>
      <div className="absolute left-2 top-3 h-3 w-7 rounded-full bg-[#EEF3FA]" />
      <div className="absolute left-[6.6rem] top-0 h-3.5 w-10 rounded-full bg-[#E7EEF8]" />
      <div className="absolute right-[6.4rem] top-2 h-3 w-8 rounded-full bg-[#EEF3FA]" />
    </div>
  );
}

export function PredictHero() {
  return (
    <section className="relative overflow-hidden rounded-[18px] bg-transparent px-1 py-2">
      <div className="grid min-h-[156px] items-center gap-4 lg:grid-cols-[220px_minmax(0,1fr)_320px]">
        <div className="hidden lg:block">
          <HouseScene />
        </div>

        <div className="flex flex-col items-center justify-center text-center">
          <h1 className="text-[34px] font-semibold tracking-[-0.055em] text-primary sm:text-[48px]">Kira tahmini</h1>
          <p className="mt-3 max-w-2xl text-[15px] leading-7 text-muted-foreground">
            İlan bilgilerini ve fotoğrafları ekleyin, beklenen kira aralığını öğrenin.
          </p>
        </div>

        <div className="hidden justify-end lg:flex">
          <CityScene />
        </div>
      </div>
    </section>
  );
}
