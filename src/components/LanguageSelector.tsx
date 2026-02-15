import { useEffect, useMemo, useState } from "react";

type Language = {
  code: string;
  name: string;
  country: string;
};

const LANGUAGE_SELECTOR_ID = "language-selector";

const LANGUAGES: Language[] = [
  // Major global
  { code: "en", name: "English", country: "us" },
  { code: "es", name: "Español", country: "es" },
  { code: "fr", name: "Français", country: "fr" },
  { code: "de", name: "Deutsch", country: "de" },
  { code: "pt", name: "Português", country: "pt" },
  { code: "pt-BR", name: "Português (Brasil)", country: "br" },
  { code: "ar", name: "العربية", country: "sa" },
  { code: "hi", name: "हिन्दी", country: "in" },
  { code: "bn", name: "বাংলা", country: "bd" },
  { code: "ur", name: "اردو", country: "pk" },
  { code: "sw", name: "Kiswahili", country: "ke" },
  { code: "am", name: "አማርኛ", country: "et" },

  // East / SE Asia
  { code: "zh-Hans", name: "中文（简体）", country: "cn" },
  { code: "zh-Hant", name: "中文（繁體）", country: "tw" },
  { code: "ja", name: "日本語", country: "jp" },
  { code: "ko", name: "한국어", country: "kr" },
  { code: "vi", name: "Tiếng Việt", country: "vn" },
  { code: "th", name: "ไทย", country: "th" },
  { code: "id", name: "Bahasa Indonesia", country: "id" },
  { code: "tl", name: "Filipino", country: "ph" },
  { code: "ms", name: "Bahasa Melayu", country: "my" },

  // Africa (more)
  { code: "ha", name: "Hausa", country: "ng" },
  { code: "yo", name: "Yorùbá", country: "ng" },
  { code: "ig", name: "Igbo", country: "ng" },
  { code: "zu", name: "isiZulu", country: "za" },
  { code: "xh", name: "isiXhosa", country: "za" },
  { code: "rw", name: "Kinyarwanda", country: "rw" },
  { code: "so", name: "Soomaali", country: "so" },

  // Europe / nearby (common)
  { code: "it", name: "Italiano", country: "it" },
  { code: "nl", name: "Nederlands", country: "nl" },
  { code: "pl", name: "Polski", country: "pl" },
  { code: "tr", name: "Türkçe", country: "tr" },
  { code: "ru", name: "Русский", country: "ru" },
  { code: "uk", name: "Українська", country: "ua" },
];

function FlagIcon({ countryCode }: { countryCode: string }) {
  return (
    <span
      className={`fi fis fi-${countryCode} inline-block rounded-full shadow-[inset_0_0_0_2px_rgba(0,0,0,.06)]`}
      style={{ width: 18, height: 18 }}
    />
  );
}

export default function LanguageSelector({
  align = "right",
  }: {
    align?: "right" | "center";
  }) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedCode, setSelectedCode] = useState<string>("en");

  const selectedLanguage = useMemo(
    () => LANGUAGES.find((l) => l.code === selectedCode) ?? LANGUAGES[0],
    [selectedCode]
  );

  const handleLanguageChange = (language: Language) => {
    setSelectedCode(language.code);
    setIsOpen(false);
  };

  useEffect(() => {
    const handleWindowClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      const btn = target?.closest("button");

      if (btn?.id === LANGUAGE_SELECTOR_ID) return;
      setIsOpen(false);
    };

    window.addEventListener("click", handleWindowClick);
    return () => window.removeEventListener("click", handleWindowClick);
  }, []);

  return (
    <div className="relative w-full">
      <button
        id={LANGUAGE_SELECTOR_ID}
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        className="inline-flex items-center justify-between w-full rounded-xl border border-gray-300 shadow-sm px-4 py-3 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50"
      >
        <span className="inline-flex items-center gap-2 min-w-0">
          <FlagIcon countryCode={selectedLanguage.country} />
          <span className="truncate">{selectedLanguage.name}</span>
        </span>

        <span className="text-gray-400">▾</span>
      </button>
      {/* dropdown for language selection */}
      {isOpen && (
        <div
          className={`origin-top-right absolute mt-2 w-max min-w-[340px] rounded-xl shadow-lg bg-white ring-1 ring-black/10
            ${align === "right" ? "right-0" : "left-1/2 -translate-x-1/2"}`}
          role="menu"
        >
          <div className="p-2 grid grid-cols-2 gap-2 max-h-80 overflow-y-auto" role="none">
            {LANGUAGES.map((language, index) => {
              const isSelected = selectedLanguage.code === language.code;
              const sideRound = index % 2 === 0 ? "rounded-l-lg" : "rounded-r-lg";

              return (
                <button
                  key={language.code}
                  onClick={() => handleLanguageChange(language)}
                  className={`${
                    isSelected ? "bg-gray-100 text-gray-900" : "text-gray-700"
                  } px-3 py-2 text-sm text-start items-center inline-flex gap-2 hover:bg-gray-50 active:bg-[#E6CAFF] active:border-[#AF69EE] border border-transparent transition ${sideRound}`}
                  role="menuitem"
                >
                  <FlagIcon countryCode={language.country} />
                  <span className="truncate">{language.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
