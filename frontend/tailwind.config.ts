import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "1rem",
      screens: {
        "2xl": "1240px",
      },
    },
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface: "var(--surface)",
        primary: {
          DEFAULT: "var(--primary)",
          foreground: "var(--primary-foreground)",
        },
        secondary: {
          DEFAULT: "var(--secondary)",
          foreground: "var(--secondary-foreground)",
        },
        muted: {
          DEFAULT: "var(--muted)",
          foreground: "var(--muted-foreground)",
        },
        border: "var(--border)",
        success: "var(--success)",
        warning: "var(--warning)",
        error: "var(--error)",
        ring: "var(--ring)",
        card: {
          DEFAULT: "var(--surface)",
          foreground: "var(--foreground)",
        },
      },
      borderRadius: {
        lg: "0.625rem",
        xl: "0.75rem",
        "2xl": "0.875rem",
        "3xl": "1rem",
      },
      boxShadow: {
        soft: "0 6px 18px rgba(15, 23, 42, 0.06)",
        card: "0 2px 8px rgba(15, 23, 42, 0.04)",
        glow: "0 10px 24px rgba(11, 58, 117, 0.12)",
      },
      backgroundImage: {
        "page-gradient": "linear-gradient(180deg, rgba(245,246,248,1) 0%, rgba(245,246,248,1) 100%)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
