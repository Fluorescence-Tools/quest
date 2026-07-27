import "./globals.css";
import type { ReactNode } from "react";
import Script from "next/script";
import { AppProvider } from "../context/AppContext";
import { AppLayout } from "../components/AppLayout";

export const metadata = {
  title: "Quenching Estimator",
  description: "QuEst protein quenching simulation workbench",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppProvider>
          <AppLayout>{children}</AppLayout>
        </AppProvider>
        <Script
          src="/vendor/ngl.js"
          strategy="beforeInteractive"
        />
      </body>
    </html>
  );
}
