import "./global.css";
import { RootProvider } from "fumadocs-ui/provider/next";
import type { ReactNode } from "react";

export const metadata = {
  title: "SO101-Nexus",
  description:
    "Full-stack robot learning for the SO-101 arm: teleoperation, imitation learning, and RL in MuJoCo",
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <RootProvider
          search={{
            options: {
              type: "static",
            },
          }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
