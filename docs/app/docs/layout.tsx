import { source } from "@/lib/source";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import type { ReactNode } from "react";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout
      tree={source.pageTree}
      nav={{ title: "SO101-Nexus" }}
      githubUrl="https://github.com/johnsutor/so101-nexus"
    >
      {children}
    </DocsLayout>
  );
}
