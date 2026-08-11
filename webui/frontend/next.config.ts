import type { NextConfig } from "next";

const backendTarget = (process.env.QUEST_BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const nextConfig: NextConfig = {
  output: "standalone",
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendTarget}/api/:path*`,
      },
      {
        source: "/health",
        destination: `${backendTarget}/health`,
      },
    ];
  },
};

export default nextConfig;
