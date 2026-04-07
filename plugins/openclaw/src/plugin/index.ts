/**
 * Headroom OpenClaw Plugin — register ContextEngine + CCR retrieval tool.
 *
 * Usage:
 *   openclaw plugins install headroom-ai/openclaw
 *
 * Configuration (in ~/.openclaw/config.json or ~/.clawdbot/clawdbot.json):
 *   {
 *     "plugins": {
 *       "slots": { "contextEngine": "headroom" },
 *       "entries": { "headroom": { "enabled": true } }
 *     }
 *   }
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { HeadroomContextEngine } from "../engine.js";
import {
  applyGatewayProviderBaseUrls,
  applyGatewayProviderBaseUrlsInPlace,
  resolveGatewayProviderIds,
} from "../gateway-config.js";
import { normalizeAndValidateProxyUrl } from "../proxy-manager.js";
import { createHeadroomRetrieveTool } from "../tools/headroom-retrieve.js";

export default function headroomPlugin(api: any) {
  const config = api.config?.plugins?.entries?.headroom?.config ?? {};
  const logger = api.logger ?? console;
  const rawProxyUrl = config.proxyUrl;
  const proxyUrl =
    typeof rawProxyUrl === "string" && rawProxyUrl.trim().length > 0
      ? normalizeAndValidateProxyUrl(rawProxyUrl)
      : undefined;

  const engine = new HeadroomContextEngine({ ...config, proxyUrl }, {
    info: (m: string) => logger.info(m),
    warn: (m: string) => logger.warn(m),
    error: (m: string) => logger.error(m),
    debug: (m: string) => logger.debug?.(m),
  });
  const gatewayProviderIds = resolveGatewayProviderIds(config);

  const ensureGatewayRouting = async () => {
    if (gatewayProviderIds.length === 0) {
      return;
    }

    try {
      const activeProxyUrl = await engine.ensureProxyUrl();

      applyGatewayProviderBaseUrlsInPlace(api.config, activeProxyUrl, gatewayProviderIds);

      const currentConfig = api.runtime?.config?.loadConfig?.();
      const writeConfigFile = api.runtime?.config?.writeConfigFile;
      if (!currentConfig || typeof writeConfigFile !== "function") {
        logger.info(
          `[headroom] Upstream gateway routing active in memory for ${gatewayProviderIds.join(", ")} via ${activeProxyUrl}`,
        );
        return;
      }

      const { changed, config: nextConfig } = applyGatewayProviderBaseUrls(
        currentConfig,
        activeProxyUrl,
        gatewayProviderIds,
      );

      if (changed) {
        await writeConfigFile(nextConfig);
        logger.info(
          `[headroom] Routed ${gatewayProviderIds.join(", ")} through Headroom proxy at ${activeProxyUrl}`,
        );
      } else {
        logger.info(
          `[headroom] Upstream gateway already routed for ${gatewayProviderIds.join(", ")} at ${activeProxyUrl}`,
        );
      }
    } catch (error) {
      logger.warn(`[headroom] Failed to configure upstream gateway routing: ${error}`);
    }
  };

  // Register as context engine
  api.registerContextEngine("headroom", () => engine);

  // Register CCR retrieval tool (active once proxy is running)
  api.registerTool((ctx: any) => {
    const activeProxyUrl = engine.getProxyUrl() ?? proxyUrl;
    if (!activeProxyUrl) return null;
    return createHeadroomRetrieveTool({ proxyUrl: activeProxyUrl });
  });

  api.on("gateway_start", async () => {
    await ensureGatewayRouting();
  });

  void ensureGatewayRouting();

  logger.info("[headroom] Plugin registered");
}
