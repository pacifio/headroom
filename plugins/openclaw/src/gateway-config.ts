/* eslint-disable @typescript-eslint/no-explicit-any */

export const DEFAULT_GATEWAY_PROVIDER_IDS = ["openai-codex"] as const;

export function resolveGatewayProviderIds(config: Record<string, unknown> | undefined): string[] {
  if (config?.routeCodexViaProxy === false) {
    return [];
  }

  return [...DEFAULT_GATEWAY_PROVIDER_IDS];
}

export function applyGatewayProviderBaseUrls<T>(
  cfg: T,
  proxyUrl: string,
  providerIds: readonly string[],
): { changed: boolean; config: T } {
  const next = structuredClone((cfg ?? {}) as any);
  const changed = applyGatewayProviderBaseUrlsInPlace(next, proxyUrl, providerIds);
  return { changed, config: next as T };
}

export function applyGatewayProviderBaseUrlsInPlace(
  cfg: any,
  proxyUrl: string,
  providerIds: readonly string[],
): boolean {
  if (!cfg || typeof cfg !== "object" || providerIds.length === 0) {
    return false;
  }

  const models = (cfg.models ??= {});
  const providers = (models.providers ??= {});
  let changed = false;

  for (const providerId of providerIds) {
    const currentValue = providers[providerId];
    const currentConfig =
      currentValue && typeof currentValue === "object" && !Array.isArray(currentValue)
        ? currentValue
        : {};
    const nextConfig = { ...currentConfig };

    if (!Array.isArray(nextConfig.models)) {
      nextConfig.models = [];
      changed = true;
    }

    if (nextConfig.baseUrl === proxyUrl) {
      providers[providerId] = nextConfig;
      continue;
    }

    nextConfig.baseUrl = proxyUrl;
    providers[providerId] = nextConfig;
    changed = true;
  }

  return changed;
}
