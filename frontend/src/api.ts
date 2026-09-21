export interface Wine {
  slug: string;
  name: string;
  category: string;
  color: string;
  region: string;
  grapes: string;
  description: string;
  winery: string;
  image_url: string;
  source_url: string;
  rosquality_rating: number | null;
}
export interface ScanResult {
  id: string;
  status: "matched" | "uncertain" | "not_found" | "demo";
  wine: Wine | null;
  candidates: { wine: Wine; similarity: number }[];
  similarity: number | null;
  margin: number | null;
  elapsed_ms: number;
  model_version: string;
  provider: string;
  created_at: string;
  message: string;
}
export interface Health {
  provider: string;
  model_ready: boolean;
  model_status: string;
  message: string | null;
  catalog_count: number;
  max_upload_mb: number;
}
export interface Meta {
  total: number;
  categories: string[];
  winery_count: number;
  featured: Wine[];
}
export interface CatalogPage {
  items: Wine[];
  total: number;
  offset: number;
  limit: number;
}
export interface Pairing {
  title: string;
  explanation: string;
  temperature: string;
  note: string;
  wines: Wine[];
}

export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      credentials: "same-origin",
      ...options,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error(
      "Не удалось загрузить данные. Проверьте соединение и попробуйте ещё раз.",
    );
  }
  if (!response.ok) {
    if (response.status >= 500) {
      throw new Error(
        path === "/api/scan"
          ? "Распознавание временно недоступно. Попробуйте позже или найдите вино в каталоге."
          : "Сервис временно недоступен. Попробуйте ещё раз немного позже.",
      );
    }
    let message = "Не удалось связаться с сервером. Попробуйте ещё раз.";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* Keep a readable fallback for non-JSON proxy errors. */
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

export function useSavedInitial(): string[] {
  try {
    const value: unknown = JSON.parse(
      localStorage.getItem("viscaner-saved") || "[]",
    );
    return Array.isArray(value)
      ? value.filter((x): x is string => typeof x === "string").slice(0, 500)
      : [];
  } catch {
    return [];
  }
}
