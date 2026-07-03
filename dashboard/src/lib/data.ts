import type {
  Calibration,
  Governance,
  Importances,
  ModelKey,
  ModelsMeta,
  RepoLinks,
  Rp2Summary,
  SweepPoint,
} from "./types";

const cache = new Map<string, Promise<unknown>>();

function getJson<T>(path: string): Promise<T> {
  let entry = cache.get(path) as Promise<T> | undefined;
  if (!entry) {
    entry = fetch(`${import.meta.env.BASE_URL}data/${path}`).then((res) => {
      if (!res.ok) {
        throw new Error(`Failed to load ${path}: ${res.status}`);
      }
      return res.json() as Promise<T>;
    });
    cache.set(path, entry);
  }
  return entry;
}

export const loadModelsMeta = (): Promise<ModelsMeta> => getJson<ModelsMeta>("models.json");
export const loadImportances = (): Promise<Importances> => getJson<Importances>("importances.json");
export const loadCalibration = (): Promise<Calibration> => getJson<Calibration>("calibration.json");
export const loadGovernance = (): Promise<Governance> => getJson<Governance>("governance.json");
export const loadRp2Summary = (): Promise<Rp2Summary> => getJson<Rp2Summary>("rp2_summary.json");
export const loadRepoLinks = (): Promise<RepoLinks> => getJson<RepoLinks>("repo_links.json");
export const loadSweep = (model: ModelKey): Promise<SweepPoint[]> =>
  getJson<SweepPoint[]>(`sweep_${model}.json`);
