import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, shareReplay, switchMap, map } from 'rxjs';

export interface SnippetMeta {
  source: string;
  files: string[];
}

export interface CoSiMManifest {
  similar:    Record<string, [string, string]>;
  notsimilar: Record<string, [string, string]>;
  snippets:   Record<string, SnippetMeta>;
}

export interface PairUUIDs {
  uuid1: string;
  uuid2: string;
}

@Injectable({ providedIn: 'root' })
export class CosimService {
  private manifest$: Observable<CoSiMManifest>;

  constructor(private http: HttpClient) {
    this.manifest$ = this.http
      .get<CoSiMManifest>('assets/cosim-manifest.json')
      .pipe(shareReplay(1));
  }

  getManifest(): Observable<CoSiMManifest> {
    return this.manifest$;
  }

  getPairUUIDs(label: 'similar' | 'notsimilar', index: number): Observable<PairUUIDs> {
    return this.manifest$.pipe(
      map(m => {
        const entry = label === 'similar'
          ? m.similar[String(index)]
          : m.notsimilar[String(index)];
        return { uuid1: entry[0], uuid2: entry[1] };
      }),
    );
  }

  getSnippetMeta(uuid: string): Observable<SnippetMeta | null> {
    return this.manifest$.pipe(map(m => m.snippets[uuid] ?? null));
  }

  loadFile(uuid: string, filePath: string): Observable<string> {
    return this.http.get(`cosim/raw/${uuid}/${filePath}`, { responseType: 'text' });
  }
}
