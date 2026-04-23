export interface SimScore { g2: number; g3: number; g4: number; }
export interface SimRow {
  engine: string;
  group: 'mysql' | 'postgresql' | 'sqlserver';
  scores: (SimScore | null)[];
}
export interface CorpusEntry {
  database: string;
  version: string;
  releaseDate: string;
  dockerKb: number | null;
  cloudKb: number | null;
  group: 'mysql' | 'postgresql' | 'sqlserver';
}

function s(g2: number, g3: number, g4: number): SimScore { return { g2, g3, g4 }; }

export const CORPUS: CorpusEntry[] = [
  { database: 'MySQL',          version: '5.6',        releaseDate: '2013/02', dockerKb: 643.5,  cloudKb: 388.1,  group: 'mysql' },
  { database: 'MySQL',          version: '5.7',        releaseDate: '2015/10', dockerKb: 637.9,  cloudKb: 392.4,  group: 'mysql' },
  { database: 'MySQL',          version: '8.0',        releaseDate: '2018/04', dockerKb: 662.5,  cloudKb: 404.1,  group: 'mysql' },
  { database: 'PostgreSQL',     version: '9.6',        releaseDate: '2016/09', dockerKb: 727.0,  cloudKb: 364.3,  group: 'postgresql' },
  { database: 'PostgreSQL',     version: '10',         releaseDate: '2017/10', dockerKb: 762.4,  cloudKb: 363.9,  group: 'postgresql' },
  { database: 'PostgreSQL',     version: '11',         releaseDate: '2018/10', dockerKb: 762.3,  cloudKb: 364.2,  group: 'postgresql' },
  { database: 'PostgreSQL',     version: '12',         releaseDate: '2019/10', dockerKb: 763.5,  cloudKb: 363.7,  group: 'postgresql' },
  { database: 'PostgreSQL',     version: '13',         releaseDate: '2020/09', dockerKb: 763.5,  cloudKb: 364.0,  group: 'postgresql' },
  { database: 'PostgreSQL',     version: '14',         releaseDate: '2021/09', dockerKb: 1173.3, cloudKb: 610.6,  group: 'postgresql' },
  { database: 'PostgreSQL',     version: '15',         releaseDate: '2022/10', dockerKb: 1071.9, cloudKb: 582.3,  group: 'postgresql' },
  { database: 'PostgreSQL',     version: '16',         releaseDate: '2023/09', dockerKb: 1202.9, cloudKb: 596.6,  group: 'postgresql' },
  { database: 'SQL Server 2017', version: 'Developer', releaseDate: '2017/10', dockerKb: 1125.9, cloudKb: null,   group: 'sqlserver' },
  { database: 'SQL Server 2017', version: 'Standard',  releaseDate: '2017/10', dockerKb: null,   cloudKb: 609.9,  group: 'sqlserver' },
  { database: 'SQL Server 2017', version: 'Enterprise',releaseDate: '2017/10', dockerKb: null,   cloudKb: 609.9,  group: 'sqlserver' },
  { database: 'SQL Server 2019', version: 'Developer', releaseDate: '2019/11', dockerKb: 1125.5, cloudKb: null,   group: 'sqlserver' },
  { database: 'SQL Server 2019', version: 'Standard',  releaseDate: '2019/11', dockerKb: null,   cloudKb: 608.8,  group: 'sqlserver' },
  { database: 'SQL Server 2019', version: 'Enterprise',releaseDate: '2019/11', dockerKb: null,   cloudKb: 613.4,  group: 'sqlserver' },
  { database: 'SQL Server 2022', version: 'Developer', releaseDate: '2022/11', dockerKb: 1125.5, cloudKb: null,   group: 'sqlserver' },
  { database: 'SQL Server 2022', version: 'Standard',  releaseDate: '2022/11', dockerKb: null,   cloudKb: 610.1,  group: 'sqlserver' },
  { database: 'SQL Server 2022', version: 'Enterprise',releaseDate: '2022/11', dockerKb: null,   cloudKb: 609.9,  group: 'sqlserver' },
];

// Columns: MySQL 5.6 Docker | MySQL 5.6 Cloud SQL | MySQL 5.7 Docker | MySQL 5.7 Cloud SQL | MySQL 8.0 Docker | MySQL 8.0 Cloud SQL
export const SIM_COLUMNS = [
  'MySQL 5.6\n(Docker)', 'MySQL 5.6\n(Cloud SQL)',
  'MySQL 5.7\n(Docker)', 'MySQL 5.7\n(Cloud SQL)',
  'MySQL 8.0\n(Docker)', 'MySQL 8.0\n(Cloud SQL)',
];

export const SIM_ROWS: SimRow[] = [
  { engine: 'MySQL 5.6',                             group: 'mysql',      scores: [null,             s(.78,.68,.64), s(.81,.73,.69), s(.77,.67,.64), s(.81,.72,.67), s(.75,.64,.61)] },
  { engine: 'MySQL 5.6 (Cloud SQL)',                  group: 'mysql',      scores: [s(.78,.68,.64),   null,           s(.77,.68,.64), s(.79,.70,.67), s(.77,.67,.63), s(.76,.66,.62)] },
  { engine: 'MySQL 5.7',                             group: 'mysql',      scores: [s(.81,.73,.69),   s(.77,.68,.64), null,           s(.78,.69,.66), s(.81,.73,.68), s(.75,.65,.61)] },
  { engine: 'MySQL 5.7 (Cloud SQL)',                  group: 'mysql',      scores: [s(.77,.67,.64),   s(.79,.70,.67), s(.78,.69,.66), null,           s(.76,.67,.63), s(.76,.65,.62)] },
  { engine: 'MySQL 8.0',                             group: 'mysql',      scores: [s(.81,.72,.67),   s(.77,.67,.63), s(.81,.73,.68), s(.76,.67,.63), null,           s(.76,.66,.63)] },
  { engine: 'MySQL 8.0 (Cloud SQL)',                  group: 'mysql',      scores: [s(.75,.64,.61),   s(.76,.66,.62), s(.75,.65,.61), s(.76,.65,.62), s(.76,.66,.63), null          ] },
  { engine: 'PostgreSQL 9.6',                        group: 'postgresql', scores: [s(.47,.25,.06),   s(.45,.23,.06), s(.47,.25,.06), s(.44,.23,.06), s(.46,.25,.06), s(.42,.23,.06)] },
  { engine: 'PostgreSQL 9.6 (Cloud SQL)',             group: 'postgresql', scores: [s(.44,.22,.05),   s(.42,.20,.05), s(.44,.22,.05), s(.41,.20,.05), s(.43,.22,.05), s(.40,.20,.05)] },
  { engine: 'PostgreSQL 10',                         group: 'postgresql', scores: [s(.47,.24,.06),   s(.45,.23,.06), s(.47,.24,.06), s(.44,.23,.06), s(.47,.25,.06), s(.43,.23,.06)] },
  { engine: 'PostgreSQL 10 (Cloud SQL)',              group: 'postgresql', scores: [s(.44,.22,.05),   s(.41,.20,.05), s(.43,.21,.05), s(.41,.20,.05), s(.43,.22,.05), s(.40,.20,.05)] },
  { engine: 'PostgreSQL 11',                         group: 'postgresql', scores: [s(.47,.24,.06),   s(.44,.23,.06), s(.46,.24,.06), s(.43,.23,.06), s(.46,.24,.06), s(.42,.23,.06)] },
  { engine: 'PostgreSQL 11 (Cloud SQL)',              group: 'postgresql', scores: [s(.44,.22,.05),   s(.41,.20,.05), s(.43,.21,.05), s(.41,.20,.05), s(.43,.21,.05), s(.40,.20,.05)] },
  { engine: 'PostgreSQL 12',                         group: 'postgresql', scores: [s(.47,.24,.06),   s(.45,.23,.06), s(.47,.24,.06), s(.44,.23,.06), s(.47,.24,.06), s(.43,.23,.06)] },
  { engine: 'PostgreSQL 12 (Cloud SQL)',              group: 'postgresql', scores: [s(.44,.22,.05),   s(.41,.20,.05), s(.43,.21,.05), s(.41,.20,.05), s(.43,.21,.05), s(.40,.20,.05)] },
  { engine: 'PostgreSQL 13',                         group: 'postgresql', scores: [s(.47,.25,.06),   s(.45,.23,.06), s(.47,.24,.06), s(.44,.23,.06), s(.47,.24,.06), s(.43,.23,.06)] },
  { engine: 'PostgreSQL 13 (Cloud SQL)',              group: 'postgresql', scores: [s(.44,.22,.05),   s(.41,.20,.05), s(.43,.22,.05), s(.41,.20,.05), s(.43,.22,.05), s(.40,.20,.05)] },
  { engine: 'PostgreSQL 14',                         group: 'postgresql', scores: [s(.41,.19,.05),   s(.39,.18,.04), s(.40,.19,.05), s(.38,.18,.04), s(.41,.19,.05), s(.37,.18,.04)] },
  { engine: 'PostgreSQL 14 (Cloud SQL)',              group: 'postgresql', scores: [s(.38,.16,.04),   s(.36,.15,.03), s(.38,.16,.04), s(.35,.15,.03), s(.38,.16,.04), s(.34,.15,.03)] },
  { engine: 'PostgreSQL 15',                         group: 'postgresql', scores: [s(.40,.19,.05),   s(.38,.18,.04), s(.40,.19,.05), s(.38,.18,.04), s(.40,.19,.05), s(.37,.18,.04)] },
  { engine: 'PostgreSQL 15 (Cloud SQL)',              group: 'postgresql', scores: [s(.38,.17,.04),   s(.35,.16,.03), s(.37,.17,.04), s(.35,.16,.03), s(.37,.17,.04), s(.34,.16,.03)] },
  { engine: 'PostgreSQL 16',                         group: 'postgresql', scores: [s(.43,.19,.05),   s(.40,.18,.04), s(.42,.20,.05), s(.40,.18,.04), s(.42,.19,.05), s(.39,.19,.04)] },
  { engine: 'PostgreSQL 16 (Cloud SQL)',              group: 'postgresql', scores: [s(.44,.22,.05),   s(.41,.20,.05), s(.43,.21,.05), s(.41,.20,.05), s(.43,.22,.05), s(.40,.20,.05)] },
  { engine: 'SQL Server 2017 Developer',             group: 'sqlserver',  scores: [s(.41,.30,.26),   s(.36,.29,.26), s(.40,.31,.27), s(.36,.28,.26), s(.39,.30,.26), s(.33,.27,.27)] },
  { engine: 'SQL Server 2017 Enterprise (Cloud SQL)', group: 'sqlserver',  scores: [s(.42,.31,.25),   s(.37,.29,.25), s(.41,.31,.26), s(.37,.29,.25), s(.41,.31,.25), s(.35,.28,.26)] },
  { engine: 'SQL Server 2017 Standard (Cloud SQL)',   group: 'sqlserver',  scores: [s(.42,.31,.25),   s(.37,.29,.25), s(.41,.31,.25), s(.37,.29,.25), s(.41,.30,.25), s(.35,.28,.26)] },
  { engine: 'SQL Server 2019 Developer',             group: 'sqlserver',  scores: [s(.41,.30,.26),   s(.36,.29,.26), s(.40,.31,.27), s(.36,.28,.26), s(.39,.30,.26), s(.33,.27,.27)] },
  { engine: 'SQL Server 2019 Enterprise (Cloud SQL)', group: 'sqlserver',  scores: [s(.39,.28,.22),   s(.35,.26,.22), s(.38,.28,.22), s(.34,.26,.22), s(.38,.27,.22), s(.32,.25,.23)] },
  { engine: 'SQL Server 2019 Standard (Cloud SQL)',   group: 'sqlserver',  scores: [s(.42,.30,.24),   s(.37,.29,.24), s(.41,.31,.25), s(.37,.28,.24), s(.40,.30,.25), s(.34,.27,.26)] },
  { engine: 'SQL Server 2022 Developer',             group: 'sqlserver',  scores: [s(.41,.30,.26),   s(.36,.29,.26), s(.40,.31,.27), s(.36,.28,.26), s(.39,.30,.26), s(.33,.27,.27)] },
  { engine: 'SQL Server 2022 Enterprise (Cloud SQL)', group: 'sqlserver',  scores: [s(.41,.31,.25),   s(.37,.29,.25), s(.41,.31,.26), s(.37,.29,.25), s(.41,.30,.25), s(.34,.28,.26)] },
  { engine: 'SQL Server 2022 Standard (Cloud SQL)',   group: 'sqlserver',  scores: [s(.39,.29,.24),   s(.35,.27,.24), s(.38,.29,.24), s(.35,.27,.24), s(.38,.29,.24), s(.33,.27,.25)] },
];

export const SVM_MISCLASSIFICATIONS = [
  { actual: 'MySQL 5.6 (Cloud SQL)',              predicted: 'MySQL 5.7 (Cloud SQL)' },
  { actual: 'MySQL 8.0 (Docker)',                 predicted: 'MySQL 5.7 (Docker)' },
  { actual: 'PostgreSQL 14 (Cloud SQL)',          predicted: 'PostgreSQL 15 (Cloud SQL)' },
  { actual: 'PostgreSQL 16 (Cloud SQL)',          predicted: 'PostgreSQL 9.6 (Cloud SQL)' },
  { actual: 'SQL Server 2019 Enterprise (Cloud SQL)', predicted: 'SQL Server 2019 Standard (Cloud SQL)' },
];
