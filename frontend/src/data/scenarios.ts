/**
 * Narrative Scenarios - Pre-built missions with briefings and scoring
 *
 * Each scenario maps to existing RunConfig options.
 * Difficulty scaling through entity count, evasion type, and par time.
 */

export interface NarrativeScenario {
  id: string;
  name: string;
  codename: string;
  difficulty: 'EASY' | 'MEDIUM' | 'HARD' | 'EXTREME';
  briefing: {
    situation: string;
    objective: string;
    constraints: string[];
    threatPicture: string;
  };
  config: {
    scenario: string;
    guidance: string;
    evasion: string;
    navConstant: number;
    numInterceptors: number;
    numTargets: number;
    enableCooperative?: boolean;
    enableSwarm?: boolean;
    enableHMT?: boolean;
    windSpeed?: number;
    windDirection?: number;
    enableDrag?: boolean;
  };
  scoring: {
    parTime: number;
    timeBonus: number;
    efficiencyWeight: number;
    accuracyWeight: number;
  };
}

export const NARRATIVE_SCENARIOS: NarrativeScenario[] = [
  {
    id: 'first-contact',
    name: 'First Contact',
    codename: 'SILENT DAWN',
    difficulty: 'EASY',
    briefing: {
      situation: 'A single unidentified aircraft has violated restricted airspace on a direct heading toward the protected zone.',
      objective: 'Intercept and neutralize the target using proportional navigation guidance.',
      constraints: ['Single interceptor', 'No evasion expected', 'Clear weather'],
      threatPicture: 'One bogey, head-on approach, constant velocity, no countermeasures.',
    },
    config: {
      scenario: 'head_on',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 1,
      numTargets: 1,
    },
    scoring: {
      parTime: 20,
      timeBonus: 10,
      efficiencyWeight: 0.3,
      accuracyWeight: 0.7,
    },
  },
  {
    id: 'evasive-action',
    name: 'Evasive Action',
    codename: 'VIPER STRIKE',
    difficulty: 'MEDIUM',
    briefing: {
      situation: 'Intelligence indicates the target is equipped with a missile warning receiver and will begin evasive maneuvers upon detection.',
      objective: 'Intercept a maneuvering target using augmented proportional navigation.',
      constraints: ['Target will execute weave maneuvers', 'Expect 3-4G turns', 'Use APN guidance'],
      threatPicture: 'One bogey, S-turn evasion pattern. Moderate threat level.',
    },
    config: {
      scenario: 'head_on',
      guidance: 'augmented_pn',
      evasion: 'weave',
      navConstant: 4,
      numInterceptors: 1,
      numTargets: 1,
      enableDrag: true,
    },
    scoring: {
      parTime: 25,
      timeBonus: 8,
      efficiencyWeight: 0.4,
      accuracyWeight: 0.6,
    },
  },
  {
    id: 'saturation-attack',
    name: 'Saturation Attack',
    codename: 'IRON STORM',
    difficulty: 'HARD',
    briefing: {
      situation: 'Multiple hostile aircraft inbound in a coordinated saturation attack. Your interceptor force must engage all threats before they reach the defended zone.',
      objective: 'Neutralize all 4 targets using optimal weapon-target assignment.',
      constraints: ['3 interceptors vs 4 targets', 'Targets executing barrel roll evasion', 'Hungarian algorithm recommended'],
      threatPicture: 'Four bogeys, staggered approach, aggressive 3D evasion. High threat level.',
    },
    config: {
      scenario: 'head_on',
      guidance: 'augmented_pn',
      evasion: 'barrel_roll',
      navConstant: 4.5,
      numInterceptors: 3,
      numTargets: 4,
      enableDrag: true,
      windSpeed: 10,
      windDirection: 270,
    },
    scoring: {
      parTime: 35,
      timeBonus: 6,
      efficiencyWeight: 0.5,
      accuracyWeight: 0.5,
    },
  },
  {
    id: 'cooperative-defense',
    name: 'Cooperative Defense',
    codename: 'SHIELD WALL',
    difficulty: 'HARD',
    briefing: {
      situation: 'Hostile forces are probing defenses across multiple sectors. Coordinate interceptors across engagement zones with handoff capability.',
      objective: 'Defend all sectors using cooperative engagement with zone-based defense.',
      constraints: ['Enable cooperative engagement', 'Multiple engagement zones active', 'Targets use random jinking'],
      threatPicture: 'Three bogeys across wide frontage. Requires coordinated multi-sector response.',
    },
    config: {
      scenario: 'head_on',
      guidance: 'proportional_nav',
      evasion: 'random_jink',
      navConstant: 4,
      numInterceptors: 4,
      numTargets: 3,
      enableCooperative: true,
      enableDrag: true,
    },
    scoring: {
      parTime: 40,
      timeBonus: 5,
      efficiencyWeight: 0.6,
      accuracyWeight: 0.4,
    },
  },
  {
    id: 'swarm-assault',
    name: 'Swarm Assault',
    codename: 'DARK HORIZON',
    difficulty: 'EXTREME',
    briefing: {
      situation: 'A massive coordinated swarm attack is inbound. Hostile drones in formation are executing aggressive random jinking. Deploy swarm interceptors and leverage human-machine teaming to manage the engagement.',
      objective: 'Intercept the incoming swarm using formation tactics and autonomous engagement.',
      constraints: ['Enable swarm mode', 'Human-on-loop authority', '8 targets, 6 interceptors', 'Extreme evasion'],
      threatPicture: 'Eight bogeys in echelon formation, random jink evasion, high closing velocity. Maximum threat.',
    },
    config: {
      scenario: 'head_on',
      guidance: 'augmented_pn',
      evasion: 'random_jink',
      navConstant: 5,
      numInterceptors: 6,
      numTargets: 8,
      enableSwarm: true,
      enableHMT: true,
      enableDrag: true,
      windSpeed: 15,
      windDirection: 180,
    },
    scoring: {
      parTime: 50,
      timeBonus: 4,
      efficiencyWeight: 0.5,
      accuracyWeight: 0.5,
    },
  },
  {
    id: 'iron-shield',
    name: 'Iron Shield',
    codename: 'IRON SHIELD',
    difficulty: 'MEDIUM',
    briefing: {
      situation: 'Three unguided rockets have been launched from the east. Impact Point Prediction indicates two are heading for the defended city. One will land in an open field.',
      objective: 'Engage only the threats targeting the protected area. Conserve interceptors — do not waste ammo on rockets landing in empty fields.',
      constraints: ['IPP active — only engage threats to defended zones', '3 rockets inbound, 2 threatening city', 'Unguided ballistic threats'],
      threatPicture: 'Three Qassam-type unguided rockets on ballistic trajectories. Two predicted to impact within city radius. One heading for open terrain.',
    },
    config: {
      scenario: 'iron_shield',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 3,
      numTargets: 3,
    },
    scoring: {
      parTime: 30,
      timeBonus: 8,
      efficiencyWeight: 0.5,
      accuracyWeight: 0.5,
    },
  },
  {
    id: 'battery-defense',
    name: 'Battery Defense',
    codename: 'IRON FORTRESS',
    difficulty: 'HARD',
    briefing: {
      situation: 'A salvo of 5 Qassam rockets has been launched from eastern Gaza toward the city of Sderot. Your Iron Dome battery must autonomously detect, evaluate, and engage all threats heading for the defended city.',
      objective: 'Let the battery BMC autonomously engage all threats predicted to impact the defended area. Conserve ammo by ignoring rockets landing in open fields.',
      constraints: ['Battery-autonomous engagement', 'IPP-based fire control', '5 rockets inbound', 'Limited magazine depth'],
      threatPicture: 'Five Qassam-type unguided rockets on ballistic trajectories. Battery must handle detection through engagement autonomously.',
    },
    config: {
      scenario: 'iron_dome_battery',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 0,
      numTargets: 5,
    },
    scoring: {
      parTime: 40,
      timeBonus: 6,
      efficiencyWeight: 0.6,
      accuracyWeight: 0.4,
    },
  },
  {
    id: 'dual-battery',
    name: 'Dual Battery',
    codename: 'TWIN SHIELDS',
    difficulty: 'EXTREME',
    briefing: {
      situation: 'A massive Grad rocket salvo of 8 rockets targets two cities simultaneously. Two Iron Dome batteries must coordinate to defend Ashkelon and Beer Sheva against the incoming barrage.',
      objective: 'Both batteries must autonomously engage threats targeting their respective defended cities. No rocket must reach a populated area.',
      constraints: ['Two autonomous batteries', '8 rockets inbound', 'Two defended cities', 'Overlapping sectors'],
      threatPicture: 'Eight Grad rockets across wide frontage. Split between two population centers. Both batteries must engage simultaneously.',
    },
    config: {
      scenario: 'iron_dome_dual_battery',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 0,
      numTargets: 8,
    },
    scoring: {
      parTime: 50,
      timeBonus: 4,
      efficiencyWeight: 0.5,
      accuracyWeight: 0.5,
    },
  },
  {
    id: 'iron-rain',
    name: 'Iron Rain',
    codename: 'IRON RAIN',
    difficulty: 'HARD',
    briefing: {
      situation: 'Intelligence reports a coordinated multi-wave rocket attack. Three waves of mixed Qassam and Grad rockets will arrive at staggered intervals. Some rockets may deploy decoy countermeasures.',
      objective: 'Survive the full barrage. Let the battery BMC manage engagement priorities. Classify and ignore decoys to conserve ammunition.',
      constraints: ['3 waves at 8s intervals', 'Mixed threat types (Qassam + Grad)', 'Decoy countermeasures active', 'Single battery defense'],
      threatPicture: 'Wave 1: 3 Qassam rockets. Wave 2: 4 Grad rockets from south. Wave 3: 5 Qassam rockets from north. Total 12+ threats.',
    },
    config: {
      scenario: 'iron_rain',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 0,
      numTargets: 3,
    },
    scoring: {
      parTime: 45,
      timeBonus: 4,
      efficiencyWeight: 0.6,
      accuracyWeight: 0.4,
    },
  },
  {
    id: 'last-stand',
    name: 'Last Stand',
    codename: 'LAST STAND',
    difficulty: 'EXTREME',
    briefing: {
      situation: 'Massive saturation attack with 4 continuous waves. Battery has reduced ammunition (30 interceptors total). Expect 30+ inbound rockets. This is an ammunition management scenario — every shot must count.',
      objective: 'Defend Ashkelon for as long as possible. Prioritize threats by IPP — only engage rockets heading for the city. Accept that some rockets will get through when ammo runs out.',
      constraints: ['4 waves escalating in size', 'Reduced ammo (30 Tamirs)', '30+ total threats', 'Ammo depletion expected'],
      threatPicture: 'Wave 1: 4 Qassam. Wave 2: 5 Grad. Wave 3: 6 Qassam. Wave 4: 8 Grad + 10 Qassam initial. Over 33 rockets total. Maximum threat.',
    },
    config: {
      scenario: 'last_stand',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 0,
      numTargets: 4,
    },
    scoring: {
      parTime: 55,
      timeBonus: 3,
      efficiencyWeight: 0.7,
      accuracyWeight: 0.3,
    },
  },
  // ─── Phase 8: Multi-Layer Defense ───
  {
    id: 'davids-sling',
    name: "David's Sling",
    codename: 'SLING SHOT',
    difficulty: 'HARD',
    briefing: {
      situation: "Three subsonic cruise missiles detected heading toward Haifa. David's Sling battery is the primary defense layer for this medium-range threat.",
      objective: "Intercept all cruise missiles using Stunner interceptors. David's Sling engages threats at 40-300km range — a tier above Iron Dome.",
      constraints: ["Cruise missiles at 300 m/s", 'Stunner interceptors (Mach 7.5)', "David's Sling min range 40km", 'Medium-range engagement'],
      threatPicture: '3 subsonic cruise missiles approaching from the east at 5km altitude. Each is guided and capable of terminal maneuvers.',
    },
    config: {
      scenario: 'davids_sling',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 0,
      numTargets: 3,
    },
    scoring: {
      parTime: 40,
      timeBonus: 5,
      efficiencyWeight: 0.6,
      accuracyWeight: 0.4,
    },
  },
  {
    id: 'aegis-total',
    name: 'Aegis Total',
    codename: 'AEGIS TOTAL',
    difficulty: 'EXTREME',
    briefing: {
      situation: 'Combined attack: short-range rockets, cruise missiles, and follow-on waves. All three defense tiers activated. TEWA system coordinates engagement across Iron Dome, David\'s Sling, and supporting batteries.',
      objective: 'Defend Tel Aviv and the Air Force Base using layered defense. TEWA automatically assigns each threat to the appropriate tier. Iron Dome handles rockets, David\'s Sling handles cruise missiles.',
      constraints: ['3 defense tiers active', 'Mixed threat types', 'TEWA automated assignment', '3 follow-on waves', '20+ total threats'],
      threatPicture: 'Initial: 4 Qassam rockets. Wave 1 (T+5s): 3 cruise missiles. Wave 2 (T+12s): 5 Qassam rockets. Wave 3 (T+20s): 4 Grad rockets. Multi-axis, multi-layer attack.',
    },
    config: {
      scenario: 'aegis_total',
      guidance: 'proportional_nav',
      evasion: 'none',
      navConstant: 4,
      numInterceptors: 0,
      numTargets: 4,
    },
    scoring: {
      parTime: 50,
      timeBonus: 3,
      efficiencyWeight: 0.7,
      accuracyWeight: 0.3,
    },
  },
];

export function calculateScore(
  scenario: NarrativeScenario,
  simTime: number,
  missDistance: number,
  interceptedCount: number,
  totalTargets: number
): { total: number; grade: string; breakdown: { time: number; accuracy: number; efficiency: number } } {
  const { scoring } = scenario;

  // Time score: bonus for finishing early
  const timeScore = Math.max(0, (scoring.parTime - simTime) * scoring.timeBonus);

  // Accuracy score: based on miss distance (0 = perfect, 200m+ = 0 points)
  const accuracyScore = Math.max(0, 100 - (missDistance / 2));

  // Efficiency score: targets intercepted / total targets * 100
  const efficiencyScore = totalTargets > 0 ? (interceptedCount / totalTargets) * 100 : 0;

  // Weighted total
  const total = Math.round(
    timeScore +
    accuracyScore * scoring.accuracyWeight +
    efficiencyScore * scoring.efficiencyWeight
  );

  // Letter grade
  let grade: string;
  if (total >= 150) grade = 'S';
  else if (total >= 120) grade = 'A';
  else if (total >= 90) grade = 'B';
  else if (total >= 60) grade = 'C';
  else if (total >= 30) grade = 'D';
  else grade = 'F';

  return {
    total,
    grade,
    breakdown: {
      time: Math.round(timeScore),
      accuracy: Math.round(accuracyScore * scoring.accuracyWeight),
      efficiency: Math.round(efficiencyScore * scoring.efficiencyWeight),
    },
  };
}
