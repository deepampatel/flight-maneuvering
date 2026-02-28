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
