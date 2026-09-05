import { toGloss, toEnglish } from "../grammar";
import { VOCABULARY } from "../vocabulary.generated";

const available = new Set(VOCABULARY.map((e) => e.gloss));

const english = [
  "What is your name?",
  "I want water",
  "Where is the hospital?",
  "I will go to the hospital tomorrow",
  "I don't understand",
  "Please help me",
  "How much does it cost?",
  "My mother is sick",
  "I went home yesterday",
  "She is a doctor",
  "Do you have medicine?",
  "I am hungry",
  "Thank you very much",
  "Can you help me please",
  "I need a doctor now",
  "How many children do you have?",
  "The food is very good",
  "I cannot hear you",
  "Call an ambulance",
  "My stomach hurts",
];
console.log("=== ENGLISH -> ISL GLOSS ===");
for (const s of english) {
  const r = toGloss(s, available);
  const miss = r.unmatched.length ? `   [no sign: ${r.unmatched.join(", ")}]` : "";
  console.log(`${s.padEnd(36)} -> ${r.notation}${miss}`);
}

/**
 * Clause type is carried on the FACE, and it is scoped to particular signs — a
 * head shake that starts one word early negates the wrong thing. The per-sign
 * markers are printed rather than only the clause label, because the scope is
 * the part that is easy to get wrong and impossible to see in a gloss.
 */
console.log("\n=== NON-MANUAL MARKING ===");
const marked = [
  "What is your name?",
  "Are you a doctor?",
  "You are a doctor",
  "I do not want medicine",
  "Tomorrow I will go to the hospital",
  "Thank you",
  "My stomach hurts",
];
for (const s of marked) {
  const r = toGloss(s, available);
  console.log(`\n${s}\n  ${r.notation}   [${r.clause}]  ${r.markers.join(" · ")}`);
  for (const t of r.tokens) {
    const on = (Object.keys(t.face) as (keyof typeof t.face)[])
      .filter((k) => Math.abs(t.face[k]) > 0.01)
      .map((k) => `${k} ${t.face[k].toFixed(2)}`);
    console.log(`    ${t.gloss.toUpperCase().padEnd(12)} ${on.join("  ") || "—"}`);
  }
}

console.log("\n=== ISL GLOSS -> ENGLISH ===");
const glosses: string[][] = [
  ["you", "name", "what"],
  ["i", "water", "want"],
  ["hospital", "where"],
  ["tomorrow", "i", "hospital", "go"],
  ["i", "understand", "no"],
  ["please", "help"],
  ["price", "howmuch"],
  ["mother", "sick"],
  ["yesterday", "i", "home", "go"],
  ["she", "doctor"],
  ["you", "medicine", "have"],
  ["i", "hungry"],
  ["thankyou"],
  ["i", "doctor", "need"],
  ["food", "good"],
  ["i", "you", "listen", "cannot"],
  ["ambulance", "call"],
  ["i", "stomach", "pain"],
  ["hello"],
  ["you", "how"],
];
for (const g of glosses) {
  console.log(`${g.join(" ").toUpperCase().padEnd(34)} -> ${toEnglish(g)}`);
}
