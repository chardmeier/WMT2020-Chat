# Problems found in the Baseline translations

These are some observations based on a handful of documents from the Taskmaster
training set, translated with Uli's EN-DE news system.

## Lexicon

- In general, the translations are _very_ unidiomatic, even when they're correct. English idioms
  are translated literally into German most of the time, which often sounds very odd and sometimes
  distorts the meaning (e.g., "I'm dying for a coffee" - can only be interpreted literally in German;
  "I'll get back to you" - indicates motion in German)
  
- Terminology is a problem. The system frequently mistranslates words that we can expect to occur
  a lot in the dialogues (e.g., "theatre" - can only mean "playhouse", not "cinema" in German;
  "pine apple" translated as "pine tree apple"; "how many people are coming" translated as "how
  many human beings...", etc.)

- Named entity: Usually named entities turn into nonsense when translated. Names of companies,
  restaurants, cinemas etc. should usually be left alone, though there might be exceptions if
  a part of a name is purely descriptive. Film titles would typically be
  translated for a German audience (but not literally - would require database lookup),
  but in Switzerland the original English title would be used.
  
# Grammar

- Elliptical sentences sometimes require case marking (e.g., in "What would you like? A coffee.",
  "a coffee" should be in accusative case because it's an argument of the verb "like".)

- Literal translations of the English tenses are often quite unnatural. In spoken German, you
  wouldn't normally use simple past except for a handful of very common verbs (to be, to have),
  and future tense would be used far less frequently. Present perfect (for past) and present
  tense (for future) would be preferred instead.

# Pronouns

- Addressee reference pronouns are a major problems. "You" should be translated consistently as "Sie"
  or "du". Typically, "Sie" would be the appropriate form in these dialogues if the addressee was
  a human, but there might be exceptions in very informal dialogues (such as the example starting
  with "Hey man what's up?"), and people might be more inclined to use "du" if they believe they're
  talking to a machine.

- The English source sometimes contains politeness markers such as "sir" that should be removed as
  they're never translated correctly. In German, this level of politeness would be marked by using
  "Sie" inflections, not by adding polite words.

- The German "Sie" form of address is homonymous with the 3rd person pronoun. In some of the dialogues,
  this creates ambiguities when the agent uses "they" to refer to the seller ("They don't
  have any seats left."), but the German translation could be read as referring to the customer
  ("You don't have any seats left." -> confusion). In these cases, it would be better and more
  idiomatic to use an impersonal paraphrase in German ("There are no seats left.").
  
- There are some examples where pronominal anaphora gets translated incorrectly, but I only saw
  one or two of them in the dialogues I checked, and the coreference resolver performed poorly
  on them as well.


