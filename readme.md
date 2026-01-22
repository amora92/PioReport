## PioSOLVER Report Analyzer: Board Classification Guide

This guide breaks down the three main categories used to classify each flop, which helps in analyzing solver strategies and recognizing patterns.

### 1. Suitedness (Flush Draw Potential)

This is the most straightforward classification, based on the number of cards of the same suit on the flop.

| **Classification**        | **Rule**                                                | **Meaning**                                 |
| ------------------------------- | ------------------------------------------------------------- | ------------------------------------------------- |
| **Monotone Boards**       | 3 cards of the same suit (e.g.,**$A♠K♠Q♠$**)       | Flush is made immediately. Heavy betting.         |
| **Two-Tone (Flush Draw)** | 2 cards of the same suit (e.g.,**$A♠K♠Q♦$**)       | Flush draws are possible. Very common board type. |
| **Rainbow**               | All 3 cards of different suits (e.g.,**$A♠K♦Q♥$**) | No flush draws are possible.                      |

---

### 2. Rank Composition (Card Ranks/Values)

This category focuses on the **card values** to identify how high/low the board is and if **Broadway cards** (Ten or higher) are present.

| **Classification**    | **Rule**                                                    | **Meaning**                                                     |
| --------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Three Broadway**    | All 3 cards are Jack or higher (J, Q, K, A)                       | Very high, top-heavy board.                                           |
| **Two Broadway**      | 2 cards are Jack or higher (J, Q, K, A)                           | High/Mid board with good connectivity potential.                      |
| **Broadway/Ten High** | 1 card is Jack or higher**and**another is a Ten (T)         | Still a high, connected board.                                        |
| **High Card**         | Highest card is Ace or King, and no more than two broadways/tens. | Generally**$Axx$**or**$Kxx$**boards.                        |
| **Mid Cards**         | Highest card is 7 through 9, and no card is 6 or lower.           | Mid-value boards, e.g.,**$987$**or**$975$**.          |
| **Low/Mid Cards**     | At least one card is 6 or lower (2 through 6).                    | Low to mixed value boards, e.g.,**$J54$**or**$863$**. |

---

### 3. Connectivity (Straight Draw Potential)

This is based on the  **gaps between the three card ranks** , determining how many hands have an open-ended or gutshot straight draw.

| **Classification**        | **Rule**                                                  | **Example**                                | **Meaning (Draw Potential)**                             |
| ------------------------------- | --------------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| **Paired/Tripled Boards** | 2 or 3 cards of the same rank.                                  | **$A♠A♦5♥$**or**$K♣K♠K♥$** | High chance for full houses/quads.**No**straight draws.  |
| **Connected**             | 0 gaps between ranks (running ranks) or wheel straight (A-5-4). | **$876$**or**$54A$**             | **Open-ended straight draw**is extremely common.         |
| **One-Gapped**            | At most 1 gap between any two adjacent ranks.                   | **$975$**or**$A43$**             | Many hands have a**gutshot**draw.                        |
| **Two-Gapped**            | At most 2 gaps between any two adjacent ranks (e.g., 9-6-3).    | **$J85$**or**$T74$**             | Draw potential is lower, usually requiring two high/low cards. |
| **Disconnected**          | More than 2 gaps between ranks (e.g., 9-5-2).                   | **$K72$**or**$A63$**             | Very low draw potential.                                       |
