/**
 * @file 158A.CPP
 * @brief Solution for a competitive programming problem about qualifying participants.
 *
 * @details This program determines how many participants will advance to the
 * next round of a competition. The rule is that a participant advances if they
 * have a positive score and their score is greater than or equal to the score
 * of the k-th ranked participant.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the total number of participants, `n`, and the rank `k`.
 * 2. Read the scores of all `n` participants into an array.
 * 3. Identify the score of the k-th participant (at index `k-1`). This is the "cutoff" score.
 * 4. Iterate through all participants' scores.
 * 5. Count how many participants have a score that is both positive and greater than or equal to the cutoff score.
 * 6. Print the final count.
 *
 * @section complexity
 * Time Complexity: O(N), where N is the number of participants. This is due to the two loops that each iterate up to N times.
 * Space Complexity: O(N), to store the scores of the participants in an array.
 */
#include <cstdio>

int main(void) {
  int v[64];
  // This loop structure is often used for multiple test cases, but for this
  // specific problem, it will only execute once.
  for (int k, n; scanf("%d %d", &n, &k) == 2;) {
    int ans = 0;
    // Block Logic: Read all participant scores into the array 'v'.
    for (int i = 0; i < n; ++i)
      scanf("%d", v + i);
    
    // Block Logic: Count participants who qualify.
    // A participant qualifies if their score is positive AND greater than or
    // equal to the score of the k-th person (v[k-1]).
    // The expression `v[i] && v[i] >= v[k-1]` cleverly combines two conditions:
    // 1. `v[i]` checks if the score is non-zero (evaluates to true).
    // 2. `v[i] >= v[k-1]` checks if the score meets the cutoff.
    for (int i = 0; i < n; ++i)
      ans += v[i] && v[i] >= v[k - 1];
    
    printf("%d\n", ans);
  }
  return 0;
}
