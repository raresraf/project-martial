
#include <bits/stdc++.h>

using namespace std;

/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A, "Next Round".
 * @details The program determines the number of participants who will advance to the
 * next round of a competition.
 *
 * Algorithm:
 * A participant advances if their score is positive and greater than or equal to the
 * score of the k-th ranked participant. The program reads the total number of
 * participants (n) and the rank (k) to benchmark against. It then reads all scores
 * into an array. It iterates through the scores, counting how many meet the
 * advancement criteria: score > 0 and score >= scores[k-1].
 *
 * Time Complexity: O(N), where N is the number of participants, due to the two
 * separate loops for reading input and counting qualifiers.
 * Space Complexity: O(N) to store the scores of the participants.
 */


// Type definitions for convenience in competitive programming.
typedef long long ll;
typedef vector<ll> vi;
// Macro for a common for-loop structure.
#define rep(i, a, b) for (int i = a; i < b; i++)

int main() {
  // n: total number of participants
  // k: the rank of the participant whose score is the threshold
  int n, k;
  cin >> n >> k;

  // Vector to store the scores of all n participants.
  vi v(n);
  rep(i, 0, n) { cin >> v[i]; }

  int count = 0;
  /**
   * @brief Count participants who advance.
   *
   * Pre-condition: Scores are populated in the vector `v`.
   * Invariant: `count` holds the number of participants checked so far who
   *            meet the advancement criteria.
   *
   * Logic: A participant advances if their score is positive AND greater than
   * or equal to the score of the k-th place finisher (v[k-1]).
   */
  rep(i, 0, n) {
    if (v[i] >= v[k - 1] && v[i] > 0) {
      count++;
    }
  }

  // Output the final count of advancing participants.
  cout << count;
  return 0;
}
