/**
 * @file 158A.CPP
 * @brief Counts participants advancing to the next round of a contest.
 *
 * This program determines how many participants will advance based on their scores.
 * The rule is that a participant advances if their score is positive and is
 * greater than or equal to the score of the k-th place finisher.
 *
 * @algorithm The program reads the number of participants `n` and the rank `k`.
 * It reads all `n` scores into an array. It then iterates through the scores again,
 * comparing each score to the k-th participant's score. A counter is incremented
 * for each participant who meets the advancement criteria (positive score and score >= k-th score).
 *
 * @note This code has several issues:
 * 1.  It uses a Variable Length Array (VLA) `int participantes[n];`, which is a
 *     non-standard C++ extension.
 * 2.  It uses 1-based indexing in its loops (`for (int i = 1; i <= n;...`) for a
 *     0-indexed C-style array. This will lead to out-of-bounds access when `i` is `n`,
 *     which is undefined behavior. The k-th participant's score is at index `k-1`,
 *     but the code accesses `participantes[k]`.
 * 3.  The advancement check should be against a positive score, but the condition
 *     `participantes[i] != 0` is used. A score can be positive without this check if the
 *     k-th score is positive. The check is only truly necessary if `participantes[k]` is 0.
 *
 * @time_complexity O(N), where N is the number of participants, due to the loops
 * for reading and checking scores.
 * @space_complexity O(N) to store the scores of the participants.
 */
#include <iostream>

using namespace std;

int main() {
  int n, k; // n: number of participants, k: the rank to compare against.
  int c = 0; // Counter for participants who advance.
  cin >> n >> k;
  
  // VLA and 1-based indexing are used, which is problematic.
  int participantes[n]; 

  /**
   * @brief Loop to read the scores of all participants.
   * @warning This loop uses 1-based indexing `i <= n`, which causes an
   * out-of-bounds write at `participantes[n]` on the last iteration.
   */
  for (int i = 1; i <= n; i++) {
    cin >> participantes[i];
  }
  
  /**
   * @brief Loop to count the number of participants who advance.
   * @warning This loop also uses 1-based indexing, causing out-of-bounds reads.
   * The comparison `participantes[i] >= participantes[k]` is incorrect for 0-indexed
   * arrays; it should be `participantes[i-1] >= participantes[k-1]`.
   */
  for (int i = 1; i <= n; i++) {
    // A participant advances if their score is >= the k-th score and is positive.
    if (participantes[i] >= participantes[k]) {
      // The condition `participantes[i] != 0` checks for a positive score.
      if (participantes[i] != 0)
        c++;
    }
  }
  
  // Print the final count.
  cout << c;
  return 0;
}
