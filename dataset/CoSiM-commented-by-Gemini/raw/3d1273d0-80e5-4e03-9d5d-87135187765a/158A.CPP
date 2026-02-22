/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem (Codeforces 158A).
 *
 * @algorithm
 * The problem is to count how many participants advance to the next round.
 * A participant advances if they have a positive score and their score is
 * equal to or greater than the score of the k-th place finisher. The input
 * scores are guaranteed to be in non-increasing order.
 *
 * This solution processes the scores in a single pass without storing the
 * entire array, which is possible due to the non-increasing score property.
 *
 * 1. Read `n` (number of participants) and `k` (the rank to check against).
 * 2. Loop `n` times, reading one score at a time.
 * 3. Identify the score of the k-th participant. This score becomes the threshold.
 * 4. Two main scenarios are handled:
 *    a) If the k-th participant's score is 0, then no one with a 0 score can
 *       advance. Since scores are non-increasing, no one after the k-th
 *       person can advance either. The answer is simply the count of
 *       participants with positive scores from ranks 1 to k.
 *    b) If the k-th participant's score is positive, then all participants from
 *       ranks 1 to k automatically advance. We then continue checking
 *       participants after rank k. Any participant with a score equal to the
 *       k-th score also advances. We stop counting as soon as we find a
 *       participant with a lower score.
 *
 * @complexity
 * - Time Complexity: O(N), as it iterates through the participants once.
 * - Space Complexity: O(1), as only a few variables are used for storage.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  // a: current participant's score
  // b: score of the k-th participant (the threshold score)
  // s: counter for advancing participants, used when k-th score > 0
  // h: counter for participants with positive scores, used when k-th score == 0
  int a, b, s = 0, h = 0;
  cin >> n >> k;

  /**
   * @brief Main loop to process `n` participants in a single pass.
   */
  for (int i = 0; i < n; i++) {

    cin >> a;
    // Pre-condition for advancing is having a positive score.
    if (a > 0) {
      h++;
    }
    
    // Logic for when we are at the k-th participant (0-indexed).
    if (i == k - 1) {
      b = a; // This is the threshold score.
      
      // If the threshold score is 0, no one with a 0 can advance.
      // Since scores are non-increasing, we can stop and report the count
      // of positive scores seen so far.
      if (b == 0) {
        cout << h;
        return 0;
      } else {
        // If the threshold score is positive, at least the first k will advance.
        s = k;
      }
    }
    
    // Logic for participants after the k-th position.
    if ((i >= k) && (a == b)) {
      // If a participant has the same score as the k-th, they also advance.
      s++;
    } else if (i >= k) {
      // If the score is different (i.e., lower), no one else can advance.
      // So, we break the loop.
      break;
    }
  }
  // Print the final count. This line is reached if the k-th score was positive,
  // or if n < k.
  cout << s;

  return 0;
}