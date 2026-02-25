/**
 * @file 158A.CPP
 * @brief Solution to Codeforces problem 158A (Next Round).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * The problem is to determine the number of contestants who will advance to the
 * next round. A contestant advances if their score is positive and is greater
 * than or equal to the score of the k-th place finisher. The input scores are
 * provided in non-increasing order.
 *
 * @section algorithm
 * The correct algorithm is:
 * 1. Read the number of contestants `n` and the rank `k`.
 * 2. Read all `n` scores into an array or vector.
 * 3. Identify the score of the k-th place finisher (`scores[k-1]`). This is the
 *    threshold score.
 * 4. Iterate through all the scores. Count how many contestants have a score
 *    that is both positive (`> 0`) and greater than or equal to the threshold
 *    score.
 * 5. Print the final count.
 *
 * @section implementation_analysis
 * The provided code attempts to solve the problem in a single pass without
 * storing the entire array of scores. While this is possible, the implementation
 * is convoluted and contains a critical bug.
 *
 * The code tries to separate the logic for three cases:
 * - Contestants before the k-th position.
 * - The contestant at the k-th position.
 * - Contestants after the k-th position.
 *
 * The variable `temp` is intended to store the k-th person's score to use as a
 * threshold for subsequent contestants.
 *
 * @subsection bug_analysis
 * A major bug exists: if the k-th contestant's score is 0, the `else if (i + 1 == k && input > 0)`
 * condition is false. Consequently, `temp` is never assigned a value and remains
 * uninitialized. The subsequent comparisons `input >= temp` result in undefined
 * behavior, leading to an incorrect answer.
 *
 * A simpler and more robust solution would be to store all scores first, as
 * described in the algorithm section.
 *
 * @section complexity
 * - Time Complexity: O(N), as the code iterates through the `n` contestants once.
 * - Space Complexity: O(1) (excluding input storage), as it attempts to process
 *   scores on the fly. A correct array-based solution would have O(N) space complexity.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  int input;
  int advance = 0;
  // `temp` is intended to store the score of the k-th participant.
  // BUG: `temp` is uninitialized. If the k-th score is 0, `temp` is never set,
  // leading to undefined behavior in later comparisons.
  int temp; 

  for (int i = 0; i < n; i++) {
    cin >> input;

    // This block of if-else statements implements a complex and buggy logic
    // to count advancing contestants in a single pass.
    if (i < k - 1 && input > 0) {
      // Handles contestants before the k-th position.
      // If their score is positive, they are counted. This assumes all scores
      // before a positive k-th score are also positive and advancing.
      advance++;
    } else if (i + 1 == k && input > 0) {
      // Handles the k-th contestant. If their score is positive,
      // it becomes the threshold score, and they are counted.
      temp = input;
      advance++;
    } else if (input >= temp && input > 0) {
      // Handles contestants after the k-th position.
      // They advance if their score is positive and meets the threshold.
      // This relies on `temp` being correctly initialized, which may not happen.
      // A check for input > 0 is also missing in the original code for this branch.
      advance++;
    }
  }

  // Print the final count of advancing contestants.
  cout << advance;

  return 0;
}
