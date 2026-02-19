/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem: "Next Round".
 *
 * This program determines the number of participants who will advance to the next round
 * of a competition. Participants advance if their score is greater than or equal to
 * the score of the k-th place finisher, and their score is strictly positive.
 *
 * Algorithm: Iterate through scores, compare with k-th score and positivity condition.
 * Time Complexity: O(N), where N is the number of participants.
 */
#include <iostream> // Functional Utility: Included for standard input/output operations.
using namespace std; // Functional Utility: Uses the standard namespace to avoid prefixing standard library elements with std::

int main() {
  // Functional Utility: Optimizes standard I/O streams for faster execution,
  // crucial in competitive programming for time-sensitive tasks.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Variable Declarations:
  // n: Total number of participants.
  // k: The k-th position whose score determines the cutoff.
  // a[50]: Array to store the scores of up to 50 participants.
  // c: Counter for participants advancing to the next round.
  int n, k, a[50], c = 0;

  // Input: Reads the total number of participants (n) and the k-th position (k).
  cin >> n >> k;

  // Block Logic: Reads the scores of all 'n' participants into array 'a'.
  // Invariant: After this loop, 'a' contains all participant scores.
  for (int i = 0; i < n; i++)
    cin >> a[i];

  // Block Logic: Iterates through all participant scores to count those who qualify.
  // Invariant: For each iteration, `a[i]` is the current participant's score being evaluated.
  for (int i = 0; i < n; i++) {
    // Block Logic: Checks qualification criteria: score >= k-th score AND score > 0.
    // Precondition: `a[k-1]` holds the score of the k-th place finisher (0-indexed).
    if (a[i] >= a[k - 1] && a[i] > 0)
      c++; // Inline: Increments the counter if the participant qualifies.
  }

  // Output: Prints the total count of participants who advance.
  cout << c;
  return 0; // Functional Utility: Indicates successful program execution.
}
