/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" competitive programming problem.
 *
 * This program determines the number of participants who will advance to the next round.
 * The rule for advancing is that a participant's score must be positive AND greater
 * than or equal to the score of the k-th place participant.
 *
 * @algorithm
 * 1. Read the total number of participants, `n`, and the benchmark rank, `k`.
 * 2. Create a vector `array1` of size `n` to store the scores.
 * 3. Read all `n` scores from standard input and store them in the vector.
 * 4. The score of the k-th participant is at index `k-1`. This is the benchmark score.
 * 5. Iterate through the scores in the vector from the beginning.
 * 6. For each score, check if it meets both conditions for advancement:
 *    a. The score is positive (`array1[i] > 0`).
 *    b. The score is greater than or equal to the benchmark score (`array1[i] >= array1[k-1]`).
 * 7. If both conditions are true, increment a counter `sum`.
 * 8. After checking all scores, print the final value of `sum`.
 *
 * @implementation_note
 * This implementation reads all scores into a vector first, then iterates through it again
 * to find the result. This is straightforward but less memory-efficient than a single-pass
 * approach, as it requires O(n) space for the vector.
 *
 * @complexity
 * Time Complexity: O(n), due to two separate loops that each iterate up to `n` times.
 * Space Complexity: O(n), for storing the `n` scores in a vector.
 */
#include <iostream>
#include <vector>

using namespace std;

int main() {
  // `n` is the number of participants, `k` is the benchmark rank.
  int n, k;
  // `sum` will count the number of participants who advance.
  int sum = 0;
  cin >> n >> k;
  
  // A vector to store the scores of all `n` participants.
  vector<int> array1(n);

  // First loop: Read all scores into the vector.
  for (int i = 0; i < n; i++) {
    cin >> array1[i];
  }

  /**
   * @brief Second loop: Count the participants who qualify.
   *
   * An invariant is that the scores are given in non-increasing order, but this code
   * does not rely on that property, making it more robust but potentially less efficient
   * if that property could be exploited.
   */
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is positive and >= the k-th person's score.
    // The k-th person's score is at index k-1.
    if (array1[i] >= array1[k - 1] && array1[i] > 0) {
      sum += 1;
    }
  }
  
  // Output the final count.
  cout << sum << endl;
  return 0;
}
