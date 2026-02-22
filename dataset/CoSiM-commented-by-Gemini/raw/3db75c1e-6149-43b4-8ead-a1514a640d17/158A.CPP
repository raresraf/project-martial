/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem (Codeforces 158A).
 *
 * @algorithm
 * The problem is to count participants who advance to the next round. A participant
 * advances if their score is positive and also greater than or equal to the
 * score of the k-th place finisher.
 *
 * This solution uses a simple and clear two-pass approach:
 * 1. Read all participant scores into an array.
 * 2. Identify the score of the k-th participant, which becomes the threshold.
 * 3. Iterate through the array of scores again.
 * 4. For each score, check if it meets both conditions:
 *    a) It is greater than 0.
 *    b) It is greater than or equal to the threshold score.
 * 5. Count how many participants meet these criteria.
 * 6. Print the final count.
 *
 * @complexity
 * - Time Complexity: O(N), for the two separate passes over the N participants.
 * - Space Complexity: O(N), for storing the scores in an array.
 */
#include <iostream>
using namespace std;
// Global variables are used, a common practice in competitive programming.
int n, k;       // n: number of participants, k: the rank to use as a threshold.
int a, b = 0;   // a: temporary for input, b: counter for advancing participants.
int mas[10000]; // Array to store the scores of all participants.
int main() {
  cin >> n >> k;
  /**
   * @brief First pass: Read all n scores into the `mas` array.
   * Note: This code uses 1-based indexing for the array (i starts at 1),
   * which means mas[0] is unused.
   */
  for (int i = 1; i <= n; i++) {
    cin >> a;
    mas[i] = a;
  }
  /**
   * @brief Second pass: Iterate through the scores to count who advances.
   */
  for (int j = 1; j <= n; j++) {
    /**
     * @brief The core logic for advancing. A participant advances if their score
     * is positive AND greater than or equal to the k-th participant's score.
     * `mas[k]` is the threshold score.
     */
    if (mas[j] >= mas[k] && mas[j] > 0) {
      b++;
    }
  }
  cout << b;
}