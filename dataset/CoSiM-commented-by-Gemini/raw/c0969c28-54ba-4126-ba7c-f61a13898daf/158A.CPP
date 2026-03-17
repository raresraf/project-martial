/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 158A "Next Round".
 *
 * @details
 * This program calculates the number of participants who will advance to the next
 * round of a contest. The rule for advancing is that a participant's score must
 * be positive and greater than or equal to the score of the k-th place finisher.
 *
 * @algorithm
 * The solution uses a single pass through the input, which is possible because the
 * problem guarantees that the participants' scores are provided in non-increasing order.
 * 1. Read `n` (number of participants) and `k` (the rank to benchmark against).
 * 2. A benchmark score, `kth`, is initialized to 1 to implicitly handle the "must be
 *    positive" rule.
 * 3. It iterates from the 1st to the n-th participant, reading each score.
 * 4. When the k-th participant is reached, their score is captured as the benchmark `kth`.
 * 5. For every participant, their score is compared against the current `kth` value.
 *    - For participants 1 to k-1, they are compared against the initial `kth` value of 1.
 *    - For participants k to n, they are compared against the actual k-th place score.
 * 6. A counter is incremented for each participant who meets the criteria.
 * This single-pass logic works correctly *only* because the input is sorted.
 *
 * @complexity
 * - Time Complexity: O(N), for the single loop through the N participants.
 * - Space Complexity: O(1), as scores are processed one by one without being stored.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main(int argc, char const *argv[]) {
  int n, k, temp;
  cin >> n >> k;

  int count = 0; // The number of participants who advance.
  
  // The score of the k-th place finisher. Initialized to 1 to enforce the
  // rule that advancing scores must be positive.
  int kth = 1; 

  /**
   * @brief A single pass to read scores and count advancing participants.
   */
  for (int i = 1; i <= n; ++i) {
    cin >> temp; // Read the score of the current participant.

    // When we reach the k-th participant, we establish the benchmark score.
    if (i == k)
      kth = max(kth, temp);

    // Increment the counter if the current score is high enough.
    // The boolean expression `(temp >= kth)` evaluates to 1 if true, 0 if false.
    count += (temp >= kth);
  }
  
  cout << count << endl;
  return 0;
}
