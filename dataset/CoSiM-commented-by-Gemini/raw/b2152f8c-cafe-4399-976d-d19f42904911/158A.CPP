/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem.
 * @details This program determines how many participants will advance to the next round of a contest.
 * A participant advances if their score is positive and is greater than or equal to the score
 * of the k-th place finisher. The input scores are provided in non-increasing order.
 *
 * @algorithm
 * 1. Read the total number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into a vector.
 * 3. Identify the score of the k-th place finisher. This score becomes the cutoff threshold.
 *    Since the scores are 0-indexed, this is the score at index `k-1`.
 * 4. Iterate through the list of scores again.
 * 5. For each score, check if it is greater than or equal to the cutoff score AND is positive.
 * 6. Count the number of participants who satisfy both conditions.
 * 7. Print the final count.
 *
 * @complexity
 * - Time: O(N), for reading the N scores and then iterating through them once to count.
 * - Space: O(N), to store the scores in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: The total number of participants.
  // k: The rank that determines the cutoff score.
  // j: The counter for participants who advance.
  // temp: A temporary variable to read each score.
  int n, k, j = 0, temp;
  cin >> n >> k;
  
  vector<int> a; // Vector to store the scores of all participants.

  /**
   * @brief Reads all n scores into the vector `a`.
   */
  for (int i = 0; i < n; ++i) {
    cin >> temp;
    a.push_back(temp);
  }

  // Determine the cutoff score from the k-th participant (at index k-1).
  int x = a[k - 1];

  /**
   * @brief Counts the number of participants who advance.
   * Iterates through all participants and checks if they meet the advancement criteria.
   */
  for (int i = 0; i < n; ++i) {
    // A participant advances if their score is >= the k-th score and is positive.
    if ((a[i] >= x) && a[i] > 0) {
      j++;
    }
  }
  
  // Output the final count of advancing participants.
  cout << j;
  return 0;
}
