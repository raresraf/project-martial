/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A - "Next Round".
 *
 * @details
 * This program determines the number of participants who advance to the next round.
 * A participant advances if their score is positive and greater than or equal to the
 * score of the k-th ranked participant.
 *
 * Algorithm:
 * 1. Read the total number of participants (n) and the rank (k).
 * 2. Adjust k to be a zero-based index (k--).
 * 3. Read all 'n' scores into a std::vector. While reading, identify and store the
 *    score of the k-th participant as the threshold score.
 * 4. Iterate through the vector of scores a second time.
 * 5. Count how many scores are both positive (non-zero) and greater than or
 *    equal to the threshold score.
 * 6. Print the final count.
 *
 * Time Complexity: O(N), for two linear passes over the N participants' scores.
 * Space Complexity: O(N), to store the scores in a std::vector.
 */
#include <iostream>
#include <vector>
using namespace std;
int main() {
  typedef long long ll;
  // n: number of participants, k: rank threshold
  // m: stores the score of the k-th participant (the threshold)
  // t: temporary variable for reading scores
  ll n, k, m, t;
  cin >> n >> k;
  // Adjust k from 1-based rank to 0-based index.
  k--;
  vector<ll> v;

  /**
   * Block Logic: Reads all scores and identifies the threshold score.
   * This loop populates the vector `v` with all participant scores.
   * It simultaneously finds the score of the k-th participant.
   */
  for (ll i = 0; i < n; i++) {
    cin >> t;
    if (k == i)
      m = t; // This is the score to beat.
    v.push_back(t);
  }

  ll count = 0;
  /**
   * Block Logic: Counts the number of qualifying participants.
   * This loop iterates through all scores to check them against the advancement criteria.
   */
  for (auto i = v.begin(); i < v.end(); i++) {
    // Pre-condition: To advance, a score must be >= the threshold score AND be positive.
    if (*i >= m && *i != 0) {
      count++;
    }
  }
  
  // Output the final count.
  cout << count << "
";
}
