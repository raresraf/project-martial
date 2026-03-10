/**
 * @file 158A.CPP
 * @brief This program solves the "Next Round" problem (Codeforces 158A).
 *   It counts the number of participants who qualify for the next round of a competition.
 *   Qualification criteria: a participant must score at least as much as the `k`-th placed
 *   participant and must have a positive score.
 *   Problem description: https://codeforces.com/problemset/problem/158/A
 * * Algorithm: Array traversal and conditional counting.
 *   1. Reads `n` participant scores into a vector, and identifies the `k`-th score during this process.
 *   2. Iterates through all scores in the vector, applying the qualification criteria to count qualifiers.
 * Time Complexity: O(N) where N is the number of participants, due to two linear passes
 *   (one for input/finding the `k`-th score, one for counting qualifiers).
 * Space Complexity: O(N) to store the `v` vector.
 */
#include <iostream> // Required for standard input/output operations.
#include <vector>   // Required for std::vector.
using namespace std;

/**
 * @brief Main function to count qualifying participants.
 * Functional Utility: Reads participant count, cutoff rank, and scores, then
 *                     applies qualification rules to determine and print the number of qualifiers.
 */
int main() {
  typedef long long ll; // Functional Utility: Alias for `long long`, used for potentially larger integer values.
  ll n, k, m, t;        // `n`: total participants, `k`: 1-indexed rank for cutoff,
                        // `m`: stores the `k`-th score, `t`: temporary for reading scores.
  // Block Logic: Reads `n` and `k`.
  cin >> n >> k;
  k--; // Functional Utility: Converts `k` from 1-indexed to 0-indexed for array/vector access.
  // Functional Utility: Creates a vector of `long long` integers to store participant scores.
  vector<ll> v;
  /**
   * Block Logic: Reads `n` participant scores from input and stores them in the `v` vector.
   * Simultaneously, it identifies and stores the score of the `k`-th participant (0-indexed) into `m`.
   * Pre-condition: `n` is a non-negative integer, `k` is a valid 1-indexed rank.
   * Invariant: `v` is populated with all scores, and `m` holds the cutoff score.
   */
  for (ll i = 0; i < n; i++) {
    cin >> t;
    if (k == i) // Conditional Logic: If the current index `i` matches the (0-indexed) cutoff rank `k`.
      m = t;    // Stores this score as the cutoff score `m`.
    v.push_back(t); // Adds the score to the vector.
  }
  ll count = 0; // Accumulates the count of qualifying participants.
  /**
   * Block Logic: Iterates through all participant scores in `v` to check qualification.
   * Pre-condition: `v` contains all scores, and `m` defines the cutoff score.
   * Invariant: `count` increments for each participant meeting the criteria.
   * Inline: `auto i = v.begin(); i < v.end(); i++` uses iterators for traversal.
   */
  for (auto i = v.begin(); i < v.end(); i++) {
    /**
     * Conditional Logic: Checks two conditions for qualification:
     *   1. `*i >= m`: The participant's score (`*i`) must be greater than or equal to
     *      the cutoff score (`m`).
     *   2. `*i != 0`: The participant's score must be non-zero (positive, as scores are non-negative).
     */
    if (*i >= m && *i != 0) {
      count++; // Increments the counter if both conditions are met.
    }
  }
  // Block Logic: Prints the total count of qualifying participants followed by a newline.
  cout << count << "\n";
}