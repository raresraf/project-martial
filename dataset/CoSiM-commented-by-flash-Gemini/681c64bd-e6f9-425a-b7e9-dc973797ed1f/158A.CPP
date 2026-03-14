/**
 * @681c64bd-e6f9-425a-b7e9-dc973797ed1f/158A.CPP
 * @brief This program solves Codeforces problem 158A, "Next Round".
 * It determines the number of participants who will advance to the next round
 * of a competition. Qualification criteria: a participant must score equal to
 * or greater than the k-th place finisher's score, and must have a positive score.
 *
 * Algorithm: This solution first reads all scores, then identifies the threshold
 * score based on the k-th participant. It then iterates from the k-th participant
 * onwards to find all participants with the same score, and then iterates backward
 * from the last qualifying participant to exclude those with zero scores.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing scores in a vector.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream> and <vector>.

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to determine qualifying participants based on scores.
 * Reads number of participants and k-th place, then all scores, and counts qualifiers.
 * @return 0 on successful execution.
 */
int main() {

  int n, k; // Declares 'n' for the number of participants, and 'k' for the k-th place.
  cin >> n >> k; // Reads 'n' and 'k' from standard input.

  vector<int> v; // Declares a vector to store the scores of 'n' participants.

  /**
   * Block Logic: Reads 'n' scores from standard input into the vector `v`.
   * Invariant: After this loop, `v` contains all participant scores in input order.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each participant.
    int x; // Temporary variable to read each score.
    cin >> x; // Reads each participant's score.
    v.push_back(x); // Adds the score to the vector.
  }
  
  // `r` is initialized to `k-1`, which is the 0-indexed position of the k-th participant.
  // This `r` will be used as the index of the last qualifying participant.
  int r = k - 1; 
  
  /**
   * Block Logic: Extends `r` to include all participants whose score is equal to
   * the k-th participant's score and are located after the initial k-th position.
   * This handles ties for the k-th place.
   * Pre-condition: `v[k-1]` is the k-th score.
   * Invariant: `r` points to the last participant who scored equal to `v[k-1]` or better.
   */
  while (r < n - 1 && v[r + 1] == v[k - 1]) // Continues as long as `r` is not the last element AND the next score is equal to the k-th score.
    r++; // Increments `r`.

  /**
   * Block Logic: Iterates backward from `r` to exclude any qualifying participants
   * who have a score of zero. Since only positive scores qualify.
   * Pre-condition: `r` points to the last potential qualifier.
   * Invariant: `r` points to the last participant who has a positive score and
   *            qualified based on the k-th score criteria.
   */
  while (r >= 0 && v[r] == 0) // Continues as long as `r` is a valid index AND the score at `r` is zero.
    r--; // Decrements `r`.

  // Prints `r + 1` because `r` is a 0-indexed count of qualifying participants.
  cout << r + 1 << endl; 

  return 0; // Indicates successful program termination.
}
