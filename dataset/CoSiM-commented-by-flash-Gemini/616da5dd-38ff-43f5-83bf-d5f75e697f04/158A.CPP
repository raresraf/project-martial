/**
 * @616da5dd-38ff-43f5-83bf-d5f75e697f04/158A.CPP
 * @brief This program solves problem 158A, which involves determining the number
 * of participants who will advance to the next round of a competition.
 * Qualification criteria: a participant's score must be strictly positive
 * AND greater than or equal to the score of the participant at the k-th place.
 * Scores are assumed to be provided in non-increasing order.
 *
 * Algorithm: Linear scan to find the k-th score, followed by conditional counting
 * based on whether the k-th score is positive or zero/negative.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing participant scores.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, vector, algorithm.
using namespace std;     // Uses the standard namespace.

int main() {
  int n, k; // Declares 'n' for total participants and 'k' for the target rank.
  cin >> n >> k; // Reads 'n' and 'k' from standard input.
  int a[n]; // Declares an array 'a' of size 'n' to store participant scores.
  
  /**
   * Block Logic: Reads 'n' participant scores into the array 'a'.
   * Invariant: After this loop, 'a' contains all participant scores in non-increasing order.
   */
  for (int i = 0; i < n; i++)
    cin >> a[i]; // Reads each score.
  
  // Inline: Stores the score of the k-th placed participant (0-indexed k-1).
  int x = a[k - 1]; 
  
  /**
   * Block Logic: Processes two main cases based on the k-th participant's score.
   * Case 1: The k-th participant has a positive score (x > 0).
   * Case 2: The k-th participant has a non-positive score (x <= 0).
   * Invariant: The correct number of qualifying participants is printed.
   */
  if (x > 0) { // Case 1: The k-th participant's score is positive.
    int i = k; // Starts checking from the (k+1)-th participant (index k).
    int cnt = 0; // Counts participants tied with the k-th participant beyond rank k.
    /**
     * Block Logic: Counts participants from index 'k' onwards who have the exact same score as 'x'.
     * Invariant: 'cnt' increases for each tied participant encountered.
     * Pre-condition: Scores are in non-increasing order.
     */
    while (a[i] == x && i < n) { // Continues as long as scores match 'x' and within bounds.
      i++; // Increments index.
      cnt++; // Increments count of tied participants.
    }
    // Prints the total number of qualifiers: initial 'k' (top k participants) + 'cnt' (those tied with the k-th).
    cout << k + cnt << "\n"; 
  } else { // Case 2: The k-th participant's score is non-positive (0 or negative).
    int i = 0; // Starts checking from the first participant (index 0).
    int cnt = 0; // Counts participants with strictly positive scores.
    /**
     * Block Logic: Counts participants whose scores are strictly greater than 0, up to (but not including) index 'k'.
     * Invariant: 'cnt' increases for each participant with a positive score.
     * Pre-condition: Scores are in non-increasing order.
     */
    while (a[i] != 0 && i < k) { // Continues as long as score is not 0 and within first 'k' positions.
      i++; // Increments index.
      cnt++; // Increments count of positive scores.
    }
    // Prints the count of participants with strictly positive scores (who are implicitly ahead of any non-positive k-th score).
    cout << cnt << "\n"; 
  }
  return 0; // Indicates successful program termination.
}