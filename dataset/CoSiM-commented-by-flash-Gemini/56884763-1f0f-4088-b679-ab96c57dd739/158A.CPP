/**
 * @56884763-1f0f-4088-b679-ab96c57dd739/158A.CPP
 * @brief This program calculates the number of participants who will advance to the next round of a competition.
 * Advancing criteria: a participant's score must be greater than or equal to the k-th place participant's score,
 * and their score must be strictly positive. It is assumed scores are provided in non-increasing order.
 *
 * Algorithm: Linear scan to determine a qualifying threshold and then another linear scan to count participants meeting the criteria.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing participant scores.
 */

#include <algorithm> // Generally included for utility functions, though not directly used for std::sort in this snippet.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <map>       // Not used in this particular snippet.
#include <stdio.h>   // For standard I/O, often included in competitive programming, but iostream is preferred for C++.
#include <vector>    // Not used in this particular snippet.

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int n, k; // Declares 'n' for the total number of participants and 'k' for the qualifying rank.
  cin >> n >> k; // Reads 'n' and 'k' from standard input.
  
  // Declares an array 'a' of size n+2 (to avoid out-of-bounds issues, common competitive programming practice).
  // This array will store the scores of the participants.
  int a[n + 2];
  /**
   * Block Logic: Reads 'n' participant scores into the array 'a'.
   * Invariant: After this loop, 'a' contains all participant scores.
   * Pre-condition: Scores are typically provided in non-increasing order.
   */
  for (int i = 0; i < n; i++)
    cin >> a[i]; // Reads each score.
  
  // Determines the qualifying score. This is the score of the participant at k-th place (0-indexed k-1).
  int mark = a[k - 1]; 
  int count1 = 0; // Initializes a counter for participants who will advance.
  
  /**
   * Block Logic: Checks if the top participant's score is positive. If not, no one advances.
   * Pre-condition: 'a[0]' holds the highest score.
   * Invariant: If 'a[0]' is non-positive, 'count1' remains 0.
   */
  if (a[0] > 0) {
    /**
     * Block Logic: Determines the qualification criteria based on the k-th score.
     * Invariant: 'count1' is updated based on scores being >= 'mark' and > 0.
     */
    if (mark != 0) { // If the k-th participant's score is not zero, all qualifiers must match or exceed it.
      /**
       * Block Logic: Counts participants whose scores are greater than or equal to 'mark'.
       * Invariant: 'count1' includes all participants encountered so far whose score meets the criteria.
       * Pre-condition: Array 'a' is in non-increasing order.
       */
      for (int i = 0; i < n; i++) {
        // Inline: Check if the current participant's score is greater than or equal to the qualifying mark.
        if (a[i] >= mark)
          count1++; // Increment if score qualifies.
        else
          break; // Optimization: If scores are non-increasing, no further participants will qualify.
      }
    } else { // If the k-th participant's score is zero, all qualifiers must have a strictly positive score.
      /**
       * Block Logic: Counts participants whose scores are strictly greater than 0.
       * Invariant: 'count1' includes all participants encountered so far whose score is positive.
       * Pre-condition: Array 'a' is in non-increasing order.
       */
      for (int i = 0; i < n; i++) {
        // Inline: Check if the current participant's score is strictly positive.
        if (a[i] > 0)
          count1++; // Increment if score qualifies.
        else
          break; // Optimization: If scores are non-increasing, no further participants will qualify.
      }
    }
    cout << count1; // Prints the total number of advancing participants.
  } else { // If the highest score (a[0]) is 0 or negative, no one can advance.
    cout << "0"; // Prints 0 as no participants qualify.
  }
  return (0); // Indicates successful program termination.
}