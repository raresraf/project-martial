/**
 * @file 158A.CPP
 * @brief Solution for competitive programming problem: "Next Round".
 *
 * @details This program determines the number of participants who will advance
 *          to the next round of a competition. Participants advance if their
 *          score is greater than or equal to the k-th place finisher's score,
 *          provided their score is strictly positive. This implementation uses
 *          a two-pass approach: first reading all scores into an array, then
 *          iterating to count the eligible participants.
 *          Algorithm: Two-pass iterative processing of scores with conditional checking.
 *          Time Complexity: O(N), where N is the number of participants, as scores
 *                           are read and processed in two separate linear passes.
 *          Domain: Competitive Programming.
 */
#include <iostream>
using namespace std;
// Declare global variables:
// 'n': Total number of participants.
// 'k': The 1-based index of the k-th place participant whose score sets the advancement threshold.
// 'a': Temporary variable to store the current participant's score during input.
// 'b': Counter for the number of participants who advance to the next round, initialized to 0.
// 'mas[10000]': An array to store the scores of up to 10000 participants.
int n, k, a, b = 0, mas[10000];
int main() {
  // Read the total number of participants (n) and the k-th place (k).
  cin >> n >> k;
  
  // Block Logic: Read all participant scores into the 'mas' array.
  // Invariant: After this loop, 'mas' contains scores for participants from index 1 to 'n'.
  for (int i = 1; i <= n; i++) {
    cin >> a;     // Read the score for the current participant.
    mas[i] = a;   // Store the score in the array.
  }
  
  // Block Logic: Iterate through the stored scores to count participants who advance.
  // Invariant: 'b' accumulates the count of eligible participants.
  for (int j = 1; j <= n; j++) {
    // Precondition: A participant advances if their score is greater than or equal to
    //               the k-th place score AND their score is positive.
    if (mas[j] >= mas[k] && mas[j] > 0) {
      b++; // Increment the counter for advancing participants.
    }
  }
  // Output the total number of participants who advance.
  cout << b;
}