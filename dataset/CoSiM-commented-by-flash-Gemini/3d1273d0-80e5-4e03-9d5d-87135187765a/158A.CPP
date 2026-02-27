/**
 * @file 158A.CPP
 * @brief Solution for competitive programming problem: "Next Round".
 *
 * @details This program calculates the number of participants who will advance
 *          to the next round of a competition. Participants advance if their
 *          score is greater than or equal to the k-th place finisher's score,
 *          provided their score is positive.
 *          Algorithm: Iterative processing of scores with conditional checking.
 *          Time Complexity: O(N), where N is the number of participants, as each
 *                           participant's score is processed once.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Declare variables:
  // 'n': total number of participants
  // 'k': the k-th place participant (whose score determines the cutoff)
  // 'a': current participant's score
  // 'b': score of the k-th place participant
  // 's': counts participants who advance based on the k-th place score (initially 0, later set to k if k-th score is positive)
  // 'h': counts participants with positive scores encountered before the k-th participant
  int n, k;
  int a, b, s = 0, h = 0;
  // Read the total number of participants (n) and the k-th place (k).
  cin >> n >> k;

  // Block Logic: Iterate through each participant's score to determine advancement.
  // Invariant: 'i' represents the current participant's 0-based index.
  for (int i = 0; i < n; i++) {

    // Read the current participant's score.
    cin >> a;
    // Precondition: If the current participant has a positive score, increment 'h'.
    // 'h' is used as a temporary count for positive scores before the k-th position is fully evaluated.
    if (a != 0) {
      h++;
    }
    
    // Block Logic: Determine the score of the k-th place participant and establish initial advancing count.
    // This block executes when the k-th participant (0-indexed k-1) is processed.
    if (i == k - 1) {
      b = a; // Store the k-th participant's score.
      // Precondition: If the k-th participant's score is zero, no one with zero score can advance.
      if (b == 0) {
        cout << h; // Output the count of positive scores encountered so far.
        return 0;  // Terminate, as subsequent zero scores won't change the count.
      } else {
        // If k-th participant has a positive score, all participants up to k-th place advance (k participants).
        s = k;
      }
    }
    
    // Block Logic: For participants after the k-th position, check if they also advance.
    // Precondition: Only consider participants beyond the k-th position (i >= k).
    if ((i >= k) && (a == b)) {
      s++; // If their score equals the k-th score, they also advance.
    } 
    // If a participant after k-th place has a score different from k-th place,
    // and thus cannot advance based on the criteria, break early.
    // This assumes scores are non-increasing after the k-th participant which is common in such problems.
    else if (i >= k) {
      break; // Stop counting, as subsequent scores would also be less than 'b' or equal to zero.
    }
  }
  // Output the final count of participants who advance.
  cout << s;

  return 0; // Indicate successful program termination.
}