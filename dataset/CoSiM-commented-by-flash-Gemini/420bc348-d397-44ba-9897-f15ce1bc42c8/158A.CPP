/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem (likely Codeforces 158A - "Next Round").
 *
 * This program determines the number of participants who will advance to the next round
 * of a competition. Participants advance if their score is greater than or equal to
 * the k-th place finisher's score AND their score is strictly positive.
 *
 * Algorithm: Linear scan.
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N) to store participant scores.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * This function reads the number of participants and the k-th position,
 * then processes the scores to find the number of advancing participants.
 */
int main() {
  int n, k, i;
  // Pre-condition: Read n (total participants) and k (k-th place index).
  // Invariant: n and k are positive integers.
  cin >> n >> k;
  // Declare an array to store scores of 'n' participants.
  int a[n];
  // Block Logic: Read 'n' scores from input and store them in the array 'a'.
  // Invariant: 'i' iterates from 0 to n-1.
  for (i = 0; i < n; i++) {
    cin >> a[i];
  }
  // Get the score of the participant at the k-th position (0-indexed: k-1).
  int val = a[k - 1];
  i = 0; // Reset counter for advancing participants.
  // Block Logic: Count participants who meet the advancement criteria.
  // Pre-condition: Iterate through participants from the beginning.
  // Invariant: 'a[i]' is the current participant's score.
  // Advancement Criteria: score >= val (k-th place score) AND score > 0.
  while (a[i] >= val && a[i] > 0) {
    i++; // Increment count of advancing participants.
    // Check if the end of the array is reached to prevent out-of-bounds access.
    if (i == n) {
      break;
    }
  }
  // Output the total number of participants who will advance.
  cout << i;
}
