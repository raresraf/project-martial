/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A - Next Round.
 *
 * This program determines the number of participants who will advance to the next
 * round of a competition. A participant advances if their score is greater than
 * or equal to the score of the k-th placed participant, and their score is
 * strictly positive.
 *
 * Algorithm: Direct comparison. After reading all scores, the program identifies
 * the k-th participant's score and then iterates through all participants,
 * counting those who meet the advancement criteria.
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N) for storing the participant scores.
 */

#include <bits/stdc++.h>
using namespace std;

// Declare a global array 'a' to store participant scores. Max 100 participants.
int a[100];

int main() {
  /**
   * @brief Main function to execute the Next Round problem logic.
   *
   * Reads the total number of participants 'n' and the k-th position 'k'.
   * Then reads all 'n' participant scores into an array.
   * Finally, it iterates through the scores to count and print how many
   * participants satisfy the advancement criteria.
   */
  // Declare 'n' for total participants and 'k' for the k-th place.
  int n, k;
  // Read the total number of participants and the k-th position.
  cin >> n >> k;
  // Block Logic: Read participant scores.
  // Invariant: 'i' iterates from 0 to n-1, populating the scores array.
  for (int i = 0; i < n; i++) {
    cin >> a[i]; // Store each participant's score.
  }
  // Initialize 'ans' to count the number of advancing participants.
  int ans = 0;
  // Block Logic: Count advancing participants.
  // Invariant: 'i' iterates through all participants.
  // Pre-condition: 'a[k-1]' holds the score of the k-th placed participant.
  for (int i = 0; i < n; i++)
    // A participant advances if their score is greater than or equal to the k-th
    // score AND their score is strictly positive.
    if (a[i] >= a[k - 1] && a[i] > 0)
      ans++; // Increment the count of advancing participants.
  // Output the total number of advancing participants.
  cout << ans;
}