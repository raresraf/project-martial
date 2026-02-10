/**
 * @file 158A.CPP
 * @brief Solution for competitive programming problem "Next Round".
 * @details This program determines how many participants will advance to the
 * next round of a competition. The rule for advancing is that a participant's
 * score must be positive and greater than or equal to the score of the
 * k-th place finisher.
 *
 * Algorithm:
 * 1. Read the total number of participants (n) and the rank of the cutoff
 *    score (k).
 * 2. Read all n participant scores into an array. The scores are given in
 *    non-increasing order.
 * 3. Identify the cutoff score, which is the score of the (k-1)-th participant
 *    in the array.
 * 4. Iterate through the array of scores.
 * 5. For each participant, check if their score is positive AND greater than
 *    or equal to the cutoff score.
 * 6. Count the number of participants who meet both criteria.
 * 7. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of participants. This is
 * dominated by the two loops for reading and checking scores.
 * Space Complexity: O(N) to store the scores in an array.
 */
#include <bits/stdc++.h>
#define ll long long
using namespace std;
int main() {
  ll int n, k, sum = 0;
  // Read the number of participants (n) and the rank (k) for the cutoff score.
  cin >> n >> k;
  ll int arr[n];
  // Read the scores of all participants.
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  // Invariant: 'sum' holds the count of participants who qualify for the next round.
  // The cutoff score is arr[k-1].
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is positive and at least as high
    // as the k-th place participant's score.
    if (arr[i] > 0 && arr[i] >= arr[k - 1]) {
      ++sum;
    }
  }
  // Output the total number of advancing participants.
  cout << sum;
  return 0;
}
