/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" contest problem.
 *
 * This program calculates the number of contestants who will advance to the
 * next round. A contestant advances if their score is positive and also
 * greater than or equal to the score of the k-th place finisher. The input
 * scores are provided in non-increasing order.
 *
 * Algorithm:
 * 1. Read the number of contestants (n) and the rank 'k'.
 * 2. Store all n scores in an array.
 * 3. The score of the k-th contestant (at index k-1) serves as the cutoff threshold.
 * 4. Iterate through all scores, counting each one that is positive and meets
 *    or exceeds the cutoff score.
 * 5. Print the total count.
 *
 * Time Complexity: O(N), for reading and then iterating through the N scores.
 * Space Complexity: O(N), to store the array of scores.
 */
#include <bits/stdc++.h>
using namespace std;
int a[100]; // Global array to store scores.
int main() {
  int n, k;
  cin >> n >> k;

  // Read all n scores into the global array.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  
  int ans = 0; // Counter for participants who advance.

  /**
   * Block Logic: Iterate through all participants to check if they advance.
   */
  for (int i = 0; i < n; i++)
    // A participant advances if their score is positive AND >= the k-th score.
    if (a[i] >= a[k - 1] && a[i] > 0)
      ans++;
      
  cout << ans;
}
