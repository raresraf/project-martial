/**
 * @file 158A.CPP
 * @brief Determines the number of participants advancing to the next round.
 *
 * This program calculates how many participants will advance in a contest.
 * The rule is that a participant advances if they have a positive score and
 * their score is equal to or greater than the score of the k-th place finisher.
 *
 * @b Algorithm:
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read the scores of all `n` participants into an array.
 * 3. Identify the score of the k-th participant.
 * 4. Iterate through all participants again, and count how many have a score
 *    that is both positive and greater than or equal to the k-th score.
 *
 * @b Time_Complexity: O(N), where N is the number of participants, due to the
 *    loops for reading and checking scores.
 * @b Space_Complexity: O(N) to store the scores in an array.
 */
#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;
int main() {

  /**
   * @brief This is a common competitive programming setup for redirecting
   * I/O to files when running locally (not on an online judge).
   */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, k, count = 0;
  cin >> n >> k;
  int arr[n];
  
  // Note: This code uses 1-based indexing for the array.
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }
  
  /**
   * @brief This loop counts the number of participants who advance.
   * A participant advances if their score is positive and >= the k-th score.
   */
  for (int j = 1; j <= n; j++) {
    if (arr[j] >= arr[k] && arr[j] > 0) {
      count++;
    }
  }
  cout << count;
}