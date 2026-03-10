/**
 * @file 158A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Next Round").
 * This program determines the number of participants who will advance to the next round.
 * The rule is: a participant advances if their score is positive and is greater
 * than or equal to the score of the k-th place participant.
 *
 * @algorithm Linear Scan. The problem guarantees that the input scores are provided
 * in non-increasing order, so no sorting is needed. The algorithm finds the
 * score of the k-th participant and then iterates through the list to count
 * how many participants meet the advancement criteria.
 *
 * @time_complexity O(N), where N is the number of participants, due to the two
 * linear passes for reading input and counting.
 * @space_complexity O(N) to store the scores of the N participants.
 */
#include <algorithm>
#include <iostream>
#include <vector>

#define ll long long

using namespace std;

int main() {
  ll n, k;
  cin >> n >> k;
  vector<ll> a(n);
  // Read the scores of all n participants.
  for (int i = 0; i < n; i++)
    cin >> a[i];

  // The input array 'a' is guaranteed to be sorted in non-increasing order.
  // sort(a.begin(),a.end());

  ll count = 0; // Counter for participants who will advance.

  /**
   * @block
   * @brief Counts the number of advancing participants.
   *
   * Invariant: The score to qualify is `a[k-1]`.
   * The loop checks each participant. A participant advances if their score `a[i]`
   * is at least the score of the k-th place finisher (`a[k-1]`) and is also
   * strictly positive.
   */
  for (int i = 0; i < n; i++)
    if (a[i] >= a[k - 1] && a[i] > 0)
      count++;
      
  cout << count << "\n";
  return 0;
}