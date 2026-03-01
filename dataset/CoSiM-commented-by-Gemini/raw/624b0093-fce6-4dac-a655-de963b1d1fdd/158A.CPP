/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem.
 *
 * @problem In a contest, 'n' participants are scored. Their scores are given in
 * non-increasing order. A participant advances to the next round if their score is
 * positive AND greater than or equal to the score of the k-th place finisher.
 * The task is to count the total number of participants who advance.
 *
 * @algorithm This code uses a direct, single-pass approach.
 * 1. It reads all 'n' scores into an array.
 * 2. It then iterates through the array from the first to the last participant.
 * 3. For each participant, it checks if their score `a[i]` satisfies two conditions:
 *    a) The score is positive (`a[i] > 0`).
 *    b) The score is greater than or equal to the k-th participant's score (`a[k-1]`).
 * 4. A counter is incremented for each participant who meets both conditions.
 * 5. The final count is printed.
 *
 * @time_complexity O(N), where N is the number of participants, due to the two
 * separate loops for reading and processing.
 * @space_complexity O(N) to store the scores of the 'n' participants.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  // Note: Using a variable-length array (VLA) `a[n]`, which is not standard C++.
  int a[n];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  // Use 'sum' as a counter for participants who advance.
  int sum = 0;
  // Iterate through all participants to check if they qualify.
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is >= the k-th score AND is positive.
    if (a[i] >= a[k - 1] && a[i] > 0) {
      sum++;
    }
  }
  cout << sum;
}
