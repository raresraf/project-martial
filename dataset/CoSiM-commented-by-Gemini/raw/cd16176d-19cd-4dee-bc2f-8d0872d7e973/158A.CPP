/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem (Codeforces 158A).
 * @details The problem is to count the number of participants who will advance
 * to the next round of a competition. A participant advances if their score is
 * positive and is greater than or equal to the score of the k-th place finisher.
 *
 * @section algorithm Algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into a vector. The scores are assumed to be sorted
 *    in non-increasing order as per the problem statement.
 * 3. Store the score of the k-th participant.
 * 4. Iterate through all participants' scores.
 * 5. Increment a counter for each participant whose score is positive AND
 *    greater than or equal to the k-th participant's score.
 * 6. Print the final counter.
 *
 * @section complexity Complexity
 * - Time Complexity: O(N), for reading the inputs and then iterating to count.
 * - Space Complexity: O(N), to store the scores of the participants.
 */
#include <bits/stdc++.h>

#define ll long long int

#define mod 1000000007

using namespace std;

int main()

{

  int n, k;

  cin >> n >> k;

  // The code uses 1-based indexing for the scores vector.
  vector<int> a(n + 1);

  // Read all n scores into the vector.
  for (int i = 1; i <= n; i++)

    cin >> a[i];

  int ans = 0; // Counter for participants who will advance.

  /**
   * @brief Main logic to count advancing participants.
   * Iterates through all participants to check if they meet the criteria.
   */
  for (int i = 1; i <= n; i++)

  {

    // A participant advances if their score is positive and >= the k-th score.
    if (a[i] >= a[k] && a[i] > 0)

      ans++;
  }

  cout << ans;
}
