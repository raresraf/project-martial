/**
 * @file 158A.CPP
 * @brief This program determines the number of participants who advance to the next round of a competition.
 * Participants advance if their score is greater than or equal to the k-th place finisher's score
 * and also strictly positive.
 * Algorithm: Read all scores, then iterate through scores to compare against the k-th score.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing the scores.
 */
#include <bits/stdc++.h>

// Defines 'll' as a shorthand for 'long long int' to simplify code for large integer types.
#define ll long long int

// Defines 'mod' as a constant for modulo operations, often used in competitive programming.
#define mod 1000000007

using namespace std;

int main()

{

  int n, k;

  cin >> n >> k;

  vector<int> a(n + 1);

  /**
   * @brief Reads the scores of 'n' participants into a vector.
   * Pre-condition: 'n' represents the total number of participants.
   * Invariant: After the loop, 'a' contains all participant scores, with 'a[k]' holding the k-th participant's score.
   */
  for (int i = 1; i <= n; i++)

    cin >> a[i];

  int ans = 0;

  /**
   * @brief Iterates through the participant scores to count those who qualify.
   * Pre-condition: 'a' contains all scores, and 'a[k]' is the score of the k-th placed participant.
   * Invariant: 'ans' accumulates the count of qualifying participants.
   */
  for (int i = 1; i <= n; i++)

  {

    /**
     * @brief Checks if a participant's score meets the qualification criteria.
     * Logic: A participant qualifies if their score is greater than or equal to the k-th score,
     * AND their score is strictly positive.
     * Post-condition: If the conditions are met, 'ans' is incremented.
     */
    if (a[i] >= a[k] && a[i] > 0)

      ans++;
  }

  // Outputs the total number of participants who qualify for the next round.
  cout << ans;
}