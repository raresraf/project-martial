/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces (158A).
 * @details The problem requires counting the number of participants who will advance
 * to the next round of a contest. A participant advances if their score is
 * equal to or greater than the k-th place finisher's score, provided their
 * score is positive.
 *
 * @algorithm
 * 1. Read the total number of participants (`k`) and the 1-based rank (`n`)
 *    that determines the cutoff score.
 * 2. Read all `k` participant scores into an array. The scores are given in
 *    non-increasing order.
 * 3. Determine the cutoff score (`s`) by accessing the score of the `n`-th
 *    place finisher, which is at index `n-1` in the array.
 * 4. Iterate through the array of scores again.
 * 5. For each score, check if it is greater than or equal to the cutoff score `s`
 *    and also if it is positive.
 * 6. Increment a counter for each participant who meets both conditions.
 * 7. Print the final counter.
 *
 * @complexity
 * Time Complexity: O(K), where K is the number of participants. The solution involves
 * two separate linear passes through the scores array.
 * Space Complexity: O(K), due to storing all participant scores in an array.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @variable k: The total number of participants.
   * @variable n: The 1-based rank determining the cutoff score.
   * @variable s: The cutoff score, i.e., the score of the n-th place finisher.
   * @variable in: An array to store the scores of all participants.
   * @variable i: A loop counter.
   * @variable player: A counter for the number of participants who advance.
   */
  int k, n, s, in[100], i, player = 0;
  
  // Read the number of participants and the cutoff rank.
  cin >> k >> n;
  
  // Read all participant scores into the array.
  for (i = 0; i < k; i++) {
    cin >> in[i];
  }
  
  // Determine the cutoff score. Since scores are in non-increasing order,
  // the score at index n-1 is the score of the n-th place finisher.
  s = in[n - 1];
  
  /**
   * @block
   * @brief This loop counts the number of participants who advance.
   * @invariant `player` holds the count of advancing participants found so far.
   */
  for (i = 0; i < k; i++) {
    // Pre-condition: A player advances if their score is >= the cutoff score
    // AND their score is positive.
    if (in[i] >= s && in[i] != 0) {
      player++;
    }
  }
  
  // Output the final count of advancing players.
  cout << player;
  return 0;
}
