/**
 * @file 158A.CPP
 * @brief Determines the number of participants advancing to the next round.
 * @details This program solves the "Next Round" problem from Codeforces.
 * The rule for advancing is that a participant's score must be positive and
 * greater than or equal to the score of the k-th place finisher. The program
 * counts how many participants meet this criterion.
 *
 * @see https://codeforces.com/problemset/problem/158/A
 *
 * Algorithm:
 * 1. Read the number of participants (n) and the rank of the cutoff score (k).
 * 2. Read all n participant scores into a vector. The scores are provided in
 *    non-increasing order.
 * 3. Determine the cutoff score by accessing the score of the k-th participant
 *    (at index k-1).
 * 4. Iterate through the scores from the beginning.
 * 5. Count every participant whose score is positive and greater than or equal to
 *    the cutoff score.
 * 6. Print the total count.
 *
 * Time Complexity: O(N), where N is the number of participants, due to reading
 * and iterating through the scores once.
 * Space Complexity: O(N) to store the scores in a vector.
 */
#include <iostream>
#include <vector>
using namespace std;

int main() {
  // n: total number of participants.
  // k: the 1-based rank of the participant whose score is the cutoff.
  int n, k;
  cin >> n >> k;

  // xs: a vector to store the scores of all participants.
  vector<int> xs(n);
  for (int i = 0; i < n; ++i)
    cin >> xs[i];

  // The cutoff score is the score of the k-th participant (at index k-1).
  int val = xs[k - 1];
  
  // total: the number of participants who will advance.
  int total = 0;

  /**
   * @brief Counts the number of advancing participants.
   *
   * Pre-condition: 'xs' contains all scores in non-increasing order, 'val' is the cutoff score.
   * Logic: A participant advances if their score is positive AND >= the cutoff score.
   * The loop short-circuits if a score is zero or less than the cutoff, since the
   * array is sorted non-increasingly.
   */
  for (int i = 0; i < n && xs[i] > 0 && xs[i] >= val; ++i)
    ++total;
    
  cout << total << endl;
  return 0;
}
