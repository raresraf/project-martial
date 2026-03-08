/**
 * @file 158A.CPP
 * @brief Solution to a contest advancement problem (e.g., Codeforces 158A "Next Round").
 * @details This program determines how many participants will advance to the next round
 * based on a cutoff score. The rule is that a participant advances if their score is
 * equal to or greater than the score of the participant who finished in k-th place,
 * as long as the participant's score is positive.
 *
 * @algorithm Linear Scan
 * @complexity
 * - Time: O(n), where 'n' is the number of participants, due to two separate loops
 *   that each iterate up to 'n' times (one for input, one for counting).
 * - Space: O(n) to store the scores of the participants in a vector.
 */
#include <iostream>
#include <vector>

using namespace std;

int main() {
  // n: total number of participants.
  // p: the 1-based rank of the participant whose score is the cutoff threshold.
  int n, p;
  cin >> n >> p;

  // Vector to store the scores of all n participants.
  vector<int> nums(n);
  /**
   * @block Input Reading
   * @logic Reads the scores of all participants into the vector.
   * The scores are assumed to be provided in non-increasing order.
   */
  for (int i = 0; i < n; i++) {
    cin >> nums[i];
  }

  // Counter for the number of participants who will advance.
  int count = 0;
  
  /**
   * @block Advancement Counting
   * @logic Iterates through each participant's score to check if they qualify.
   * @invariant `count` holds the number of advancing participants from the first `i-1` checked.
   */
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is >= the k-th person's score
    // (at index p-1) and their score is positive.
    if (nums[i] >= nums[p - 1] && nums[i] > 0)
      count++;
  }

  // Output the total number of advancing participants.
  cout << count << endl;
  return 0;
}
