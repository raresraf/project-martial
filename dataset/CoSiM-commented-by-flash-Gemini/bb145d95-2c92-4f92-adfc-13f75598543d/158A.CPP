/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces (Problem 158A).
 * @details This program calculates how many participants advance to the next round.
 * A participant advances if they have a positive score and it's greater than or
 * equal to the k-th participant's score. This solution leverages the fact that
 * the input array of scores is guaranteed to be in non-increasing order.
 *
 * @algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read the `n` scores, which are sorted in non-increasing order.
 * 3. Check the score of the k-th participant (`nums[k-1]`).
 * 4. **Case 1: `nums[k-1] > 0`**.
 *    All participants from 1 to `k` are guaranteed to advance. We then check for
 *    participants after `k` who have the same score as the k-th participant.
 *    The total count is `k` plus the number of these tied participants.
 * 5. **Case 2: `nums[k-1] <= 0`**.
 *    No one from the k-th position onwards can advance (as their scores are not positive).
 *    We must find the last participant before `k` who has a positive score. We do this by
 *    searching backwards from `k-2`. The total count is the rank of this last
 *    positively-scoring participant.
 * 6. Print the calculated count.
 *
 * @time_complexity O(N) in the worst case, for reading the input and potentially scanning for ties.
 * @space_complexity O(N) to store the scores.
 */
/**
 * I'm sorry I was an Idiot
 */
#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int nums[n];
  for (int i = 0; i < n; ++i) {
    cin >> nums[i];
  }

  // Case 1: The threshold score is positive.
  if (nums[k - 1] > 0) {
    int m = k;
    /**
     * @block_logic Since the k-th person scored > 0, at least k people advance.
     * This loop finds any additional participants after k-th rank who also
     * qualify by having the same score as the k-th person.
     */
    while (m < n && nums[m] == nums[k - 1]) {
      ++m;
    }
    cout << m << endl;
  } else { // Case 2: The threshold score is 0 or less.
    int m = k - 2;
    /**
     * @block_logic The k-th person's score is not positive, so no one from k onwards can qualify.
     * We need to find the last person before rank k who has a positive score.
     * This loop searches backwards from rank k-1.
     */
    while (m >= 0 && nums[m] <= 0) {
      --m;
    }
    // m+1 gives the count of participants with positive scores.
    cout << m + 1 << endl;
  }
  return 0;
}