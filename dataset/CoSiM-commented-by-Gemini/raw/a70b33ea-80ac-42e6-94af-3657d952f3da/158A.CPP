/**
 * @file 158A.CPP
 * @brief Solution for CodeForces Problem 158A - "Next Round".
 * @author Nikhil Nagrale (nikhilnagrale2)
 *
 * @details
 * The problem is to count the number of participants who will advance to the next round.
 * A participant advances if they have a positive score and their score is greater than
 * or equal to the score of the k-th place finisher. The scores are given in non-increasing order.
 *
 * @algorithm
 * 1. Read the total number of participants, `n`, and the rank `k`.
 * 2. Read all `n` scores into a vector. Since the input is sorted, the scores are stored
 *    in non-increasing order.
 * 3. Determine the score of the k-th participant. This is the score at index `k-1`.
 *    This score is the threshold for advancing.
 * 4. Initialize a counter for advancing participants to 0.
 * 5. Iterate through all the scores:
 *    - If a participant's score is greater than or equal to the k-th place score AND
 *      their score is positive, increment the counter.
 * 6. Print the final count.
 *
 * @time_complexity O(N), where N is the number of participants. The process involves two
 *                  separate loops through the participants' data (one for reading, one for counting).
 * @space_complexity O(N) to store the scores in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  vector<int> a; // Vector to store the scores of the participants.
  
  /**
   * @brief Read all participant scores into the vector.
   */
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    a.push_back(temp);
  }

  // Get the score of the k-th participant (at index k-1). This is the cutoff score.
  int cutoff_score = a[k - 1];
  int count = 0; // Counter for participants who advance.

  /**
   * @brief Count the number of participants who advance.
   * A participant advances if their score is >= the cutoff score and is positive.
   */
  for (int x : a) {
    if (x >= cutoff_score && x > 0)
      count++;
  }

  cout << count << endl;
  return 0;
}
