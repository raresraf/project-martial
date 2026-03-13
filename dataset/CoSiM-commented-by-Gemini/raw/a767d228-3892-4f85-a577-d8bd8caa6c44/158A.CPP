/**
 * @file 158A.CPP
 * @brief Solution for CodeForces Problem 158A - "Next Round".
 * @author Sohanuzzaman Soad
 *
 * @details
 * The problem is to count the number of participants who will advance to the next round.
 * A participant advances if they have a positive score and their score is greater than
 * or equal to the score of the k-th place finisher. The scores are given in non-increasing order.
 *
 * @algorithm
 * 1. Read the total number of participants, `n`, and the rank `k`.
 * 2. Read all `n` scores into an array.
 * 3. Determine the score of the k-th participant (at index `k-1`). This is the threshold.
 * 4. Initialize a counter for advancing participants to 0.
 * 5. Iterate through all the scores:
 *    - If a participant's score is >= the threshold AND is positive, increment the counter.
 * 6. Print the final count.
 *
 * @time_complexity O(N), where N is the number of participants.
 * @space_complexity O(N) to store the scores in an array.
 *
 * @note The use of a Variable Length Array (`int a[n]`) is a non-standard C++ extension,
 *       commonly supported by compilers like GCC and Clang. A `std::vector` would be
 *       standard and safer.
 */
#include <iostream>

using namespace std;

int main() {
  int n; // Total number of participants.
  int k; // The rank of the participant whose score is the cutoff.
  int c = 0; // Counter for participants who advance.
  
  cin >> n >> k;
  int a[n]; // Array to store the scores.
  
  /**
   * @brief Read all participant scores into the array.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }

  // Get the score of the k-th participant. This is the cutoff score.
  int val = a[k - 1];

  /**
   * @brief Count the number of participants who advance.
   * A participant advances if their score is >= the cutoff score and is positive.
   */
  for (int i = 0; i < n; i++) {
    if (a[i] >= val && a[i] > 0) {
      c++;
    }
  }
  
  cout << c << endl;
  return 0;
}
