/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details The problem is to count how many participants advance to the next
 * round. A participant advances if they have a positive score and their score
 * is greater than or equal to the score of the k-th place finisher. The input
 * scores are sorted in non-increasing order.
 *
 * @algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read all `n` scores into an array. A quick check is made for the edge
 *    case where the top score is 0.
 * 3. The score of the k-th participant (at index `k-1`) is used as the threshold.
 * 4. A second loop iterates through the scores to count the qualifiers.
 * 5. For each participant, check if their score is positive and also greater than
 *    or equal to the threshold score.
 * 6. If the condition is met, increment a counter.
 * 7. If a participant's score is less than the threshold, break the loop, as no
 *    one after them can qualify due to the non-increasing order of scores.
 * 8. Print the final count.
 *
 * @complexity
 * Time Complexity: O(n), for reading the scores and then iterating through them.
 * Space Complexity: O(n), to store the scores in an array.
 */
#include <iostream>
using namespace std;
int main() {
  int n, k, i;
  int a; // Counter for the number of participants who advance.
  cin >> n >> k;
  
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  int arr[n];
  a = 0;

  /**
   * @block
   * @brief Read all participant scores into an array.
   */
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    // Edge case: If the top score is 0, no one can have a positive score,
    // so no one advances.
    if (arr[i] == 0 && i == 0) {
      cout << 0;
      return 0;
    }
  }

  /**
   * @block
   * @brief Count the number of participants who qualify.
   */
  for (i = 0; i < n; i++) {
    // Since scores are non-increasing, we can stop as soon as we find
    // a score lower than the threshold score.
    if (arr[i] < arr[k - 1])
      break;
    // A participant advances if their score is >= the k-th score AND is positive.
    else if (arr[i] >= arr[k - 1] && arr[i] != 0)
      a++;
  }
  
  cout << a;
  return 0;
}
