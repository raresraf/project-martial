/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A, "Next Round".
 * @details The program determines how many participants advance to the next
 * round based on their scores. A participant advances if they have a positive
 * score that is greater than or equal to the score of the k-th ranked
 * participant.
 *
 * Algorithm:
 * The code implements the logic by breaking it down into several cases after
 * reading all `n` scores into an array.
 * 1. It first checks if the top score is zero or less. If so, no one
 *    advances, and it prints 0.
 * 2. It then checks if the k-th participant's score is zero. If so, it
 *    iterates only through the participants ranked higher than k (from 1 to k-1)
 *    and counts how many of them have a positive score.
 * 3. In the main case (the k-th score is positive), it assumes all top `k`
 *    participants advance. It then iterates from the (k+1)-th participant
 *    onward, counting any additional participants who have the same score as
 *    the k-th participant. The final answer is `k` plus this count.
 *
 * Note: The code uses a Variable Length Array (VLA) `arr[a]`, which is a
 * non-standard C++ extension.
 *
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(N) to store the scores in an array.
 */
#include <iostream>
using namespace std;

typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;

int main(int argc, char const *argv[]) {
  ios_base::sync_with_stdio(false);
  lli a, b, count = 0;
  cin >> a >> b;
  lli arr[a]; // Using a non-standard Variable Length Array (VLA).
  
  // Read all scores into the array first.
  for (int i = 0; i < a; ++i) {
    cin >> arr[i];
  }

  // Case 1: If the top score is not positive, no one advances.
  if (arr[0] <= 0) {
    cout << 0 << endl;
    return 0;
  }

  /**
   * @brief Case 2: The k-th participant's score is 0.
   * Since scores are non-increasing and must be positive to advance, only
   * participants ranked higher than k need to be checked.
   */
  if (arr[b - 1] == 0) {
    for (int i = 0; i < b - 1; ++i) {
      if (arr[i] > 0) {
        count++;
      }
    }
    cout << count << endl;
    return 0;
  }

  /**
   * @brief Main case: The k-th participant has a positive score.
   * All top `b` (or `k`) participants automatically qualify. This loop
   * counts any additional participants after the k-th rank who tie with
   * the k-th score.
   */
  for (int i = b; i < a; ++i) {
    if (arr[i] == arr[b - 1] && arr[i] > 0) {
      count++;
    }
  }

  // The total is the top `b` participants plus any who tied.
  cout << b + count << endl;
  return 0;
}
