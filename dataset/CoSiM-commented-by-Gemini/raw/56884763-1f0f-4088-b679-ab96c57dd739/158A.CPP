/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details The program counts the number of participants who advance to the next
 * round. A participant advances if they have a positive score AND their score is
 * greater than or equal to the k-th place participant's score. The scores are
 * given in non-increasing order.
 *
 * @algorithm
 * This solution handles the problem by splitting it into several cases.
 * 1. Read all `n` scores into an array.
 * 2. Get the score of the k-th participant, `mark`.
 * 3. First, check if the top score is positive. If not, no one can advance, so the answer is 0.
 * 4. If the top score is positive, check if the threshold score `mark` is positive.
 *    a. If `mark > 0`: Count all participants whose score is `>= mark`.
 *    b. If `mark == 0`: The condition `score >= 0` is met, so we only need to count
 *       participants with a positive score (`score > 0`).
 * 5. In both cases, the loop can break early as soon as a participant fails to
 *    qualify, since the scores are sorted non-increasingly.
 *
 * @note The logic can be simplified by combining the conditions into a single
 * check: `score >= mark && score > 0`.
 *
 * @complexity
 * Time Complexity: O(n), for reading the input and a single pass to count.
 * Space Complexity: O(n) to store the scores.
 */
#include <algorithm>
#include <iostream>
#include <map>
#include <stdio.h>
#include <vector>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  int a[n + 2];
  for (int i = 0; i < n; i++)
    cin >> a[i];

  int mark = a[k - 1]; // The threshold score is the score of the k-th person.
  int count1 = 0;      // The final count of advancing participants.

  // First check: if the top score isn't positive, no one can advance.
  if (a[0] > 0) {
    // Case 1: The threshold score is positive.
    if (mark != 0)
      /**
       * @block
       * @brief Count players with scores >= a positive threshold.
       */
      for (int i = 0; i < n; i++) {
        if (a[i] >= mark)
          count1++;
        else
          break; // Optimization: scores are sorted, so no one else can qualify.
      }
    // Case 2: The threshold score is 0.
    else {
      /**
       * @block
       * @brief Count players with scores > 0, since the threshold is 0.
       */
      for (int i = 0; i < n; i++) {
        if (a[i] > 0)
          count1++;
        else
          break; // Optimization: scores are sorted.
      }
    }
    cout << count1;
  } else {
    // If the top score is not positive, the answer is 0.
    cout << "0";
  }
  return (0);
}
