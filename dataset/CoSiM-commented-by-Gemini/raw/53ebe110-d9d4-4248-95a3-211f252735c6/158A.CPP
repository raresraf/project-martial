/**
 * @file 158A.CPP
 * @brief An attempt to solve the "Next Round" problem from Codeforces.
 * @details The problem is to count how many participants advance. A participant
 * advances if their score is positive and greater than or equal to the score of
 * the k-th ranked participant.
 *
 * @algorithm (as implemented)
 * This implementation attempts to solve the problem in a single loop.
 * 1. Read n (number of participants) and k (the rank).
 * 2. Dynamically allocate an array for the scores.
 * 3. Loop n times, reading each score.
 * 4. **Logical Flaw**: Inside the same loop, it immediately compares the just-read
 *    score `arr[i]` with the score at `arr[k-1]`. However, if `i < k-1`, the value
 *    at `arr[k-1]` has not been read from the input yet, leading to a comparison
 *    with an uninitialized value. This makes the logic incorrect.
 * 5. It increments a counter if the flawed comparison and a positive score check
 *    are met.
 * 6. Finally, it prints the counter.
 *
 * @complexity
 * Time Complexity: O(n) for the single loop.
 * Space Complexity: O(n) for the dynamically allocated array.
 */
#include <bits/stdc++.h>

using namespace std;
int main() {
  int *arr; // Pointer for a dynamically allocated array of scores.
  int n, k; // n: number of participants, k: the rank to use as a threshold.
  int count = 0; // Counter for participants who (supposedly) advance.
  
  cin >> n >> k;
  
  arr = new int[n]; // Allocate memory for n scores.

  /**
   * @block
   * @brief Reads scores and attempts to count qualifiers in a single pass.
   * @warning This loop contains a logical flaw. For any index `i < k-1`, it
   * compares `arr[i]` with `arr[k-1]`, but `arr[k-1]` has not yet been read
   * from the input at that point. It compares against an uninitialized value.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    if (arr[i] > 0) {
      // This comparison is incorrect for i < k-1.
      if (arr[i] >= arr[k - 1]) {
        count++;
      }
    }
  }
  
  cout << count << endl;
  
  // Note: The dynamically allocated memory 'arr' is not freed with 'delete[]'.
  // This is a memory leak, but common in short-lived competitive programming submissions.
}
