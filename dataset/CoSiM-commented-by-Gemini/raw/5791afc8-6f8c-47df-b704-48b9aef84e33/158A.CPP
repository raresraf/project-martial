/**
 * @file 158A.CPP
 * @brief An incorrect and confused attempt to solve the "Next Round" problem from Codeforces.
 * @details The "Next Round" problem requires counting participants whose score is
 * positive and greater than or equal to the k-th place finisher's score. This
 * implementation does not correctly apply this logic.
 *
 * @algorithm (as implemented)
 * The logic in this file is flawed and does not solve the problem.
 * 1. **Swapped Inputs**: It reads the number of participants `n` into variable `k` and the
 *    rank `k` into variable `n`.
 * 2. It reads the scores into an array.
 * 3. **Incorrect Loop**: It enters a `while` loop with a condition that primarily
 *    checks if the input array is sorted and if the original rank `k` is positive.
 * 4. Inside the loop, it decrements the original rank `k` (stored in `n`). When this
 *    counter reaches zero, it enters another `while` loop with unrelated logic that
 *    can run out of bounds.
 * 5. The core logic of comparing scores against the k-th place score is missing.
 *
 * @complexity
 * Time Complexity: O(n) for the loops, where n is the number of participants.
 * Space Complexity: O(n) for the scores array.
 */
#include <iostream>

using namespace std;

int main() {
  int k, n;
  // NOTE: The variables are swapped. The problem's 'n' is read into 'k' and 'k' into 'n'.
  cin >> k >> n;
  
  // Uses a non-standard VLA of size n+1 (since the input 'k' is n).
  int a[k + 1];
  for (int i = 0; i < k; i++)
    cin >> a[i];
  
  // Sets a sentinel value at the end of the array.
  a[k] = 0;
  int i = 0;

  /**
   * @block
   * @warning The logic in this loop is incorrect for the problem statement.
   * It does not use the k-th score as a threshold and seems to be based on
   * a misunderstanding of the problem's rules.
   */
  while (a[i] >= a[i + 1] && n > 0) {
    if (a[i] == 0)
      break;
    n--; // Decrements the rank k (stored in n).
    
    // This block is entered once, when the loop has run 'k' times.
    if (n == 0) {
      // This inner loop is also flawed. It compares adjacent elements and can
      // run past the end of the array if the last few scores are identical.
      while (a[i] == a[i + 1])
        i++;
    }
    i++;
  }
  
  // The output 'i' does not correctly represent the number of advancing players.
  cout << i << endl;
  return 0;
}
