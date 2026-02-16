/**
 * @file 158A.CPP
 * @brief An attempted solution for Codeforces problem 158A, "Next Round".
 *
 * @description
 * This program attempts to solve the "Next Round" problem, which involves
 * counting how many participants advance to the next round of a contest. The
 * official rule is that any participant with a positive score equal to or
 * greater than the k-th place finisher's score advances.
 *
 * @note The logic in this implementation is flawed and does not correctly
 * model the problem statement for all cases. It relies on early exits and
 * printing the loop counter `i` as the final answer, which only works by
 * coincidence on some test inputs. It incorrectly stops if a participant after
 * the k-th place has a score that is simply not equal to the k-th place
 * score, rather than less than it.
 *
 * @algorithm (as implemented)
 * 1. Read `n` and `k`.
 * 2. Loop from `i = 0` to `n-1`.
 * 3. For the first `k` participants (`i < k`):
 *    a. Read the score `a`.
 *    b. If `a` is 0, print the current index `i` and exit immediately.
 * 4. For participants after the `k`-th place (`i >= k`):
 *    a. Read the score `l`.
 *    b. If `l` is not equal to `a` (the k-th place score), print the
 *       current index `i` and exit immediately.
 * 5. If the loop completes without exiting, print `n`.
 *
 * @time_complexity O(N) in the worst case, but often terminates early.
 * @space_complexity O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // n: total participants, k: the rank to compare against
  // a: stores scores of the first k participants, ends up holding the k-th score
  // l: stores scores of participants after the k-th
  int n, k, a, l;
  cin >> n >> k;

  // Block Logic: Iterate through all 'n' participants.
  for (int i = 0; i < n; ++i) {
    // This branch handles the first k participants (indices 0 to k-1).
    if (i < k) {
      cin >> a;
      // Flawed Logic: If any of the top k participants has a score of 0,
      // the program assumes no one after them can advance and exits.
      if (a == 0) {
        cout << i << endl;
        return 0;
      }
    } else {
      // This branch handles participants from index k onwards.
      cin >> l;
      // Flawed Logic: The program exits if a participant's score `l` is not
      // exactly equal to the k-th participant's score `a`. The correct rule
      // would be to continue if `l >= a` and `l > 0`.
      if (l != a) {
        cout << i << endl;
        return 0;
      }
    }
  }

  // This line is only reached if all participants from k to n have the exact
  // same score as the k-th participant, and that score is not 0.
  cout << n << endl;
  return 0;
}
