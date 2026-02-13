/**
 * @file 158A.CPP
 * @brief A clear and direct solution for the "Next Round" problem.
 * @details This program determines how many participants will advance to the next round of a
 * competition. The rule is that a participant advances if their score is equal to or
 * greater than the score of the k-th place finisher, as long as their score is also
 * positive. The input scores are provided in non-increasing order.
 *
 * @section algorithm Algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read the `n` scores into a vector.
 * 3. The score of the k-th participant (`a[k-1]`) is used as the cutoff score.
 * 4. Initialize a counter for advancing participants to 0.
 * 5. Iterate through all participants' scores from the first to the last.
 * 6. For each score, check if it meets both advancement criteria:
 *    a. The score is greater than or equal to the cutoff score (`a[k-1]`).
 *    b. The score is positive (not zero).
 * 7. If both conditions are met, increment the counter.
 * 8. After checking all participants, print the final count.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of participants, for reading the scores and then iterating through them once.
 * - Space Complexity: O(N) to store the scores in a vector.
 */
#include <iostream>
#include <vector>
using namespace std;

// A. Next Round
// https://codeforces.com/problemset/problem/158/A
int main() {
  int n, k;
  cin >> n >> k;
  vector<int> a(n);
  // Read all the scores into a vector.
  for (int i = 0; i < n; i++)
    cin >> a[i];

  int count = 0;
  // Iterate through each participant's score to check if they advance.
  for (int i = 0; i < n; i++) {
    // A participant advances if their score is >= the k-th person's score,
    // and their score is positive.
    if (a[i] >= a[k - 1] && a[i] != 0)
      count++;
  }
  cout << count << endl;
  return 0;
}
