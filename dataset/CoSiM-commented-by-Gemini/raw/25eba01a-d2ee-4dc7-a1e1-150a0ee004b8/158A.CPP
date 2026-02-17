/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" problem on Codeforces (https://codeforces.com/problemset/problem/158/A).
 * @details The program is intended to solve the "Next Round" problem, where the goal is to count the
 * number of participants who advance to the next round. The rule is that a participant advances
 * if their score is positive and greater than or equal to the k-th place finisher's score.
 *
 * @warning The implementation has logical flaws with respect to the problem statement.
 * 1. It sorts the input array in ascending order, while the problem guarantees a non-increasing order
 *    which could be used directly.
 * 2. It incorrectly identifies the k-th place score after sorting.
 * 3. It counts elements less than or equal to this incorrect score, rather than greater than or equal to the correct threshold.
 * 4. It fails to check if the scores are positive.
 * The documented logic describes what the code *does*, not what it *should* do to solve the problem correctly.
 *
 * @algorithm
 * 1. Reads the number of participants (n) and the rank k (as m).
 * 2. Reads all n scores into a vector.
 * 3. Sorts the scores in ascending order.
 * 4. Identifies a threshold score, incorrectly using the element at index `m`.
 * 5. Uses `std::upper_bound` to find the first element strictly greater than this threshold.
 * 6. The result is the count of elements less than or equal to the threshold, which is the index of the
 *    iterator returned by `upper_bound`.
 *
 * @complexity
 * Time Complexity: O(N log N) due to the sorting step.
 * Space Complexity: O(N) to store the scores in a vector.
 */
// https://codeforces.com/problemset/problem/158/A

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace std;

int main() {
  // your code goes here
  int n, m;
  cin >> n >> m;

  int k;
  vector<int> v;
  // Block Logic: Read n scores from input into a vector.
  for (int i = 0; i < n; i++) {
    cin >> k;
    v.push_back(k);
  }
  
  // The scores are sorted in ascending order.
  sort(v.begin(), v.end());
  
  // Flawed Logic: 'upper_bound' finds the first element strictly greater than the threshold.
  // The threshold itself, v[m], is incorrectly chosen. It should be v[n-m] to get the
  // m-th largest score after an ascending sort. Also, the problem requires counting scores
  // >= the threshold, but this logic counts scores <= the threshold.
  // The positive score requirement is also ignored.
  vector<int>::iterator it = upper_bound(v.begin(), v.end(), v[m]);
  
  // The distance from the beginning to the upper_bound iterator gives the count of elements
  // that are less than or equal to v[m].
  cout << it - v.begin();
  // your code goes here
  return 0;
}