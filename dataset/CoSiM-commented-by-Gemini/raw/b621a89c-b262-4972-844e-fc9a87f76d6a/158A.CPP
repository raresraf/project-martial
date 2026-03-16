/**
 * @file 158A.CPP
 * @brief An attempted but incorrect solution to the "Next Round" problem (Codeforces 158/A).
 *
 * @details
 * This program attempts to solve the "Next Round" problem, but contains several
 * logical flaws that prevent it from passing all test cases. The goal of the problem
 * is to count how many participants have a score greater than or equal to the k-th
 * participant's score, provided their score is also positive. The input scores are
 * guaranteed to be in non-increasing order.
 *
 * The implemented algorithm deviates from a correct solution in several ways:
 * 1. It uses the score of the (k+1)-th participant (index `k`) as the threshold,
 *    not the k-th (index `k-1`).
 * 2. It incorrectly sorts the scores in ascending order, which invalidates the
 *    ranking provided in the input.
 * 3. It has a flawed early-exit condition for non-positive threshold scores.
 *
 * The final count is derived by using `lower_bound` on the incorrectly sorted array.
 * The comment "I didn't get all the test cases" indicates the author's awareness
 * of the issues.
 *
 * Time Complexity: O(N log N) due to the sorting step.
 * Space Complexity: O(N) to store the scores in a vector.
 */
#include <bits/stdc++.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
int main() {
  vector<int> v; // Using this vector, the score of the participants are stored.
  int c = 0, n, t, k;
  scanf("%d", &n);
  cin >> k;
  for (int i = 0; i < n; i++) {
    cin >> t;
    v.push_back(t);
  }

  // Logical Flaw: The k-th participant is at index k-1, not k. This uses the
  // score of the (k+1)-th participant as the threshold.
  int k1 = v[k];

  // Logical Flaw: The condition to advance is that a participant's score must be
  // positive. The threshold score itself can be zero or negative. This early exit
  // is incorrect and will fail cases where the threshold is 0 but other participants
  // have positive scores.
  if (k1 <= 0) { // According to teh condition,the score should not be<=0
    std::cout << "0" << std::endl;
    exit(0);
  }

  // Logical Flaw: The problem guarantees scores are in non-increasing order.
  // Sorting the array invalidates this ranking and the meaning of the k-th score.
  sort(v.begin(), v.end()); // if it isn't ,sort the array in ascending order

  // After sorting, this finds the first element not less than the (incorrect)
  // threshold score `k1`.
  auto i =
      lower_bound(v.begin(), v.end(), k1); // lower bound means all values >=k

  // This correctly computes the number of elements >= k1 in the sorted vector,
  // but the result is not the correct answer to the problem due to the prior flaws.
  cout << v.end() - i; // prins the no. of elements >=k,since v.end() points to
                       // the end of the vector
}

// Problem:https://codeforces.com/problemset/problem/158/A
// I didn't get all the test cases,feel free to modify