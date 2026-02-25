/**
 * @file 158A.CPP
 * @brief Solution for the "Next Round" programming contest problem.
 * @details The program counts the number of contestants who advance to the next round.
 * A contestant advances if they have a positive score and their score is greater than
 * or equal to the k-th place finisher's score. The input scores are sorted in
 * non-increasing order.
 *
 * Algorithm:
 * 1. Read the number of contestants `n` and the rank `k`.
 * 2. Read the `n` scores and store them in a `std::vector`.
 * 3. The score of the k-th place contestant, `scores[k-1]`, is the threshold.
 * 4. Iterate through all `n` scores.
 * 5. For each score, check if it is positive AND greater than or equal to the threshold.
 * 6. Increment a counter for each contestant who meets these criteria.
 * 7. Print the final counter value.
 *
 * Time Complexity: O(n), for reading the scores and then iterating through them.
 * Space Complexity: O(n) to store the scores in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, i, c = 0, num;
  vector<int> a;
  cin >> n >> k;

  // Read all scores into a vector.
  for (i = 0; i < n; i++) {
    cin >> num;
    a.push_back(num);
  }

  /**
   * @block
   * @description This loop iterates through all contestants to count who advances.
   * It compares each contestant's score against the k-th contestant's score.
   */
  for (i = 0; i < n; i++) {
    // A contestant advances if their score is >= the k-th score and is positive.
    if ((a[i] >= a[k - 1]) && (a[i] > 0)) {
      c++;
    }
  }
  
  // Output the total count of advancing contestants.
  cout << c << endl;
  return 0;
}