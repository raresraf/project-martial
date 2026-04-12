/**
 * @file 158A.CPP
 * @brief Logic for determining qualification in a multi-stage competition.
 * 
 * Algorithm: Threshold-based filtering using the k-th percentile value.
 * Time Complexity: $O(N \log N)$ due to sorting (though $O(N)$ is possible if input is pre-sorted).
 * Space Complexity: $O(N)$ to store participant scores.
 */

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k;
  // Reads the total number of participants (n) and the target rank (k).
  cin >> n >> k;
  vector<int> scores;
  int count = 0;

  // Functional Utility: Populates the score vector from standard input.
  while (n--) {
    int a;
    cin >> a;
    scores.emplace_back(a);
  }

  /**
   * Functional Utility: Identifies the qualification threshold based on the original 
   * position of the k-th participant (assuming input order matters or is pre-ranked).
   */
  int kScore = scores[k - 1];
  
  // Logic: Reorders scores for systematic evaluation of the qualification criteria.
  sort(scores.begin(), scores.end());

  /**
   * Block Logic: Evaluates each participant against the dual qualification constraints.
   * Condition 1: Score must be greater than or equal to the k-th participant's score.
   * Condition 2: Score must be strictly positive (score > 0).
   */
  for (vector<int>::iterator it = scores.begin(); it != scores.end(); ++it) {
    int score = (*it);
    if (score >= kScore && score > 0)
      ++count;
  }
  
  // Output the total number of qualified participants.
  cout << count;
  return 0;
}
