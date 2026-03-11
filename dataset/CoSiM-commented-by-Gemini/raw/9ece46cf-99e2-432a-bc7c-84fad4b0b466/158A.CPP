/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem.
 *
 * This program determines how many participants will advance to the next round of a contest.
 * A participant advances if their score is equal to or greater than the score of the
 * k-th place finisher, provided their score is positive.
 *
 * Algorithm: The code reads all scores, identifies the score of the k-th participant,
 *            sorts the scores, and then counts how many participants meet the criteria.
 *
 * Time Complexity: O(N log N) due to the sorting operation, where N is the number of participants.
 * Space Complexity: O(N) to store the scores in a vector.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n; // Total number of participants.
  int k; // The rank of the participant whose score is the benchmark.
  cin >> n >> k;
  vector<int> scores; // Vector to store the scores of all participants.
  int count = 0; // Counter for participants who will advance.

  /**
   * @brief Read all participant scores into a vector.
   *
   * Pre-condition: 'n' holds the number of scores to read.
   */
  while (n--) {
    int a;
    cin >> a;
    scores.emplace_back(a);
  }

  // The score of the k-th participant, which is the threshold for advancing.
  // Note: The problem statement implies the input 'scores' are already in non-increasing order.
  int kScore = scores[k - 1];

  // The scores are sorted in ascending order. This is not strictly necessary for a more
  // optimal solution but is part of this implementation's logic.
  sort(scores.begin(), scores.end());

  /**
   * @brief Count the number of participants who advance.
   *
   * Invariant: Iterates through each score to check against the advancement criteria.
   * Logic: A participant advances if their score is positive and greater than or equal to kScore.
   */
  for (vector<int>::iterator it = scores.begin(); it != scores.end(); ++it) {
    int score = (*it);
    if (score >= kScore && score > 0)
      ++count;
  }
  
  // Output the total count of advancing participants.
  cout << count;
  return 0;
}
