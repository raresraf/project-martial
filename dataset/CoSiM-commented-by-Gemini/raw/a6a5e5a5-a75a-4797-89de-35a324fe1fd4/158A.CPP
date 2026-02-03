/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem.
 * @see https://codeforces.com/problemset/problem/158/A
 *
 * This program determines how many contestants will advance to the next round
 * of a competition. The rule is that a contestant advances if their score is
 * equal to or greater than the score of the k-th place finisher, provided
 * their score is positive.
 *
 * Algorithm:
 * 1. Read the number of contestants (n) and the rank of the advancing
 *    contestant (k).
 * 2. Read the scores of all n contestants into a vector.
 * 3. Count the number of contestants who have a positive score and a score
 *    greater than or equal to the k-th contestant's score.
 * 4. The loop first counts the top k contestants with positive scores.
 * 5. Then, it continues to count any subsequent contestants who have the same
 *    score as the k-th contestant.
 *
 * Time Complexity: O(N), where N is the number of contestants.
 * Space Complexity: O(N) to store the scores.
 */
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::vector;

int main(void) {
  int n, k;
  vector<int> scores;

  cin >> n >> k;
  int count = 0;
  int score;

  // Invariant: Reads n scores from the input.
  while (count++ < n) {
    cin >> score;
    scores.push_back(score);
  }
  count = 0;
  int index;
  // Block Logic: This loop counts the number of contestants in the top k
  // places who have a positive score.
  for (index = 0; index < n && index < k; count++) {
    if (scores[index++] <= 0) {
      break;
    }
  }
  // Block Logic: This loop continues to count contestants who are tied with the
  // k-th place finisher and have a positive score.
  while (index < n && scores[index] > 0 && scores[index - 1] == scores[index]) {
    index++;
    count++;
  }
  cout << count;

  return 0;
}
