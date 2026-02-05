/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem.
 * 
 * This program determines how many contestants will advance to the next round.
 * A contestant advances if their score is equal to or greater than the score
 * of the k-th place finisher, provided their score is positive.
 *
 * Algorithm: The core logic involves finding the k-th largest score to set a
 * threshold and then iterating through all scores to count how many meet the
 * advancement criteria.
 *
 * Time Complexity: O(N), where N is the number of participants, due to the two
 * linear passes over the scores (one for input, one for counting).
 * Space Complexity: O(N) to store the scores of the participants.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: number of participants, k: 1-based rank of the threshold finisher.
  int n, k, e, i, a, c = 0;

  // Read the total number of participants and the rank k.
  cin >> n;
  cin >> k;

  // Vector to store the scores of all participants.
  vector<int> b;
  /**
   * Block Logic: Read all N participant scores from standard input
   * and store them in a vector.
   * Pre-condition: N is the number of scores to be read.
   * Invariant: After each iteration, one more score is added to the vector 'b'.
   */
  for (i = 0; i < n; i++) {
    cin >> a;
    b.push_back(a);
  }

  // Determine the threshold score. It's the score of the k-th participant.
  // Using k-1 for 0-based vector indexing.
  e = b[k - 1];

  /**
   * Block Logic: Count the number of participants who will advance.
   * Pre-condition: 'e' holds the threshold score from the k-th place. 'b' contains all scores.
   * Invariant: 'c' holds the count of advancing participants found so far in the scanned portion of 'b'.
   * A participant advances if their score is >= the threshold and is positive.
   */
  for (i = 0; i < n; i++) {
    if (e <= b[i] && b[i] > 0) {
      c = c + 1;
    }
  }

  // Output the final count of advancing participants.
  cout << c;
}