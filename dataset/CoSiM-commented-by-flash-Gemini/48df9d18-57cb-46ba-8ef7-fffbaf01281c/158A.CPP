/**
 * @48df9d18-57cb-46ba-8ef7-fffbaf01281c/158A.CPP
 * @brief This program calculates the number of participants who advance in a competition.
 *        Participants advance if their score is greater than or equal to the score
 *        of the K-th place finisher and their score is positive.
 * Algorithm: Linear scan to read scores into a vector, then a linear scan to count
 *            participants meeting the criteria based on the K-th score.
 * Time Complexity: O(N) where N is the number of participants, due to two passes through the scores.
 * Space Complexity: O(N) for storing participant scores in a `std::vector`.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Declare variables:
  // n: Total number of participants.
  // k: The k-th place to consider for the cutoff score.
  // i: Loop counter.
  // c: Counter for the number of participants who qualify (initialized to 0).
  // num: Temporary variable to read each participant's score.
  int n, k, i, c = 0, num;
  // a: A dynamic array (vector) to store the scores of participants.
  vector<int> a;
  // Functional Utility: Read the total number of participants (n) and the k-th place (k).
  cin >> n >> k;
  // Block Logic: Read the scores of all participants and store them in the vector 'a'.
  for (i = 0; i < n; i++) {
    cin >> num;
    a.push_back(num); // Add each score to the vector.
  }
  // Block Logic: Iterate through all participants to count those who qualify.
  // Pre-condition: A participant qualifies if their score (a[i]) is greater than or equal to
  // the score of the K-th place finisher (a[k-1]) AND their score is positive.
  for (i = 0; i < n; i++) {
    if ((a[i] >= a[k - 1]) && (a[i] > 0)) { // Check qualification criteria.
      c++; // Increment the counter if the participant qualifies.
    }
  }
  // Functional Utility: Output the total number of qualifying participants.
  cout << c << endl;
  return 0; // Indicate successful program termination.
}
