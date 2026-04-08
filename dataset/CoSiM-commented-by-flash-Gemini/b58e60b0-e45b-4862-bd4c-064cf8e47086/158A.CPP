/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem from Codeforces (Problem 158A).
 * Problem Statement: Determine the number of participants who will advance to the next round
 *                    based on specific criteria: their score must be greater than or equal to
 *                    the k-th place finisher's score, and also positive.
 * Algorithm: Two-pass iteration. In the first pass, all scores are read, and the k-th
 *            place finisher's score is identified. In the second pass, participants
 *            meeting the criteria are counted.
 * Time Complexity: O(N) where N is the number of participants.
 * Space Complexity: O(N) for storing the participant scores.
 * Domain: Competitive Programming, Array Processing.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to solve the "Next Round" problem.
 * Functional Utility: Reads participant scores and determines how many qualify for the next round
 *                     based on the given rules.
 */
int main() {
  // Variables:
  // n: Total number of participants.
  // k: The k-th place position.
  // arr: An array to store the scores of all participants.
  // count: Stores the number of participants who qualify for the next round.
  // k_th_score: Stores the score of the participant in k-th place.
  int n, k;
  cin >> n >> k;
  int arr[n], count = 0, k_th_score; // Renamed 'max' to 'k_th_score' for clarity.
  // Block Logic: Reads all participant scores and identifies the score of the k-th place finisher.
  // Pre-condition: `n` and `k` are positive integers; `arr` is an array of size `n`.
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    // Block Logic: When the loop reaches the k-th participant (0-indexed k-1), store their score.
    if (i + 1 == k) {
      k_th_score = arr[i];
    }
  }
  // Block Logic: Iterates through all scores again to count qualifying participants.
  // Invariant: A participant qualifies if their score is greater than or equal to `k_th_score` AND positive.
  for (int i = 0; i < n; i++) {
    if (arr[i] >= k_th_score && arr[i] > 0) {
      count++;
    }
  }
  // Block Logic: Outputs the total count of qualifying participants.
  cout << count << endl;
  return 0;
}