/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem.
 *
 * This program calculates the number of participants who advance to the next round of a competition.
 * Participants advance if their score is greater than or equal to the k-th place finisher's score,
 * and their own score is strictly positive.
 *
 * Algorithm: Iterative comparison with k-th score.
 * Time Complexity: O(N) for reading input and iterating through participants, where N is the number of contestants.
 * Space Complexity: O(N) to store participant scores in an array.
 */
#include <bits/stdc++.h>
#include <string>
#define ll long long // Type alias for long long integer.
using namespace std;
/**
 * @brief Main function to determine the number of advancing participants.
 *
 * Functional Utility: Handles input for number of contestants, k-th place, processes scores, and counts qualifiers.
 */
int main() {
  /**
   * Functional Utility: Conditional compilation block for redirecting standard input/output
   * to files during local development. This is typically used for testing with predefined
   * input and capturing output without console interaction.
   * Pre-condition: `ONLINE_JUDGE` macro is not defined (i.e., local environment).
   */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);  // Redirects standard input from "input.txt".
  freopen("output.txt", "w", stdout); // Redirects standard output to "output.txt".
#endif
  int n, k, count = 0; // 'n' for total contestants, 'k' for k-th place, 'count' for qualifiers.
  cin >> n >> k;      // Read the number of contestants and the k-th place.
  int arr[n];         // Declare a dynamic array to store scores (VLA, C++ extension or C99).

  /**
   * Block Logic: Reads 'n' participant scores from standard input.
   * Stores scores starting from index 1 in the array (arr[1] to arr[n]).
   * Pre-condition: 'n' (number of contestants) has been read.
   * Invariant: `arr` contains the scores of 'n' participants.
   */
  for (int i = 1; i <= n; i++) {
    cin >> arr[i]; // Read score into array.
  }

  /**
   * Block Logic: Iterates through the participant scores to count how many qualify for the next round.
   * A participant qualifies if their score is greater than or equal to the k-th place score (arr[k])
   * AND their score is strictly positive.
   * Pre-condition: All scores are read into `arr`, and 'k' is a valid index.
   * Invariant: `count` accurately reflects the number of qualifiers encountered so far.
   */
  for (int j = 1; j <= n; j++) {
    if (arr[j] >= arr[k] && arr[j] > 0) {
      count++; // Increment count if participant qualifies.
    }
  }
  cout << count; // Output the total number of qualifiers.
}