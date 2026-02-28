/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 158A - "Next Round").
 *
 * This program calculates the number of participants who will advance to the next round.
 * A participant advances if their score meets two conditions:
 * 1. Their score is greater than or equal to the score of the k-th place finisher.
 * 2. Their score is strictly positive.
 *
 * Algorithm: Linear scan with dynamic array and conditional counting.
 * The program dynamically allocates an array to store all participant scores.
 * It then identifies the score of the k-th placed participant.
 * Finally, it performs a second pass through the array, comparing each participant's score
 * against the k-th score and checking for positivity. A counter is incremented
 * for each participant meeting both criteria.
 *
 * Time Complexity: O(N), where N is the total number of participants.
 * This includes one pass to read all scores and a second pass to count the advancing participants.
 * Space Complexity: O(N) for storing the participant scores in the dynamically allocated array.
 * Note: This implementation has an implicit memory leak as `delete[] score` is missing.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <new>      // Required for dynamic memory allocation using `new`.
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads participant data, applies advancement rules, and prints the total count of advancing participants.
 * Uses dynamic memory allocation for scores.
 */
int main() {
  int n, k, count = 0, thresh; // 'n': total participants, 'k': k-th place index,
                               // 'count': count of advancing participants, 'thresh': k-th score.
  int *score; // Pointer to dynamically allocated array for scores.

  // Pre-condition: Read 'n' (total participants) and 'k' (the k-th position).
  cin >> n >> k;
  // Functional Utility: Dynamically allocate memory for 'n' integer scores.
  score = new int[n];
  // Block Logic: Read 'n' scores from input and store them in the dynamically allocated array.
  // Invariant: 'i' iterates from 0 to n-1.
  for (int i = 0; i < n; i++) {
    cin >> score[i];
  }
  // Determine the threshold score based on the k-th placed participant (0-indexed k-1).
  thresh = score[k - 1];
  // Block Logic: Iterate through the scores to count participants who meet the advancement criteria.
  // Invariant: 'j' iterates from 0 to n-1.
  for (int j = 0; j < n; j++) {
    // Advancement Criteria:
    // 1. Current participant's score (`score[j]`) must be greater than or equal to the threshold score (`thresh`).
    // 2. Current participant's score (`score[j]`) must be strictly greater than 0.
    if (score[j] >= thresh && score[j] > 0)
      count++; // Increment the counter if both conditions are met.
  }
  // Output the total number of participants who will advance.
  cout << count << endl; // Use endl for flushing output.
  // Note: Missing `delete[] score;` here, which would lead to a memory leak in a long-running program.
  return 0; // Indicate successful execution.
}
