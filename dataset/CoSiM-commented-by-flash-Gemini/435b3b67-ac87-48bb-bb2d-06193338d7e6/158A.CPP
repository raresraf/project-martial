/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 158A - "Next Round").
 *
 * This program calculates the number of participants who will advance to the next round.
 * A participant advances if their score meets two conditions:
 * 1. Their score is greater than or equal to the score of the k-th place finisher.
 * 2. Their score is strictly positive.
 *
 * Algorithm: Linear scan with conditional counting.
 * The program first reads all participant scores into an array.
 * Then, it performs a second pass through the array, comparing each participant's score
 * against the score of the k-th participant and checking if it's positive.
 * A counter is incremented for each participant meeting both criteria.
 *
 * Time Complexity: O(N), where N is the total number of participants.
 * This includes one pass to read all scores and a second pass to count the advancing participants.
 * Space Complexity: O(N) for storing the participant scores in an array.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads participant data, applies advancement rules, and prints the total count of advancing participants.
 */
int main() {
  int n, k, i, counter = 0; // 'n': total participants, 'k': k-th place index,
                           // 'i': loop counter, 'counter': count of advancing participants.
  int a[103]; // Declare an array to store scores. Size 103 for up to 100 participants with 1-based indexing.

  // Pre-condition: Read 'n' (total participants) and 'k' (the k-th position).
  cin >> n >> k;
  // Block Logic: Read 'n' scores from input and store them in the array 'a' using 1-based indexing.
  // Invariant: 'i' iterates from 1 to n.
  for (i = 1; i <= n; i++) {
    cin >> a[i];
  }

  // Block Logic: Iterate through the scores to count participants who meet the advancement criteria.
  // Pre-condition: Scores are available in array 'a'.
  // Invariant: 'i' iterates from 1 to n.
  for (i = 1; i <= n; i++) {
    // Advancement Criteria:
    // 1. Current participant's score (`a[i]`) must be greater than or equal to the k-th participant's score (`a[k]`).
    // 2. Current participant's score (`a[i]`) must be strictly greater than 0.
    if (a[i] >= a[k] && a[i] > 0) {
      counter++; // Increment the counter if both conditions are met.
    }
  }
  // Output the total number of participants who will advance, followed by a newline.
  cout << counter << endl;
}
