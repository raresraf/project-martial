/**
 * @file 158A.CPP
 * @brief This competitive programming solution addresses the "Next Round" problem.
 *        In a competition, participants advance to the next round if their score is
 *        greater than or equal to the score of the k-th placed participant,
 *        provided their score is positive. The task is to count how many participants advance.
 * Algorithm: Linear Scan and Conditional Counting.
 *            1. Read the number of participants `n` and the k-th position `k`.
 *            2. Read all `n` participants' scores.
 *            3. Identify the threshold score: the score of the participant at the k-th position.
 *            4. Iterate through all participants' scores. For each participant, check if their
 *               score meets two criteria: it must be positive, AND it must be greater than or
 *               equal to the threshold score.
 *            5. Count participants who satisfy both criteria.
 * Time Complexity: O(N), where N is the number of participants. The program involves
 *                  a single pass to read scores and another pass to count advancing participants.
 * Space Complexity: O(N) for storing the `n` participants' scores in an array.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function of the program. Reads participant scores, determines the threshold,
 *        counts advancing participants, and prints the result.
 * Functional Utility: Serves as the entry point for the "Next Round" problem.
 */
int main() {
  int a[50]; // Declares an array to store up to 50 participant scores.
  int coun = 0; // Counter for participants who advance to the next round.
  int n, i, k; // `n`: number of participants; `k`: the k-th position; `i`: loop counter.
  
  cin >> n >> k; // Reads the number of participants and the k-th position.

  // Block Logic: Reads `n` scores into the array `a`.
  // Note: The loop iterates from 1 to `n` (1-indexed thinking), but stores into `a[i-1]` (0-indexed array).
  // Invariant: After this loop, the array `a` contains all `n` participant scores.
  for (i = 1; i <= n; i++) {
    cin >> a[i - 1]; // Reads each score.
  }

  // `val` stores the score of the k-th placed participant (0-indexed: `a[k-1]`).
  int val = a[k - 1]; 

  // Block Logic: Handles a specific edge case where the first participant has a score of 0.
  // In some competitive programming problem contexts, a score of 0 might implicitly mean
  // no one with a positive score can advance if the k-th score is also 0.
  // This check might be a simplification based on problem constraints not fully stated.
  if (a[0] == 0)
    cout << "0"; // If the first score is 0, outputs 0 directly.
  else {
    // Block Logic: Iterates through all participants to count those who advance.
    // Invariant: `coun` accurately counts participants whose score is positive and
    //            greater than or equal to `val`.
    for (int j = 0; j < n; j++) {
      // Functional Utility: Checks the advancement criteria:
      // A participant advances if their score (`a[j]`) is greater than or equal
      // to the k-th placed participant's score (`val`), AND their score is positive.
      // (The `a[j] > 0` part is implicitly handled by `val` typically being positive
      // from problem constraints, or by the initial `if (a[0] == 0)` check.)
      if (a[j] >= val && a[j] > 0) // Explicitly added a[j] > 0 for clarity.
        coun++; // Increments counter if participant advances.
    }
    cout << coun; // Prints the total count of advancing participants.
  }

  return 0; // Indicates successful program termination.
}
