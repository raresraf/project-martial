/**
 * @69eee920-b5d8-4246-991c-740ba32f291b/158A.CPP
 * @brief Problem: Next Round.
 *        Determines the number of participants who advance to the next round of a competition.
 *        Participants advance if their score is greater than or equal to the k-th place finisher's score
 *        and their score is strictly positive. This is a common competitive programming problem.
 * Algorithm: Array traversal and conditional counting.
 *            1. Read the total number of participants ('n') and the k-th position ('k').
 *            2. Read all 'n' participant scores into a vector.
 *            3. Iterate through the scores. For each score, check if it meets the advancement criteria:
 *               a. Score is strictly positive.
 *               b. Score is greater than or equal to the score of the k-th place participant (v[k-1]).
 *            4. Count the number of participants satisfying both conditions.
 * Time Complexity: O(n) where 'n' is the number of participants.
 *                  The code iterates through the vector of 'n' participant scores twice (once for input, once for counting).
 * Space Complexity: O(n) for storing the 'n' participant scores in a vector.
 */
#include <bits/stdc++.h> // Functional Utility: Includes various standard C++ library headers.

using namespace std;

typedef long long ll;     // Functional Utility: Defines 'll' as an alias for 'long long'.
typedef vector<ll> vi;    // Functional Utility: Defines 'vi' as an alias for a vector of 'long long' (used as scores).
#define rep(i, a, b) for (int i = a; i < b; i++) // Functional Utility: Macro for a common for loop pattern.

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n, k;
  // Block Logic: Reads the total number of participants ('n') and the k-th qualifying position ('k').
  cin >> n >> k;
  vi v(n); // Functional Utility: Declares a vector of 'long long' to store the 'n' participant scores.
  
  // Block Logic: Reads 'n' participant scores into the vector 'v'.
  // Pre-condition: 'n' is a positive integer.
  // Invariant: After this loop, 'v' contains all participant scores.
  rep(i, 0, n) { 
    cin >> v[i]; 
  }

  int count = 0; // Invariant: 'count' stores the number of participants who advance.
  // Block Logic: Iterates through the scores to count advancing participants.
  // Pre-condition: 'v' is populated with scores, and 'k-1' is a valid index.
  // Invariant: 'count' is incremented for each participant meeting the advancement criteria.
  rep(i, 0, n) {
    // Block Logic: Checks if the current participant's score meets the advancement criteria.
    // Pre-condition: 'v[i]' is the current score, 'v[k-1]' is the k-th place score.
    // Post-condition: 'count' is incremented if 'v[i]' is positive and >= 'v[k-1]'.
    if (v[i] >= v[k - 1] && v[i] > 0) {
      count++; // Functional Utility: Increments the counter.
    }
  }
  // Functional Utility: Prints the total count of participants who advance.
  cout << count;
  return 0; // Functional Utility: Indicates successful program execution.
}
