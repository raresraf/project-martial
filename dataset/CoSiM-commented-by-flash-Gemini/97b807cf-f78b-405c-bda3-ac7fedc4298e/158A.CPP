/**
 * @file 158A.CPP
 * @brief This program determines the number of participants who advance to the next round of a competition.
 * Participants advance if their score is greater than or equal to the k-th place finisher's score
 * and their score is positive.
 *
 * Algorithm: Direct comparison and counting.
 * 1. Read the number of participants (n) and the k-th position.
 * 2. Read all participant scores.
 * 3. Iterate through scores, comparing each against the k-th score and checking for positivity.
 * Time Complexity: O(N) where N is the number of participants, due to a single pass for input and another for counting.
 * Space Complexity: O(N) to store participant scores.
 */

#include <algorithm>
#include <iostream>
#include <vector>

#define ll long long

using namespace std;

/**
 * @brief Main function of the program.
 * Reads inputs for participant scores and determines qualifiers based on the k-th score.
 * @pre n and k are positive integers, 1 <= k <= n.
 * @pre Scores in vector 'a' are integers.
 * @post Prints the total count of participants who qualify.
 * @return 0 upon successful execution.
 */
int main() {
  ll n, k;
  // Functional Utility: Reads the total number of participants (n) and the significant k-th rank.
  cin >> n >> k;
  // Functional Utility: Declares a dynamic array (vector) to store the 'n' participant scores.
  vector<ll> a(n);
  // Block Logic: Populates the vector 'a' with 'n' participant scores.
  // Invariant: After loop completion, 'a' contains all 'n' scores.
  for (int i = 0; i < n; i++)
    cin >> a[i];
  // sort(a.begin(),a.end()); // Original commented line, preserved as per instruction to not alter code.

  // Functional Utility: Initializes a counter for qualifying participants.
  ll count = 0;
  // Block Logic: Iterates through all participant scores to identify qualifiers.
  // Invariant: 'count' accurately reflects the number of qualifying participants encountered so far.
  for (int i = 0; i < n; i++)
    // Block Logic: Checks if the current participant's score meets the two criteria for qualification.
    // Pre-condition: a[i] is a valid participant score.
    // Invariant: The criteria for advancing (score >= k-th score and positive) are applied correctly.
    if (a[i] >= a[k - 1] && a[i] > 0)
      count++;
  // Functional Utility: Outputs the final count of participants advancing to the next round.
  cout << count << "
";
  return 0;
}
