/**
 * @file 158A.CPP
 * @brief Solution to Codeforces Problem 158A: "Next Round".
 *        This program determines the number of participants who will advance
 *        to the next round of a competition. Participants advance if their
 *        score is greater than or equal to the score of the k-th placed participant,
 *        provided their score is also positive.
 *
 * Algorithm: Array traversal and conditional counting. After reading all scores,
 *            it iterates through them, comparing each participant's score to
 *            the k-th participant's score and checking if it's positive.
 * Time Complexity: O(N) where N is the number of participants, due to a single pass
 *                  iteration through the scores.
 * Space Complexity: O(N) to store the scores of all participants.
 *
 * Original comments:
 * By Anmol Gupta :)
 * Created on: Oct 26, 2017
 * Author: athena
 */
#include <bits/stdc++.h>
using namespace std;
#pragma GCC optimize("O3") // Compiler optimization directive for GCC.
#define pb push_back             // Macro for `vector::push_back`.
#define debug(x) cout << #x << " " << x << endl; // Debug macro to print variable name and value.
#define mp make_pair             // Macro for `std::make_pair`.
#define ff first                 // Macro for accessing the first element of a pair.
#define ss second                // Macro for accessing the second element of a pair.
#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end()); // Macro to remove duplicate elements from a sorted vector.
#define endl "\n"                // Macro for a newline character, often faster than `std::endl`.
#define MOD 1000000007           // Constant for modulo operation (prime number).
#define MOD9 1000000009          // Another constant for modulo operation.
#define gc getchar_unlocked      // Macro for `getchar_unlocked`, a faster character input function.
#define ms(s, n) memset(s, n, sizeof(s)) // Macro to fill a block of memory with a particular value.
#define prec(n) fixed << setprecision(n) // Macro to set output precision for floating-point numbers.
#define eps 0.000001             // Constant for epsilon value, often used in floating-point comparisons.
#define pll pair<ll, ll>         // Type definition for a pair of long longs.
#define all(v) v.begin(), v.end() // Macro for getting iterators to the beginning and end of a container.
#define bolt ios::sync_with_stdio(0) // Macro to untie C++ streams from C standard streams for faster I/O.
#define forr(i, n) for (ll i = 0; i < n; i++) // Macro for a common for loop pattern.
#define eof (scanf("%d", &n)) != EOF // Macro for checking end-of-file with scanf.
typedef long long ll;

int main() {
  /**
   * Functional Utility: Manages the overall competition logic, including reading participant scores,
   *                     determining the cutoff score, and counting qualified participants.
   */
  ios_base::sync_with_stdio(false); // Unties C++ standard streams from C standard streams for faster I/O.
  cin.tie(NULL); // Unties cin from cout, preventing flushing of cout before each cin operation.
  cout.tie(NULL); // Unties cout from cin, preventing flushing of cout before each cin operation.
  // printf("%I64d", n)
  ll n, k, temp, res; // n: number of participants, k: k-th place, temp: k-th place score, res: result count.
  bool ok = true;       // Flag to check if all initial scores are zero.
  cin >> n >> k;        // Read total participants (n) and k-th place (k).
  ll arr[n + 1];        // Array to store participant scores. (1-indexed for convenience).
  /**
   * Block Logic: Reads all participant scores and checks if any score is non-zero.
   * Pre-condition: `n` is the number of participants; `arr` is declared.
   * Invariant: `arr` contains all participant scores. `ok` is true if all scores are 0, else false.
   */
  forr(i, n) { // Loop through each participant.
    cin >> arr[i + 1]; // Read the score for the (i+1)-th participant.
    if (arr[i + 1] != 0) // Check if the current score is non-zero.
      ok = false;        // If any score is non-zero, the `ok` flag becomes false.
  }
  /**
   * Block Logic: Handles the edge case where all initial scores are zero.
   * Pre-condition: `ok` flag is set based on whether all scores are zero.
   * Invariant: If all scores are zero, 0 participants advance, and the program terminates.
   */
  if (ok == true) {
    cout << "0"; // Output 0 participants if all scores are zero.
    return 0;    // Terminate the program.
  }
  temp = arr[k]; // Stores the score of the k-th placed participant, which acts as the cutoff.
  // if(temp>0)
  res = 0; // Initialize the count of participants advancing to the next round.
  int i = 1; // Initialize loop counter (1-indexed for scores array).
  /**
   * Block Logic: Iterates through the scores to count participants who qualify for the next round.
   * Pre-condition: `arr` contains scores; `temp` is the cutoff score; `res` is initialized to 0.
   * Invariant: `res` counts participants whose score is >= `temp` and > 0.
   *            The loop continues as long as participants meet the criteria and there are more participants.
   */
  while (i <= n and arr[i] >= temp and arr[i] != 0) { // Check if participant 'i' qualifies based on score and positivity.
    res++;                                            // Increment the count of qualifying participants.
    i++;                                              // Move to the next participant.
  }
  cout << res << endl; // Output the total number of participants who advance to the next round.
  // Implicit return 0; due to successful program completion.
}