/**
 * @48879334-c008-491c-b0d3-e5bc3cd174c0/158A.CPP
 * @brief This program calculates the number of participants who advance in a competition.
 *        Participants advance if their score is greater than or equal to the score
 *        of the K-th place finisher and their score is positive.
 * Algorithm: Linear scan to read scores, then a linear scan to count participants
 *            meeting the criteria based on the K-th score.
 * Time Complexity: O(N) where N is the number of participants, due to two passes through the scores.
 * Space Complexity: O(N) for storing participant scores in an array.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#define fi first            // Macro for accessing the first element of a pair.
#define se second           // Macro for accessing the second element of a pair.
#define pb push_back        // Macro for adding an element to the end of a vector.
#define mp make_pair        // Macro for creating a pair.
#define pi 2 * acos(0.0)    // Definition of pi.
#define eps 1e-9            // A small value for floating-point comparisons (epsilon).
#define PII pair<int, int>  // Type definition for a pair of integers.
#define PDD pair<double, double> // Type definition for a pair of doubles.
#define LL long long        // Type definition for long long integer.
#define INF 1000000000      // Definition of a large integer value representing infinity.

using namespace std;

// Global variables:
// N: Total number of participants.
// K: The K-th place to consider for the cutoff score.
// x, y: Loop counters or temporary variables.
// ans: Stores the final count of qualifying participants.
int N, K, x, y, ans;
// arr: Array to store the scores of participants. Size 60 suggests a maximum of 59 participants (1-indexed).
int arr[60];

int main() {
  // Functional Utility: Read the total number of participants (N) and the K-th place (K).
  scanf("%d %d", &N, &K);
  // Block Logic: Read the scores of all participants into the array.
  // Participants are 1-indexed, so scores are stored from arr[1] to arr[N].
  for (x = 1; x <= N; x++)
    scanf("%d", &arr[x]);

  // Initialize the count of qualifying participants to zero.
  ans = 0;
  // Block Logic: Iterate through the participants to count those who qualify.
  // Pre-condition: A participant qualifies if their score is greater than or equal to the
  // score of the K-th place finisher (arr[K]) AND their score is positive.
  // The loop continues as long as the current participant's score is valid for comparison
  // (i.e., x <= N and arr[x] >= arr[K]).
  for (x = 1; (x <= N) && (arr[x] >= arr[K]); x++) {
    if (arr[x] == 0) // If a score is zero, no more participants beyond this point can qualify (problem specific rule).
      break;
    ans++; // Increment the count of qualifying participants.
  }

  // Functional Utility: Print the final count of qualifying participants.
  printf("%d\n", ans);
  return 0; // Indicate successful program termination.
}
