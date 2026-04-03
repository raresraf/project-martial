/**
 * @8d59ae67-cfdf-425c-a4f6-76a63e7a8be5/158A.CPP
 * @brief This program solves the "Next Round" problem (Codeforces 158A). It determines
 * the number of participants who will advance to the next round of a competition.
 * Participants advance if their score is greater than or equal to the score of the
 * k-th placed participant AND their score is strictly positive.
 * Algorithm: Array traversal and comparison (filtering).
 * Time Complexity: O(N) for input and another O(N) for filtering, resulting in O(N).
 * Space Complexity: O(N) to store the participant scores.
 */

#include <bits/stdc++.h> // Includes various standard libraries, often used in competitive programming.
using namespace std;

int main() {
  // @var k: The 1-indexed position of the participant whose score sets the cutoff.
  // @var n: The total number of participants.
  int k, n;
  cin >> n >> k; // Read the number of participants (n) and the cutoff position (k).

  // @var a: An array to store the scores of the participants.
  int a[n]; // C-style array declaration, size determined at runtime (VLA - Variable Length Array, GCC extension).
  
  // Block Logic: Read 'n' participant scores into the array.
  // Pre-condition: 'n' specifies the number of scores to read.
  for (int i = 0; i < n; i++)
    cin >> a[i];
  
  // @var num: Stores the score of the k-th placed participant (0-indexed: k-1).
  int num = a[k - 1];

  // @var ans: Counter for participants who advance to the next round.
  int ans = 0;
  
  // Block Logic: Iterate through the participant scores to count those who advance.
  // Invariant: 'ans' accumulates the count of qualifying participants.
  // Pre-condition: 'a' contains all participant scores, 'num' holds the cutoff score.
  for (int i = 0; i < n; i++)
    // Block Logic: Check if the current participant's score meets the two criteria for advancing.
    // 1. Score is greater than or equal to the k-th participant's score.
    // 2. Score is strictly positive.
    if (a[i] >= num && a[i] > 0)
      ans++; // Increment the counter if both conditions are met.
  
  // Post-condition: Print the total number of participants who advance.
  cout << ans;
  return 0; // Explicitly return 0 for successful execution.
}
