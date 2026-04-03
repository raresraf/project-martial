/**
 * @8dcc5e33-d184-431d-ab7b-ea1a3bb842c7/158A.CPP
 * @brief This program solves the "Next Round" problem (Codeforces 158A). It counts
 * the number of participants who qualify for the next round of a competition.
 * A participant qualifies if their score is positive and greater than or equal to
 * the score of the k-th placed participant.
 * Algorithm: Iterative filtering and counting based on a dynamic cutoff score.
 * Time Complexity: O(N), where N is the number of participants.
 * Space Complexity: O(1) for storing variables, as scores are processed on the fly without storing the full array.
 */

#include <iostream> // Required for standard input/output operations (cin, cout)
using namespace std;

int main() {
  // @var k: The 1-indexed position of the participant whose score sets the cutoff.
  // @var n: The total number of participants.
  // @var num: Stores the cutoff score (score of the k-th participant).
  // @var a: Stores the score of the current participant being read.
  // @var cnt: Counter for participants who advance to the next round.
  int k, n, num, a, cnt = 0;
  
  // Block Logic: Read the total number of participants (n) and the cutoff position (k).
  cin >> n >> k;
  
  // Block Logic: Iterate through each participant's score.
  // Invariant: 'cnt' accumulates the count of qualifying participants.
  // Pre-condition: 'n' specifies the number of scores to read.
  for (int i = 1; i <= n; i++) {
    cin >> a; // Read the current participant's score.
    
    // Block Logic: If the current participant is among the first 'k' participants
    // and has a positive score, they automatically qualify.
    if (i <= k && a > 0)
      cnt++;
    
    // Block Logic: When processing the k-th participant, establish their score as the cutoff.
    if (i == k) {
      if (a > 0) // Only set 'num' if the k-th participant's score is positive.
        num = a;
    }
    
    // Block Logic: For participants beyond the k-th position, check if their score
    // is equal to the cutoff score ('num') and add them to the count if they are.
    // Note: This logic assumes 'num' is correctly set by the k-th participant and 'a > 0' is implicitly
    // covered by the initial check for i <= k. For i > k, it only checks 'a == num'.
    // A more robust check might be `if (i > k && a >= num && a > 0)`.
    if (i > k && num == a)
      cnt++;
  }
  
  // Post-condition: Print the total number of participants who advance.
  cout << cnt;
  return 0; // Explicitly return 0 for successful execution.
}
