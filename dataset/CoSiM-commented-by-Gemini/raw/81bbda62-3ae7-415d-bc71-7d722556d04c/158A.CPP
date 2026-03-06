/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" competitive programming problem.
 *
 * The problem is to count how many participants advance to the next round.
 * A participant advances if they have a positive score and their score is
 * equal to or greater than the k-th place finisher's score. The list of scores
 * is provided in non-increasing order.
 *
 * @algorithm Sequential processing without storage.
 * @details The solution cleverly processes the scores in a stream without storing
 * them in an array. It reads the first k scores to find the threshold score (the
 * k-th score) and handle the zero-score edge case. Then, it continues reading
 * the remaining scores to count how many additional participants tie with the k-th
 * place finisher.
 *
 * @complexity
 * Time: O(N), where N is the number of participants, as we iterate through the
 *       scores at most once.
 * Space: O(1), as we only use a few variables to store the state.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to handle input and determine the number of advancers.
 */
int main() {
  int n; // Total number of participants.
  int k; // The rank of the participant whose score is the threshold.
  cin >> n >> k;

  int a; // Stores the score of the k-th participant.

  // --- Stage 1: Process the first k participants ---
  // Pre-condition: 'k' is the number of participants to check to find the threshold score.
  // Block Logic: Read the scores of the first 'k' participants to find the k-th score
  // and check for the positive score rule.
  for (int i = 0; i < k; i++) {
    cin >> a;
    // If any participant in the top k has a score of 0, no one with a score of 0 can advance.
    // Since scores are non-increasing, all subsequent participants also have a score of 0.
    // Therefore, only the 'i' participants before this one (who had positive scores) advance.
    if (a == 0) {
      cout << i << endl;
      return 0;
    }
  }

  // Invariant: At this point, all first 'k' participants have positive scores,
  // and 'a' holds the score of the k-th participant. All 'k' participants are potential advancers.

  int b; // A temporary variable for scores of participants from k+1 to n.
  // --- Stage 2: Process the remaining participants ---
  // Pre-condition: We have at least 'k' advancers. 'a' is the score to beat.
  // Block Logic: Check how many more participants (from k+1 to n) have a score
  // equal to the k-th participant's score.
  for (int i = k; i < n; i++) {
    cin >> b;
    // Since the scores are non-increasing, the first score not equal to 'a'
    // must be smaller. At this point, no more participants can advance.
    if (b != a) {
      cout << i << endl; // 'i' is the total count of advancers so far.
      return 0;
    }
  }

  // If the second loop completes, it means all 'n' participants had a score
  // equal to or greater than the k-th finisher (and all were positive).
  // Therefore, all 'n' participants advance.
  cout << n << endl;
  
  return 0;
}
