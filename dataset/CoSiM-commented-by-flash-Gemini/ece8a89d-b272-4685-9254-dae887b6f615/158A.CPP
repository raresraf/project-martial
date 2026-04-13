/**
 * @file 158A.CPP
 * @brief This program determines the number of participants who advance to the
 *        next round of a competition. Participants advance if their score is
 *        positive and greater than or equal to the score of the k-th placed participant.
 * Algorithm: Iterative filtering based on score thresholds.
 * Time Complexity: O(N) where N is the number of participants, due to linear traversals.
 * Space Complexity: O(N) for storing the scores of all participants.
 */
#include <bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef vector<string> vs;
typedef pair<int, int> pi;
typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;

#define sz(v) ((int)((v).size()))
#define all(v) ((v).begin(), (v).end())
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
#define nl << "\n"

int main() {
  // Functional Utility: Optimizes C++ standard input/output operations for faster execution.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  // N: total number of participants.
  // a[50]: array to store scores of up to 50 participants.
  // nb: count of participants advancing to the next round.
  // k: the k-th place threshold.
  int N, a[50], nb = 0, k, i, j;
  // Reads the total number of participants (N) and the k-th place (k).
  cin >> N >> k;
  /**
   * Block Logic: Reads the scores of all N participants into array 'a'.
   * Invariant: After each iteration, one participant's score is read.
   */
  for (i = 0; i < N; i++) {
    cin >> a[i];
  }
  // sort(a, a + N); // Commented out sorting, implying scores are already sorted or sorting is not required for the logic.

  // Initializes loop counter 'i' for iterating through participants.
  i = 0;
  /**
   * Block Logic: Counts participants from the beginning who have positive scores
   *              and are within the top 'k' positions.
   * Pre-condition: 'a' contains participant scores, 'N' is total participants, 'k' is the threshold.
   * Invariant: 'nb' increments as long as participants meet the criteria for the top 'k' places
   *            and have a positive score.
   */
  while (i < N && a[i] > 0 && i < k) {
    nb++; // Increment count of advancing participants.
    i++;  // Move to the next participant.
  }
  // Adjusts 'i' to point to the k-th participant's index (0-indexed).
  j = i--;
  /**
   * Block Logic: Continues counting participants who have positive scores and
   *              a score equal to the k-th participant's score (if k-th participant existed and had a positive score).
   * Pre-condition: 'j' is set to the index immediately after the k-th position (or where the first '0' score was found),
   *                'i' is set to the k-th position.
   * Invariant: 'nb' increments as long as participants meet the score threshold (a[j] >= a[i])
   *            and have a positive score.
   */
  while (j < N && a[j] > 0 && a[j] >= a[i]) {
    j++;  // Move to the next participant.
    nb++; // Increment count of advancing participants.
  }
  // Prints the total number of participants who advance.
  cout << nb << endl;
}
