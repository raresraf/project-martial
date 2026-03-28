/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from competitive programming platforms.
 *
 * The problem is to determine how many participants will advance to the next round.
 * A participant advances if they have a positive score and their score is greater
 * than or equal to the k-th place participant's score.
 *
 * @b Algorithm: The code uses a complex two-stage counting method. The first loop
 *    counts participants up to the k-th position, stopping early if a non-positive
 *    score is found. The second loop continues from the k-th position, counting
 *    any remaining participants who meet the score criteria based on the score of the
 *    last person counted in the first loop. This is a convoluted way to implement
 *    the simple rule of the problem.
 * @b Time_Complexity: O(N) for reading inputs and iterating through the scores.
 * @b Space_Complexity: O(N) to store the participant scores.
 */
#include <bits/stdc++.h>

using namespace std;

// Common macros and typedefs for competitive programming.
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
  // Fast I/O setup.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  
  int N, k;          // N: number of participants, k: rank for the cutoff score.
  int a[50];         // Array to store scores.
  int nb = 0;        // Counter for participants who advance.
  int i, j;

  cin >> N >> k;
  for (i = 0; i < N; i++) {
    cin >> a[i];
  }

  i = 0;
  /**
   * @brief First pass: Count participants up to the k-th rank.
   * This loop counts participants who have a positive score, but only up to the
   * k-th person.
   */
  while (i < N && a[i] > 0 && i < k) {
    nb++;
    i++;
  }
  
  // Set up pointers for the second pass. j starts where the first loop
  // stopped, and i is moved back to point to the last person who qualified.
  j = i--;

  /**
   * @brief Second pass: Count participants after the k-th rank.
   * This loop counts remaining participants who have a score equal to the
   * cutoff score (a[i]) and are positive.
   */
  while (j < N && a[j] > 0 && a[j] >= a[i]) {
    j++;
    nb++;
  }
  
  cout << nb << endl;
}