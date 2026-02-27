/*
   By Anmol Gupta :)
*/

/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details The problem is to count the number of participants who will advance
 * to the next round. A participant advances if their score is positive and
 * is greater than or equal to the score of the k-th place participant.
 *
 * @algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read the `n` scores into an array. The scores are given in non-increasing order.
 * 3. Handle the edge case where all scores are 0; in this case, no one advances.
 * 4. Get the score of the k-th participant. This score is the advancement threshold.
 * 5. Iterate through the participants from the first place downwards.
 * 6. For each participant, check if their score meets the criteria:
 *    a. Score is positive (greater than 0).
 *    b. Score is greater than or equal to the k-th place score.
 * 7. Count how many participants satisfy both conditions.
 * 8. Since the scores are sorted, we can stop counting as soon as we find a
 *    participant who does not meet the criteria.
 * 9. Print the final count.
 *
 * @complexity
 * Time Complexity: O(n), for reading the input and one pass to count the qualifiers.
 * Space Complexity: O(n), to store the scores in an array.
 */
#include <bits/stdc++.h>
using namespace std;
// A collection of common C++ macros for competitive programming.
#pragma GCC optimize("O3")
#define pb push_back
#define debug(x) cout << #x << " " << x << endl;
#define mp make_pair
#define ff first
#define ss second
#define allunique(v) v.erase(unique(v.begin(), v.end()), v.end());
#define endl "\n"
#define MOD 1000000007
#define MOD9 1000000009
#define gc getchar_unlocked
#define ms(s, n) memset(s, n, sizeof(s))
#define prec(n) fixed << setprecision(n)
#define eps 0.000001
#define pll pair<ll, ll>
#define all(v) v.begin(), v.end()
#define bolt ios::sync_with_stdio(0)
#define forr(i, n) for (ll i = 0; i < n; i++)
#define eof (scanf("%d", &n)) != EOF
typedef long long ll;

int main() {
  // Fast I/O
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  // printf("%I64d", n)
  ll n, k;      // n: number of participants, k: the rank to check against.
  ll temp;      // The score of the k-th participant (the threshold).
  ll res;       // The result: number of participants who advance.
  bool ok = true; // Flag to check if all scores are zero.
  
  cin >> n >> k;
  
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  // The array is 1-indexed.
  ll arr[n + 1];
  forr(i, n) {
    cin >> arr[i + 1];
    // If any score is non-zero, the 'ok' flag is set to false.
    if (arr[i + 1] != 0)
      ok = false;
  }

  // Edge case: If all scores are 0, nobody advances.
  if (ok == true) {
    cout << "0";
    return 0;
  }

  // Get the threshold score from the k-th participant.
  temp = arr[k];
  
  res = 0;
  int i = 1;
  /**
   * @block
   * @brief Count participants who meet the advancement criteria.
   * The loop continues as long as the participant's score is positive and
   * at least as high as the threshold score.
   */
  while (i <= n && arr[i] >= temp && arr[i] != 0) {
    res++;
    i++;
  }
  
  cout << res << endl;
}
