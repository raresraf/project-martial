/**
 * @file 158A.CPP
 * @brief Compact solution for the "Next Round" programming contest problem.
 * @details The program determines how many contestants advance to the next round.
 * A contestant advances if they have a positive score that is equal to or
 * greater than the score of the k-th place finisher. The input scores are
 * provided in non-increasing order.
 *
 * Algorithm:
 * This solution leverages the non-increasing order of scores for an efficient check.
 * 1. The scores are read into a 1-indexed array.
 * 2. A single loop iterates from the first contestant (index 1).
 * 3. The loop's continuation condition is key: it continues as long as the current
 *    contestant's score `arr[x]` is greater than or equal to the k-th contestant's
 *    score `arr[K]`. Because the scores are sorted, as soon as this condition is
 *    false, no subsequent contestant can qualify on score.
 * 4. Inside the loop, an additional check ensures that the contestant's score is
 *    positive. If a score of 0 is encountered, the loop terminates, as no one
 *    with a zero score can advance.
 * 5. A counter is incremented for each contestant that passes both checks.
 *
 * Time Complexity: O(N) in the worst case (for reading), but the counting loop
 * can terminate early, making it O(P) where P is the number of participants who
 * advance.
 * Space Complexity: O(1), as the array size is fixed and small.
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

// Common competitive programming macros.
#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long
#define INF 1000000000

using namespace std;

int N, K, x, y, ans;
int arr[60];

int main() {
  // Read problem parameters N (number of contestants) and K (the rank to compare against).
  scanf("%d %d", &N, &K);
  
  // Read scores into a 1-indexed array.
  for (x = 1; x <= N; x++)
    scanf("%d", &arr[x]);

  ans = 0;
  
  /**
   * @block
   * @description This single loop efficiently counts all advancing contestants.
   * It iterates as long as the current contestant's score is high enough.
   * Because the scores are sorted, no one after the first failure can pass.
   */
  for (x = 1; (x <= N) && (arr[x] >= arr[K]); x++) {
    // A contestant cannot advance with a score of 0.
    if (arr[x] == 0)
      break;
    ans++;
  }

  printf("%d\n", ans);
  return 0;
}