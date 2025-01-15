/*
Link: https://codeforces.com/problemset/problem/158/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  int n, k, i, scores[50];
  cin >> n >> k;
  for (i = 0; i < n; i++)
    cin >> scores[i];
  if (scores[0] == scores[n - 1])
    if (scores[0] == 0)
      k = 0;
    else
      k = n;
  else {
    if (scores[k - 1] == 0) {
      while (scores[k - 1] == 0 && k > 0)
        k--;
    } else {
      while (scores[k - 1] == scores[k] && k <= 49)
        k++;
    }
  }
  cout << k;
  return 0;
}