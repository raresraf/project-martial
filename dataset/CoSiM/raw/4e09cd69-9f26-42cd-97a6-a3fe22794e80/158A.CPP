/*
Next Round

"Contestant who earns a score equal to or greater than the k-th place finisher's
score will advance to the next round, as long as the contestant earns a positive
score..."   an excerpt from contest rules.

A total of n participants took part in the contest (n   k), and you already know
their scores. Calculate how many participants will advance to the next round.
*/
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  int arr[n];
  for (int i = 0; i < n; i++)
    cin >> arr[i];
  int ans = k;
  for (int i = k; i < n; i++) {
    if (arr[i] == arr[k - 1])
      ans++;
    else
      break;
  }
  if (arr[k - 1] == 0) {
    int j = 0;
    while (arr[j++] > 0)
      ;
    ans = j - 1;
  }
  cout << ans;
}