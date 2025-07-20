#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, i, playerScoreAtK, count = 0;
  cin >> n >> k;
  int arr[n];
  for (i = 0; i < n; i++) {
    cin >> arr[i];
  }
  playerScoreAtK = arr[k - 1];
  i = 0;
  while (arr[i] >= playerScoreAtK && arr[i] != 0) {
    count++;
    ++i;
    if (i == n)
      break;
  }
  cout << count << endl;
}
