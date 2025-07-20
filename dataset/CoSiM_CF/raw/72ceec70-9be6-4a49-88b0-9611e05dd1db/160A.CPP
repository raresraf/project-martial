#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, sum = 0, bs = 0, k;
  cin >> i;
  vector<int> aayu(i);
  for (int j = 0; j < i; j++) {
    cin >> aayu[j];
    sum = sum + aayu[j];
  }
  sort(aayu.begin(), aayu.end());

  for (k = 0; k < i; k++) {
    bs = bs + aayu[i - k - 1];
    if (bs > sum / 2)
      break;
  }
  k = k + 1;
  cout << k;
  return 0;
}