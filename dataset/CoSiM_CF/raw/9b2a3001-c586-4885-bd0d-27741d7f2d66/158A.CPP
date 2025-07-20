#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int k;
  cin >> k;
  vector<int> ar(n);
  for (auto &i : ar)
    cin >> i;
  int sum = 0;
  for (int i : ar) {
    if (i >= ar[k - 1] and i > 0)
      sum++;
    else
      break;
  }
  cout << sum;
  return 0;
}