// 116A. Tram Codeforce

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  int out[n], in[n], j = 0, k = 0, l = 0;

  for (int i = 0; i < n; i++) {
    cin >> out[i];
    cin >> in[i];
  }

  for (int i = 0; i < n; i++) {
    l = j - out[i] + in[i];
    j = l;
    if (l > k)
      k = l;
  }

  cout << k;

  return 0;
}