#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, s, in[100], i, player = 0;
  cin >> k >> n;
  for (i = 0; i < k; i++) {
    cin >> in[i];
  }
  s = in[n - 1];
  for (i = 0; i < k; i++) {
    if (in[i] >= s && in[i] != 0) {
      player++;
    }
  }
  cout << player;
  return 0;
}
