

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  int s = 0;
  int k[n];
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    s = s - a;
    s = s + b;
    k[i] = s;
  }
  int max = k[0];
  for (int i = 1; i < n; i++) {
    if (k[i] > max)
      max = k[i];
  }
  cout << max;

  return 0;
}
