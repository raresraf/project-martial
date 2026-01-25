#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  cin >> n;
  int res = 0;
  int t;
  for (int i = 0; i < n; i++) {
    int a = 0;
    for (int j = 0; j < 3; j++) {
      cin >> t;
      a += t;
    }
    if (a > 1)
      res++;
  }
  cout << res << "\n";
  return 0;
}
