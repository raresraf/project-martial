#include <bits/stdc++.h>
using namespace std;
#define ll long long

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int n, a, b, c;
  cin >> n;
  int res = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if (a + b + c > 1)
      res++;
  }
  cout << res;
}
