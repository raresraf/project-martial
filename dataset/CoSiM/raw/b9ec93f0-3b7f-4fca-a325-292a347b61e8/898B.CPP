#include <bits/stdc++.h>
using namespace std;

#define ll long long

int main() {

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int n, a, b, x;
  cin >> n >> a >> b;

  for (int i = 0; n >= 0; i++) {

    if ((n % b) == 0) {
      cout << "YES\n";
      cout << i << " " << n / b;
      return 0;
    }
    n = n - a;
  }

  cout << "NO";
  return 0;
}