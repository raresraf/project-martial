#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, a, b, c;
  cin >> n;
  int ans = 0;

  for (int i = 0; i < n; ++i) {
    cin >> a >> b >> c;
    if (a + b + c >= 2) {
      ans++;
    }
  }

  cout << ans << endl;
}

// g++ -std=c++11 -O2 -Wall 231A.cpp -o out