#include <bits/stdc++.h>
using namespace std;
int main() {
  int a, b, c, t;
  cin >> t;
  for (int i = 0; i < t; i++) {
    cin >> a >> b;
    int ans = (a + b) / 3;
    int minimum = min(ans, min(a, b));
    cout << minimum << endl;
  }
}