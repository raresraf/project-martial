#include <bits/stdc++.h>
using namespace std;

int main() {
  int cnt = 0, n;
  string x;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> x;
    if (x == "++X" || x == "X++")
      cnt++;
    else if (x == "--X" || x == "X--")
      cnt--;
  }
  cout << cnt;
}