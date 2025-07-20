#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, count = 0;
  string str;

  cin >> t;

  while (t--) {
    cin >> str;

    if (str == "++X" || str == "X++")
      count++;
    else if (str == "X--" || str == "--X")
      count--;
  }
  cout << count;

  return 0;
}
