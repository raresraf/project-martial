#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);

  long long t, x = 0;
  string str;
  cin >> t;
  for (int i = 0; i <= t; i++) {
    getline(cin, str);
    if (str == "++X") {
      ++x;
    } else if (str == "X++") {
      x++;
    } else if (str == "--X") {
      --x;
    } else if (str == "X--") {
      x--;
    }
  }
  cout << x;

  return 0;
}
