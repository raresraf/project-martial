#include <bits/stdc++.h>
using namespace std;

int main() {
  string c;
  int count = 1;
  cin >> c;
  for (int i = 1; i < c.length(); ++i) {
    if (c[i] == c[i - 1]) {
      count++;
      if (count == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else
      count = 1;
  }
  cout << "NO" << endl;

  return 0;
}
