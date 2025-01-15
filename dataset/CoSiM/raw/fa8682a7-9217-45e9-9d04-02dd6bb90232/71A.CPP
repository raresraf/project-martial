#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  while (n--) {
    string x;
    cin >> x;
    if (x.length() > 10) {
      string y = x[0] + to_string((x.length() - 2)) + x[x.length() - 1];
      cout << y << endl;
    } else
      cout << x << endl;
  }
  return 0;
}