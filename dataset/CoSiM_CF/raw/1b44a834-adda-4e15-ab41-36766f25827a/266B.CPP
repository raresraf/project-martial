#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;

  string str;
  cin >> str;

  while (t--) {
    regex pattern("BG");

    str = regex_replace(str, pattern, "GB");
  }

  cout << str << endl;

  return 0;
}
