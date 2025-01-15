// Problem Link https://codeforces.com/problemset/problem/58/A
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;

  string h = "hello";

  int i = 0;
  bool y = false;
  for (char c : s) {
    if (c == h[i]) {
      i++;
    }

    if (i == h.length()) {
      y = true;
    }
  }

  if (y) {
    cout << "YES"
         << "\n";
  } else {
    cout << "NO"
         << "\n";
  }
}