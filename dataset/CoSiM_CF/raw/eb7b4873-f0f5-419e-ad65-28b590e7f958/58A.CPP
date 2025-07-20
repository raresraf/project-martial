#include <bits/stdc++.h>

using namespace std;

int main() {

  char n, i = 0;
  string hello = "hello";

  while (cin >> n) {
    if (n == hello[i]) {
      i++;
    }
  }
  cout << (i == (int)hello.length() ? "YES" : "NO");
}