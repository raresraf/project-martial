#include <bits/stdc++.h>
using namespace std;

int main() {
  string inp;
  cin >> inp;

  int len = inp.length();
  for (int i = 0; i < len; i++) {
    char c = tolower(inp[i]);
    if (c == 'a' || c == 'e' || c == 'y' || c == 'u' || c == 'i' || c == 'o')
      continue;

    cout << '.' << c;
  }
}
