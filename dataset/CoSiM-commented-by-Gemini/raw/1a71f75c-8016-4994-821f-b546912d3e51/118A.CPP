#include <bits/stdc++.h>
using namespace std;
int main() {
  string inp, res;
  cin >> inp;
  for (int i = 0; i < inp.size(); i++) {
    char c = tolower(inp[i]);
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
      continue;
    else {
      res += '.';
      res += c;
    }
  }
  cout << res << endl;
}
