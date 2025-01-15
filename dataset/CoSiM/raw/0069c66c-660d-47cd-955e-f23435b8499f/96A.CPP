#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  int contigua = 1;

  for (size_t i = 1; i < s.length(); i++) {
    if (s[i] == s[i - 1]) {
      contigua += 1;
      if (contigua == 7) {
        cout << "YES";
        return 0;
      }
    } else {
      contigua = 1;
    }
  }

  cout << "NO";

  return 0;
}
