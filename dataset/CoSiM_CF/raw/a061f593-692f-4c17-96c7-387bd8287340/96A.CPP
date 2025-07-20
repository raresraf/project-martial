#include <bits/stdc++.h>
using namespace std;

#define end1 "\n"

int main() {
  int counter = 1;
  string s;
  cin >> s;

  for (int i = 0; i < s.size() - 1; ++i) {
    if (s[i] == s[i + 1]) {
      counter++;

      if (counter == 7) {
        cout << "YES";
        return 0;
      }
    } else {
      counter = 1;
    }
  }

  cout << "NO";
}
