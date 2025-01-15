#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  int size = 0, count = 0;
  cin >> s;
  size = s.size();

  for (int i = 0; i < size; i++) {
    if (s[i] == s[i + 1]) {
      count++;
      if (count >= 6) {
        cout << "YES";
        return 0;
      }
    } else
      count = 0;
  }
  cout << "NO";
}
