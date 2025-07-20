#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  int count = 0;

  for (int i = 0; i < s.size() - 1;) {
    int j = i + 1;
    while (s[j] == s[i] && j < s.size()) {
      count++;
      j++;
    }
    i = j;
  }
  cout << count;

  return 0;
}
