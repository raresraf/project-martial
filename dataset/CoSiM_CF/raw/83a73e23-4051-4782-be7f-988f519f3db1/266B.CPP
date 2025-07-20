#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int n, time;
  cin >> n >> time;
  string s;
  cin >> s;
  for (int i = 0; i < time; i++) {
    for (int j = 0; j < s.size(); j++) {
      if (s[j] == s[j + 1])
        continue;
      else if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        j = j + 1;
      }
    }
  }
  cout << s;

  return 0;
}