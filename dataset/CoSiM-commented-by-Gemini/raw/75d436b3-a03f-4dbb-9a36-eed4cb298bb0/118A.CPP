#include <bits/stdc++.h>

using namespace std;

void task(string &s);

int main() {
  string s;

  cin >> s;
  task(s);

  return 0;
}

void task(string &s) {
  string result = {};

  for (int i = 0; i < s.size(); ++i) {
    // ambas las convertimos en minusculas
    if (s[i] < 97) {
      s[i] += 32;
    }

    if (s[i] == 'a' || s[i] == 'o' || s[i] == 'y' || s[i] == 'e' ||
        s[i] == 'u' || s[i] == 'i') {
      continue;
    } else {
      result.push_back('.');
      result.push_back(s[i]);
    }
  }

  cout << result;
}