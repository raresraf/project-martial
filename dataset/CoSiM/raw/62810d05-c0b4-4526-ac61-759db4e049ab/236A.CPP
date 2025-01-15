#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  int count = 0, i;
  cin >> s;
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  // Used this trick here
  sort(s.begin(), s.end());
  for (i = 0; i < s.length(); i++) {
    if (s[i] != s[i + 1]) {
      count++;
    }
  }
  if (count % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else {
    cout << "IGNORE HIM!" << endl;
  }

  return 0;
}