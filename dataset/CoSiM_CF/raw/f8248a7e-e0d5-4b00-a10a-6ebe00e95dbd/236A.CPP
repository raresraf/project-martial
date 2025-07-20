#include <bits/stdc++.h>

using namespace std;

int main() {
  string s1;
  cin >> s1;
  int k = 0, l = 0;
  for (int i = 0; i < s1.length(); i++) {
    for (int j = i + 1; j < s1.length(); j++)
      if (s1[i] == s1[j])
        l = 1;
    if (l == 1) {
      l = 0;
      continue;
    }
    k++;
  }
  if (k % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}