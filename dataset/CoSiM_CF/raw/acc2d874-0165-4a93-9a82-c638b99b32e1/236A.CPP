#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  int length = 0;
  cin >> s;
  int size = s.size();
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (s[i] == s[j])
        s[j] = '0';
    }
  }
  for (int k = 0; k < size; k++) {
    if (s[k] != '0')
      length += 1;
  }
  if (length % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}
