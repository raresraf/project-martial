#include <bits/stdc++.h>
using namespace std;

#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"

int main() {
  IOS;
  string s;
  cin >> s;

  for (int i = 0; i < s.size(); i++) {
    for (int j = i + 1; j < s.size(); j++) {
      if (s[i] == s[j]) {
        s.erase(s.begin() + j);
        j--;
      }
    }
  }

  if (s.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}
