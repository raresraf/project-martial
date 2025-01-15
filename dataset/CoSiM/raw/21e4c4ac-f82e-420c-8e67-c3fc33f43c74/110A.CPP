#include <bits/stdc++.h>
using namespace std;
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

int main() {
  fast string s;
  int cnt = 0;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '4' || s[i] == '7')
      cnt++;
  }
  if (cnt == 4 || cnt == 7)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
