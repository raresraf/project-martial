#include <bits/stdc++.h>

using namespace std;

int main() {
  string s, q;
  int count = 0;
  int l = 0;
  cin >> s;

  for (int i = 0; i < s.length(); i++)
    if (s[i] == '4' || s[i] == '7')
      count++;

  q = to_string(count);

  for (int i = 0; i < q.length(); i++) {
    if (q[i] == '4' || q[i] == '7')
      l = 1;
    else {
      l = 0;
      break;
    }
  }

  if (l == 1)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
