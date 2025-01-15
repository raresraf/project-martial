#include <bits/stdc++.h>
using namespace std;

int main() {
  string ch;
  cin >> ch;
  long long len = ch.length();
  long long total = 0;
  for (long long i = 0; i < len; i++) {
    if (ch[i] == '4' || ch[i] == '7')
      total++;
  }
  if (total == 7 || total == 4)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
