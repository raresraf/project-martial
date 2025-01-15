#include <bits/stdc++.h>
using ll = long long;
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  string str;
  cin >> str;

  int cnt = count_if(str.begin(), str.end(),
                     [](char a) { return a == '4' || a == '7'; });

  if (cnt == 4 || cnt == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
