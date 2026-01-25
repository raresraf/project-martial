#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  int ck = 0;
  cin >> str;
  sort(str.begin(), str.end());
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == '+')
      continue;
    if (ck != 0)
      cout << "+";
    cout << str[i];
    ck = 1;
  }
  cout << endl;
  return 0;
}
