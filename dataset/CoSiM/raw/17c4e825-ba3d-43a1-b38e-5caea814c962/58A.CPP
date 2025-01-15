#include <bits/stdc++.h>
using namespace std;
int main() {
  string a;
  cin >> a;
  string b = "hello";
  int j = 0;
  for (int i = 0; i < a.length(); i++) {
    if (a[i] == b[j])
      j++;
  }
  if (j == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
