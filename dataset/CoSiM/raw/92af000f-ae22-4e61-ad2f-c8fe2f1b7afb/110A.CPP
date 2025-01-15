#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;
string a;
int jami;
int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  cin >> a;
  for (int i = 0; i < a.size(); i++) {
    if (a[i] == '7' || a[i] == '4') {
      jami++;
    }
  }
  if (jami == 4 || jami == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
