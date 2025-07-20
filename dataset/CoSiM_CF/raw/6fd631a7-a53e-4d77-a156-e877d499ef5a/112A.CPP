#include <bits/stdc++.h>
using namespace std;
int main() {
  string s1, s2;
  cin >> s1 >> s2;
  int l = s1.length();
  for (int i = 0; i < l; i++) {
    s1[i] = toupper(s1[i]);
    s2[i] = toupper(s2[i]);
  }
  int val = s1.compare(s2);
  cout << val << endl;
}
