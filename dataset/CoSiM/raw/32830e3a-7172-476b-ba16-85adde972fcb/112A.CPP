#include <algorithm>
#include <bits/stdc++.h>
using namespace std;
void lower(string &s) {
  int n = s.length();
  for (int i = 0; i < n; i++) {
    // cout<<isupper(s[i]);
    if (isupper(s[i])) {
      s[i] = tolower(s[i]);
      // cout<<"s[i]="<<s[i]<<endl;
    }
  }
  // cout<<"After:"<<s<<endl;
}

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  lower(s1);
  lower(s2);
  // cout<<s1<<endl;
  //  cout<<s2<<endl;
  if (s1 == s2)
    cout << 0;
  else if (s1 < s2)
    cout << -1;
  else
    cout << 1;

  return 0;
}
