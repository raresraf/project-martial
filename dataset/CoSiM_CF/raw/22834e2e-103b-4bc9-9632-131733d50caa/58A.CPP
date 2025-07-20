#include <iostream>
#include <vector>
#define pb push_back
using namespace std;
int main() {
  string s;
  cin >> s;
  vector<int> v1, v2, v3, v4;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'h')
      v1.pb(i);
    else if (s[i] == 'e')
      v2.pb(i);
    else if (s[i] == 'l')
      v3.pb(i);
    else if (s[i] == 'o')
      v4.pb(i);
  }
  if (v1.size() == 0 || v2.size() == 0 || v3.size() == 0 || v4.size() == 0)
    cout << "NO\n";
  else {
    int index = v1[0];
    int flag = 0;
    // cout<<index<<endl;
    for (int i = 0; i < v2.size(); i++) {
      if (index < v2[i]) {
        index = v2[i];
        flag = 1;
        break;
      }
    }
    // cout<<index<<endl;
    int j = 0;
    for (int i = 0; i < v3.size(); i++) {
      if (j == 2) {
        flag = 1;
        break;
      }
      if (index < v3[i] && flag == 1) {
        index = v3[i];
        j++;
      }
    }
    // cout<<index<<endl;
    if (j != 2)
      flag = 0;
    int flag2 = -1;
    for (int i = 0; i < v4.size(); i++) {
      if (index < v4[i] && flag == 1) {
        index = v4[i];
        flag = 1;
        flag2 = 1;
        break;
      }
    }
    // cout<<index<<endl;
    if (flag == 0 || flag2 == -1)
      cout << "NO\n";
    else
      cout << "YES\n";
  }
}