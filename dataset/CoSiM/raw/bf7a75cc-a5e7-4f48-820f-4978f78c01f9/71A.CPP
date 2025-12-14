/*
    Problem Link    :   https://codeforces.com/problemset/problem/71/A
    Contest Link    :   https://codeforces.com/contest/71/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    if (s.length() > 10) {
      string ans;
      ans.push_back(s.front());
      ans += to_string(s.length() - 2);
      ans += s.back();
      cout << ans << endl;

      // or you can do it like cout<<s[0]<<s.size()-2<<s[s.size()-1]<<endl;
    } else
      cout << s << endl;
  }
  return 0;
}