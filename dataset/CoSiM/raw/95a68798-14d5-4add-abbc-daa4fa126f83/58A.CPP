#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  string str;
  cin >> str;

  string s;
  for (int i = 0; i < str.size(); i++) {
    if (s.empty() && str[i] == 'h')
      s.push_back('h');

    else if (s == "h" && str[i] == 'e')
      s.push_back('e');

    else if (s == "he" && str[i] == 'l')
      s.push_back('l');

    else if (s == "hel" && str[i] == 'l')
      s.push_back('l');

    else if (s == "hell" && str[i] == 'o')
      s.push_back('o');
  }

  (s == ("hello")) ? cout << "YES\n" : cout << "NO\n";
}