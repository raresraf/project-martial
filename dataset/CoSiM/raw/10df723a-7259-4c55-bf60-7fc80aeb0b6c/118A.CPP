// link : https://codeforces.com/problemset/problem/118/A

#include <bits/stdc++.h>

using namespace std;

string s1, s;

main() {

  cin >> s;

  for (int i = 0; i < s.size(); i++) {

    s[i] = tolower(s[i]);

    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'y' ||
        s[i] == 'o' || s[i] == 'u')
      continue;
    else
      cout << "." << s[i];
  }
}
