// link : https://codeforces.com/contest/96/problem/A

#include <bits/stdc++.h>

using namespace std;

string s;

main() {

  cin >> s;

  if (s.size() >= 7) {

    for (int i = 0; i < s.size() - 6; i++) {
      //	cout<<"1";
      int k = 0;
      char r = s[i];

      for (int j = i; j < i + 7; j++) {

        if (r != s[j]) {
          k++;
        }
      }

      if (k == 0) {
        cout << "YES";
        return 0;
      }
    }
  }

  cout << "NO";
  return 0;
}
