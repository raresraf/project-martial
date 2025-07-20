#include <bits/stdc++.h>

using namespace std;

int main() {
  // turn off buffer sync between c and c++ streams. Decouple cin and cout
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
  freopen("error.txt", "w", stderr);
#endif

  string str;
  cin >> str;
  char prev = ' ';
  int count = 0;
  for (auto ch : str) {
    if (ch == prev) {
      ++count;
    } else {
      prev = ch;
      count = 1;
    }

    if (count == 7) {
      cout << "YES";
      return 0;
    }
  }

  cout << "NO";
  return 0;
}
