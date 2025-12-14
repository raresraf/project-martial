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
  if ((str.find("1111111") != string::npos) ||
      (str.find("0000000") != string::npos))
    cout << "YES";
  else
    cout << "NO";

  cerr << "time taken: " << (float)clock() / CLOCKS_PER_SEC << " secs" << endl;
  return 0;
}
